# ====================================================================================
# Functions related to running deep reinforcement learning.
# ====================================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import keras
import scipy.signal
import logging
import sys
sys.path.append("./")
from drlUtils import normalized_columns_initializer

# ====================================================================================
class A3C_Network():
    '''
    Global A3C network LOOSELY based on the guide in
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
    '''
    def __init__(self, scope, trainer, 
                 architecture_kws={'n_values_size':1, 'n_values_delta':0, 'n_inputs':1, 'architecture':[128, 64, 32, 16, 10], 'n_doseOptions':2},
                 reward_kws={'gamma':0.9999, 'base':0.1, 'hol':0.05, 'punish':-.1, 'drug_use':0},
                 global_episodes=tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)):
        # Setup the DRL architecture and reward function. Because I'm using dictionaries to keep this information
        # the default values in the function declaration may be lost if the user provides only one of arguments (e.g. n_values_size).
        # To ensure everything is intiialized as desired, assign default values here if none have been provided
        defaults_architecture = {
            'n_values_size':1, # Number of prior tumor sizes provided as inputs
            'n_values_delta':0, # Number of prior tumor size changes provided as inputs
            'n_inputs':1, # Total number of inputs. Provided separately in case we want to play with additional variables as input (e.g. initial growth rate)
            'architecture':[128, 64, 32, 16, 10], # Size of hidden layers
            'n_doseOptions':2 # Agent can choose 2 different intensities of chemotherapy
        }
        for key in defaults_architecture.keys():
            architecture_kws[key] = architecture_kws.get(key, defaults_architecture[key])
        # Reward funtion
        defaults_reward = {
            'gamma':0.9999, # Discount rate for advantage estimation and reward discounting
            'base':0.1, # Base reward given
            'hol':0.05, # Addition for a treatment holiday
            'punish':-.1, # Punishment for tumor progression within simulation time frame
            'drug_use':0 # Reward for "correct" drug use
        }
        for key in defaults_reward.keys():
            reward_kws[key] = reward_kws.get(key, defaults_reward[key])
        self.reward_kws = reward_kws

        with tf.variable_scope(scope):
            self.num_patients_treated = tf.constant(0)

            # Setup network architecture
            # 1. Input layers
            self.architecture_kws = architecture_kws
            self.n_inputs = architecture_kws['n_inputs']
            self.input = tf.placeholder(shape=[None, self.n_inputs, 1], dtype=tf.float32, name="%s_input" %(scope))
            self.firstLSTM = keras.layers.LSTM(self.n_inputs)
            self.LSTM_output = self.firstLSTM(self.input)

            # 2. Hidden layers
            self.architecture = architecture_kws['architecture']
            self.hidden = slim.fully_connected(self.LSTM_output, self.architecture[0], activation_fn = tf.nn.relu)
            for size in self.architecture[1:]:
                self.hidden = slim.fully_connected(self.hidden, size, activation_fn = tf.nn.relu)

            # 3. Output layers for policy and value estimations
            self.n_doseOptions = architecture_kws['n_doseOptions']
            self.policy = slim.fully_connected(self.hidden,self.n_doseOptions,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(self.hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,self.n_doseOptions,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-8))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = .25*(0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01)

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
            else:
                self.global_episodes = global_episodes # TODO: Where does this feed in? Added it as an argument. Needs to be fixed.
                self.increment = self.global_episodes.assign_add(1)
                self.global_patients = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
                self.increment_patients = self.global_patients.assign_add(1)
                self.patient_treatment_numbers = []

    def load_existing_network(self, file_path, sess):
        logging.info("\nLoading existing model...")

        saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(file_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        logging.info("Model loaded.\n")

    # Calculate a simple survival reward
    def calculate_reward(self, state, n0, time, action):
        tumor_size = state['TumourSize'].iloc[-1]
        done = False
        reward = self.reward_kws['base']
        if tumor_size > 1.2 * n0:
            done = True
            reward = self.reward_kws['punish']
            # print("Punished for tumor size " + str(tumor_size))
        elif time > 20 * 365:
            done = True
            reward += 5
        if not done:
            if tumor_size > n0 and action == 0:  # Punishment for near failure
                reward += self.reward_kws['punish'] 
            if action == 0:  # Holiday Reward
                reward += self.reward_kws['hol']
        return reward, done

    # Discounting function for the reward
    def discount(self, x, gamma):
        # This is what the lfilter function does:
        # discounted_rewards = []
        # k = 0
        # for reward in x:
        #     discounted_rewards.append(reward * gamma**k)
        #     k += 1
        # running_total = 0
        # for reward in discounted_rewards[::-1]:
        #     running_total += reward * gamma**k
        #     print(running_total)
        #     k -= 1

        # print(scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1])
        # return None
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]