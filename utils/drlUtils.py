# ====================================================================================
# Functions related to running deep reinforcement learning.
# ====================================================================================
import numpy as np
import pandas as pd
import os
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import threading
from time import sleep, asctime
from tqdm import tqdm
import tensorflow as tf
from itertools import product

sns.set(style="white")
sys.path.append("./")
from myUtils import mkdir
from LotkaVolterraModel import LotkaVolterraModel, ExponentialModel, StemCellModel

# ====================================================================================
def update_target_graph(from_scope,to_scope):
    '''
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    '''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# ====================================================================================
def normalized_columns_initializer(std=1.0):
    '''
    Used to initialize weights for policy and value output layers.
    '''
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# ====================================================================================
from drlModel import A3C_Network
from LotkaVolterraModel import LotkaVolterraModel
class Worker():
    def __init__(self, name, master_network, trainer, saver, model_path, global_episodes,
                 logging_interval=50, verbose=2):
        # General admin variables
        self.name = 'Worker_' + str(name)
        self.number = name
        self.model_path = model_path
        self.master_network = master_network
        self.trainer = trainer
        self.saver = saver
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.n_inputs = master_network.n_inputs
        self.verbose = verbose
        self.logging_interval = logging_interval

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = A3C_Network(architecture_kws=master_network.architecture_kws,
                                    scope=self.name, trainer=trainer, global_episodes=global_episodes)
        self.update_local_ops = update_target_graph(from_scope='global', to_scope=self.name)

        self.time_of_last_treatment = 0
        self.previous_state = np.zeros(master_network.n_inputs)

    def work(self, session, coord, model_name, training_patients_df, updating_interval=7, max_epochs=np.inf):
        '''
        Work function corrected to give correct input of 7 previous tumour sizes and deltas.
        '''
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.model_path, model_name, "worker_"+str(self.number)))
        writer = tf.summary.FileWriter(os.path.join(self.model_path, model_name), session.graph)
        n_trainingPatients = training_patients_df.shape[0]
        action_history = {}
        high_score = 0
        patients_treated = 0
        total_chemo_cycles = 0
        start_time = datetime.datetime.now()
        if self.verbose>0: logging.info("Starting worker " + str(self.number))
        with session.as_default(), session.graph.as_default():

            while not coord.should_stop():
                session.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                patient_reward = 0
                patient_chemo_cycles = 0

                # Initialize the ODE model instance & parameterize it
                model = LotkaVolterraModel(method='RK45', dt=updating_interval)
                currParamsDic = training_patients_df.iloc[np.random.randint(0, n_trainingPatients)].to_dict()
                model.SetParams(**currParamsDic)

                # Book keeping variables
                done = False
                num_treatments = 0
                patient_action_history = ""
                n0 = currParamsDic['n0']
                # Loop til finished
                t_start = 0
                t_end = 0

                while not done:
                    # Set the new end time
                    t_start = t_end
                    t_end += updating_interval
                    num_treatments += 1  # Increment the treatment counter

                    # Get the current state to provide as input to the DRL model
                    # 1. Tumor sizes
                    # Get the state as the initial tumor size and the tumor sizes each day for the past week.
                    # In case this is the start of the treatment cycle, the input vector is just the
                    # initial tumor size and the 0's.
                    results = model.resultsDf
                    observations_sizes = []
                    n_values_size = self.local_AC.architecture_kws['n_values_size']
                    for i in range(n_values_size):
                        try:
                            # IF not, then the observation is the last n_values_size time steps
                            observations_sizes.append(results['TumourSize'].iloc[-(i + 1)])
                        except TypeError:
                            # resultsDF doesn't exist b/c we haven't simulated anything
                            observations_sizes.append(n0)
                        except IndexError:
                            # we have not yet reached n_values_size time steps
                            observations_sizes.append(0)

                    observations_sizes = np.array(observations_sizes)/n0

                    # In addition, the network uses the per-day changes for each day over the past week,
                    # which we compute here.
                    observations_deltas = []
                    n_values_delta = self.local_AC.architecture_kws['n_values_delta']
                    for i in range(n_values_delta):
                        observations_deltas.append(observations_sizes[i + 1] - observations_sizes[i])
                    observations_deltas = np.array(observations_deltas)

                    network_input = np.concatenate([observations_sizes, observations_deltas])
                    network_input = np.expand_dims(network_input, axis=2) # Add a fake dimension to make it compatible with the LSTM layer which needs 3 dims
                    assert not np.any(np.isnan(network_input)), "Invalid input occured"


                    # Take an action using probabilities from policy network output.
                    a_dist, v = session.run([self.local_AC.policy, self.local_AC.value],
                                         feed_dict={self.local_AC.input: [network_input]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # Treat the patient and observe the response
                    if a == 0:
                        # Treat with 0% strength (holiday)
                        model.Simulate([[t_start, t_end, 0]], scaleTumourVolume=False)
                        patient_action_history += "H"
                    else:
                        model.Simulate([[t_start, t_end, model.paramDic['DMax']]], scaleTumourVolume=False)
                        patient_action_history += "T"

                    # Calculate the reward
                    results = model.resultsDf
                    reward, done = self.master_network.calculate_reward(state=results, n0=n0, time=t_end, action=a)
                    
                    # Record this iteration
                    patient_reward += reward
                    self.previous_state = network_input
                    patient_chemo_cycles += 1
                    # episode_buffer.append([network_input,a,reward,s1,done,v[0,0]])
                    episode_buffer.append([network_input, a, reward, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    if done:
                        if high_score == -1:
                            high_score = reward
                        if reward > high_score:
                            high_score = reward
                        patients_treated += 1
                        current_ttp = model.resultsDf.Time.iloc[-1]
                        session.run(self.master_network.increment_patients)
                        logging.debug("%i patients treated. Total treatments: %i x %i = %i. Total score: %f" % (
                        session.run(self.master_network.global_patients),
                        num_treatments,
                        updating_interval,
                        num_treatments * updating_interval,
                        patient_reward))
                        break

                # Update the network using the experience buffer at the end of the episode.
                num_treatments_this_patient = num_treatments  # len(patient_action_history)
                v_l, p_l, e_l, g_n, v_n = self.train(global_AC=self.master_network, 
                                                    rollout=episode_buffer, 
                                                    sess=session, 
                                                    bootstrap_value=0.0,
                                                    patient_id=session.run(self.master_network.global_patients),
                                                    current_ttp=current_ttp,
                                                    num_treatments_this_patient=num_treatments_this_patient,
                                                    pat_act_hist=patient_action_history, 
                                                    model_name=model_name)
                
                # Log progress                
                action_history[patients_treated] = {"Treatments": patient_action_history,
                                                    "Survival time": patient_reward}
                total_chemo_cycles += 1
                self.episode_rewards.append(patient_reward)
                self.episode_lengths.append(patient_chemo_cycles)
                self.episode_mean_values.append(np.mean(episode_values))
                mean_reward = np.mean(self.episode_rewards[-5:])
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_value = np.mean(self.episode_mean_values[-5:])

                summary = tf.Summary()
                summary.value.add(tag='HighScore', simple_value=float(high_score))
                summary.value.add(tag='TumorReduction', simple_value=float(reward))
                summary.value.add(tag='Therapy/Lifetime', simple_value=float(patient_reward))
                summary.value.add(tag='Therapy/Num_treatments', simple_value=float(num_treatments))
                self.summary_writer.add_summary(summary, patients_treated)

                # Log the performance of this worker
                if patients_treated % self.logging_interval == 0:
                    self.summary_writer.flush()
                    writer.flush()
                    action_frame = pd.DataFrame.from_dict(action_history, orient='index')
                    action_frame.to_csv(os.path.join(self.model_path, model_name, "treatment_history_core-" + self.name + ".csv"))

                # Save a copy of the global network at regular intervals
                num_patients = int(session.run(self.master_network.global_patients))
                if num_patients % self.logging_interval == 0:
                    logging.debug("%i patients treated. Most recent TTP: %d" % (
                        num_patients,
                        current_ttp))
                    # Network
                    save_dir = self.saver.save(session,
                                        os.path.join(self.model_path, model_name, str(num_patients) + "_patients_" + str(model_name),
                                                    "a3c_chemo_core.cpkt"))

                    # Summary statistics
                    treatment_frame = pd.DataFrame(self.master_network.patient_treatment_numbers)
                    treatment_frame.to_csv(os.path.join(self.model_path, model_name, str(num_patients) + "_patients_" + str(model_name),
                                                        "patient_treatments_" + str(num_patients) + ".csv"))
                    # Configurations
                    paramDic={**self.master_network.architecture_kws, **self.master_network.reward_kws, **model.paramDic}
                    with open(os.path.join(self.model_path, model_name, str('paramDic_' + model_name + '.txt')), 'w') as f:
                        for key in paramDic.keys():
                            f.write("%s:%s\n" % (key, paramDic[key]))
                now = datetime.datetime.now()
                
                # Stop if have run more than the pre-set number of epochs
                num_global = int(session.run(self.master_network.global_episodes))
                if num_global > max_epochs: coord.request_stop()


    def train(self, global_AC, rollout, sess, bootstrap_value, patient_id, current_ttp, num_treatments_this_patient,
              pat_act_hist, model_name=0, trained_frame_name=None):

        rollout = np.array(rollout)  # https://www.quora.com/What-is-rollout-in-machine-learning
        # print(rollout.shape)
        observations = rollout[:, 0]
        # print(observations.shape)
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 4]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        gamma = self.master_network.reward_kws['gamma']
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.master_network.discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.master_network.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.input: np.stack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        sess.run(global_AC.increment)
        self.master_network.patient_treatment_numbers.append({"PatientId": patient_id,
                                                         "Treatments": num_treatments_this_patient,
                                                         "Total_Reward": discounted_rewards[0],
                                                         "TTP": current_ttp,
                                                         "actions": str(pat_act_hist),
                                                         "Model_Name": model_name,
                                                         "trained_frame_name": trained_frame_name})
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

# ====================================================================================
def run_training(training_patients_df=None, architecture_kws={}, reward_kws={},
                learning_rate=1e-4, updating_interval=7, num_workers=2, max_epochs=np.inf, model_name=None,
                load_model=False, model_loaded_name=None, model_path="./", logging_interval=1000, verbose=1):
    '''
    Function to train the DRL network.
    :param training_patients_df: dataframe with model parameters (ODE or ABM). Each row is assumed to be a different patient and the model will randomly iterate through these during training.
    :param architecture_kws: dictionary with variables defining the network architecture
    :param reward_kws: dictionary with variables defining the reward function
    :param learning_rate: learning rate used in Adam optimizer
    :param interval: updating interval at which a new DRL treatment decision is made
    :param num_workers: number of independent workers used for training
    :param model_name: identifier used for naming model log files etc
    :param load_model: boolean for whether to start with pre-trained model
    :param model_loaded_name: name of the directory holding the pre-trained model's cpkt files
    :param model_path: path of model where the pre-trained model's directory can be found
    :param logging_interval: will save the state of the network every logging_interval patients
    :param verbose: verboseness level
    '''
    # Setup the environment
    mkdir(model_path)
    model_name = asctime().replace(' ','').replace(':','') if model_name is None else model_name
    mkdir(os.path.join(model_path, model_name))

    # Setup output for logging file
    logging.basicConfig(filename=os.path.join(model_path, model_name, 'trainingLog_' + model_name +'.log'),
                        filemode='w+', level=logging.DEBUG,
                        format=('%(asctime)s - %(name)s'
                                + '- %(levelname)s - %(message)s'))

    # Set up A3C network, which consists of one global network, and num_workers copies
    # which are used during training to increase learning performance.
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Generate global network
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate) #, epsilon = 1e-01) #TODO: move defining of default training params here & fix passing of reward parameters (so that they're actually passed)
        master_network = A3C_Network(architecture_kws=architecture_kws, reward_kws=reward_kws,
                                    scope='global', trainer=None, global_episodes=global_episodes)
        workers = []
        # Create worker classes
        saver = tf.train.Saver(max_to_keep=None)
        for i in range(num_workers):
            workers.append(Worker(name=i, master_network=master_network,
                                trainer=trainer, saver=saver, model_path=model_path, 
                                global_episodes=global_episodes, logging_interval=logging_interval, verbose=verbose))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        # Initialize model
        sess.run(tf.global_variables_initializer())
        if load_model:
            logging.info('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(os.path.join(model_path, model_loaded_name))
            saver.restore(sess, ckpt.model_checkpoint_path)

        # Train the model
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            param_frame = None
            worker_work = lambda: worker.work(session=sess, coord=coord, model_name=model_name, 
                                             training_patients_df=training_patients_df, 
                                             updating_interval=updating_interval, max_epochs=max_epochs) 
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)

# ====================================================================================
def run_evaluation(model_path, patients_to_evaluate, architecture_kws={}, n_replicates=100, updating_interval=7, results_path="./", results_file_name="results.csv",
                   ODE_model=LotkaVolterraModel, tqdm_output=sys.stderr, verbose=0, seed=42):
    '''
    Evaluate the model by simulating a patient drug administration policy chosen by the DRL.
    :param model_path: path to the model directory
    :param architecture_kws: dictionary with variables defining the network architecture
    :param results_path: path to directory to save the results
    :param model_loaded_name: name of the model to load
    :param n_replicates: number of replicates to simulate
    :param paramDic: dictionary of parameters for the model
    :param verbose: verbosity level
    :return:
    '''
    # Setup env and parse input
    mkdir(results_path);
    np.random.seed(seed)
    tf.set_random_seed(seed)
    if isinstance(patients_to_evaluate, pd.DataFrame): 
        patientsDf = patients_to_evaluate
    elif isinstance(patients_to_evaluate, dict): 
        patientsDf = pd.DataFrame.from_dict(patients_to_evaluate)

    if isinstance(ODE_model, str):
        ODE_model = {'LotkaVolterraModel': LotkaVolterraModel,
                     'ExponentialModel': ExponentialModel,
                     'StemCellModel': StemCellModel}[ODE_model]
    
    # Set up A3C network, which consists of one global network, and num_workers copies
    # which are used during training to increase learning performance.
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Generate global network
        master_network = A3C_Network('global', None, architecture_kws=architecture_kws)
        saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

        # Create a data log of the input vectors and the decision results
        timestep_summmaryList = []

        # Evaluate it
        if isinstance(tqdm_output, str): tqdm_output = open(tqdm_output, 'w')
        for patientId, replicateId in tqdm(list(product(patientsDf.PatientId.unique(),np.arange(n_replicates))), file=tqdm_output):
            # Initialize the ODE model instance & parameterize it
            model = ODE_model(method='RK45', dt=updating_interval)

            # Load the parameter set for this virtual patient from the database with the evaluation parameter sets.
            currParamsDic = patientsDf.loc[patientsDf.PatientId==patientId].iloc[0].to_dict()
            model.SetParams(**currParamsDic)
            n0 = currParamsDic['n0']

            # Book keeping variables
            done = False
            num_treatments = 0
            patient_action_history = ""
            currPatient_reward = 0
            currPatient_nCycles = 0
            t_start = 0
            t_end = 0
            a = 0

            # This is the treatment loop
            while not done:
                # Set the new end time
                t_start = t_end

                t_end += updating_interval  # Regular analysis
                # t_end += updating_interval + currParamsDic['sigma']  # For benchmarking tau and sigma sens analysis
                # t_end += updating_interval + int(np.random.exponential(scale=currParamsDic['sigma']))
                num_treatments += 1  # Increment the treatment counter

                # Get the current state to provide as input to the DRL model
                # 1. Tumor sizes
                # Get the state as the initial tumor size and the tumor sizes each day for the past week.
                # In case this is the start of the treatment cycle, the input vector is just the
                # initial tumor size and the 0's.
                results = model.resultsDf
                observations_sizes = []
                n_values_size = master_network.architecture_kws['n_values_size']
                for i in range(n_values_size):
                    try:
                        # IF not, then the observation is the last n_values_size time steps
                        observations_sizes.append(results['TumourSize'].iloc[-(i + 1)])
                    except TypeError:
                        # resultsDF doesn't exist b/c we haven't simulated anything
                        observations_sizes.append(n0)
                    except IndexError:
                        # we have not yet reached n_values_size time steps
                        observations_sizes.append(0)

                observations_sizes = np.array(observations_sizes)/n0

                # In addition, the network uses the per-day changes for each day over the past week,
                # which we compute here.
                observations_deltas = []
                n_values_delta = master_network.architecture_kws['n_values_delta']
                for i in range(n_values_delta):
                    observations_deltas.append(observations_sizes[i + 1] - observations_sizes[i])
                    
                network_input = np.concatenate([observations_sizes, observations_deltas]) 
                network_input = np.expand_dims(network_input, axis=2) # Add a fake dimension to make it compatible with the LSTM layer which needs 3 dims
                assert not np.any(np.isnan(network_input))

                # Take an action using probabilities from policy network output.
                a_dist, v = sess.run([master_network.policy, master_network.value],
                                        feed_dict={master_network.input: [network_input]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

                # Treat the patient and observe the response
                if a == 0:
                    # Treat with 0% strength (holiday)
                    model.Simulate([[t_start, t_end, 0]], scaleTumourVolume=False)
                    patient_action_history += "H"
                else:
                    model.Simulate([[t_start, t_end, model.paramDic['DMax']]], scaleTumourVolume=False)
                    patient_action_history += "T"

                # Calculate the reward
                results = model.resultsDf
                reward, done = master_network.calculate_reward(state=results, n0=n0, time=t_end, action=a)
                
                # Record this iteration
                currPatient_reward += reward
                currPatient_nCycles += 1
                currTumourState = results.loc[results.Time == t_start]
                timestep_summmaryList.append({'ReplicateId': replicateId, 'Time': t_start,
                                              'TumourSize': currTumourState['TumourSize'].values[0],
                                              'S': currTumourState['S'].values[0],
                                              'R': currTumourState['R'].values[0],
                                              'DrugConcentration': currTumourState['DrugConcentration'].values[0],
                                              'Support_Hol': a_dist[0][0], 'Support_Treat': a_dist[0][1],
                                              'Action': patient_action_history[-1],
                                              'PatientId': patientId})
                if verbose > 0: print("Replicate ID: " + str(replicateId) +
                                      " - Treating interval %i to %i" % (t_start, t_end) +
                                      " - Choice is: " + str(a) +
                                      " - Current size is: " + str(round(results['TumourSize'].iloc[-1], 3)))

                if done:  # Record final timestep where progression has occured
                    finTumourState = results.iloc[-1].to_frame().transpose()
                    timestep_summmaryList.append({'ReplicateId': replicateId, 'Time': t_end,
                                                 'TumourSize': finTumourState['TumourSize'].values[0],
                                                 'S': finTumourState['S'].values[0],
                                                 'R': finTumourState['R'].values[0],
                                                 'DrugConcentration': finTumourState['DrugConcentration'].values[0],
                                                 'Support_Hol': a_dist[0][0], 'Support_Treat': a_dist[0][1],
                                                 'Action': patient_action_history[-1],
                                                 'PatientId': patientId})

        # Save the results
        longitudinalTrajectoriesDf = pd.DataFrame(timestep_summmaryList)
        longitudinalTrajectoriesDf.to_csv(
            os.path.join(results_path, results_file_name))