'''
Script to train personalised models 
'''
import os, sys
from time import sleep, asctime
import pandas as pd
import multiprocessing

sys.path.append("utils")
from myUtils import convert_ode_parameters
from drlUtils import run_training

if __name__ == '__main__':
    # Environment parameters
    # DRL parameters
    gamma = 0.9999  # Discount rate for advantage estimation and reward discounting
    n_doseOptions = 2  # Agent can choose 2 different intensities of chemotherapy
    base = 0.1  # Base reward given
    hol = 0.05  # Addition for a treatment holiday
    punish = -1  # Punishment when tumor progresses
    day_interval = 30  # Interval between decision points (time step)
    learning_rate = 1e-4  # Learning rate
    num_workers = 6  #multiprocessing.cpu_count() # Set workers to number of available CPU threads
    curr_epochs = 100000  
    model_path = "models"
    model_name = "test_currSizeOnly_p25_step1"
    load_model = True
    logging_interval = 5000  # Will save the state of the network every logging_interval patients
    model_loaded_name = os.path.join(model_name, "%d_patients_%s"%(curr_epochs, model_name))
    verbose = 2
    
    max_epochs = curr_epochs + 50000
    trainingDataDf = pd.read_csv("models/truncPatientsDf_bruchovsky.csv", index_col=0)

    def train(id):
        patients_df = trainingDataDf[trainingDataDf.PatientId==id]
        model_name = "test_currSizeOnly_step2_trunc_params_id%s"%(int(id))
    
        # Run training
        run_training(training_patients_df=patients_df,
                    architecture_kws={'n_values_size':1, 'n_values_delta':0, 'architecture':[128, 64, 32, 16, 10], 'n_doseOptions':n_doseOptions},
                    reward_kws={'gamma':gamma, 'base':base, 'hol':hol, 'punish':punish},
                    learning_rate=learning_rate, updating_interval=day_interval, num_workers=num_workers, max_epochs=max_epochs, model_name=model_name,
                    load_model=load_model, model_loaded_name=model_loaded_name, model_path=model_path, logging_interval=logging_interval, verbose=verbose)

    # Parallel approach to training on multiple patients - OS dependent
    def main():
        pool = multiprocessing.Pool(processes=7)
        zip(*pool.map(train, trainingDataDf.PatientId.unique()))

    main()

    # Back-up serial approach is the parallel approach fails
    # for id in trainingDataDf.PatientId.unique():
    #     train(id)
