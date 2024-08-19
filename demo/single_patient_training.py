'''
Script to train model with only the current tumor size as input.
Run from the top level directory from virtual environment with drl_env.yml installed. (see [SetUp](../README.md#setup))
'''
import os
import sys
from time import sleep, asctime
import pandas as pd

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
    punish = -0.5  # Punishment when tumor progresses
    day_interval = 30  # Interval between decision points (time step)
    learning_rate = 1e-4  # Learning rate
    num_workers = 8  #multiprocessing.cpu_count() # Set workers to number of available CPU threads
    max_epochs = 100000
    model_path = "models"  # Path to save model
    model_name = "test_currSizeOnly_p25_monthly"
    load_model = False
    logging_interval = 20000  # Will save the state of the network every logging_interval patients
    model_loaded_name = None
    verbose = 2
    
    # Tumour model parameters
    trainingDataDf = pd.read_csv("models/trainingPatientsDf_bruchovsky.csv", index_col=0)
    trainingDataDf = trainingDataDf[trainingDataDf.PatientId==25]

    # Run training
    run_training(training_patients_df=trainingDataDf,
                architecture_kws={'n_values_size':1, 'n_values_delta':0, 'architecture':[128, 64, 32, 16, 10], 'n_doseOptions':n_doseOptions},
                reward_kws={'gamma':gamma, 'base':base, 'hol':hol, 'punish':punish},
                learning_rate=learning_rate, updating_interval=day_interval, num_workers=num_workers, max_epochs=max_epochs, model_name=model_name,
                load_model=load_model, model_loaded_name=model_loaded_name, model_path=model_path, logging_interval=logging_interval, verbose=verbose)
