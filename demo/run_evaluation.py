from cProfile import run
from operator import index
import pandas as pd
import numpy as np
import os
import sys
import json

sys.path.append("../utils")
from drlUtils import run_evaluation

if __name__ == "__main__":
    if len(sys.argv)>1:
        with open(sys.argv[1], 'r') as j:
            contents = json.loads(j.read())
        run_evaluation(**contents)

    else:
        model_name = "test_currSizeOnly_step2"
        model_path = os.path.join(os.pardir, "models", model_name, "%d_patients_%s"%(10000, model_name))
        results_path = './results/development'
        n_replicates = 2
        updating_interval = 7
        verbose = 2
        
        # Tumour model parameters
        evaluation_patients_df = pd.read_csv("../models/kits_model/randSynthetic10Patients.csv", index_col=0)
        
        # Run training
        # def run_evaluation(model_path, patients_to_evaluate, n_replicates=100, updating_interval=7, results_path="./", results_file_name="results.csv",
        #                ODE_model='LK', tqdm_output=sys.stderr, verbose=0, seed=42):
        run_evaluation(model_path=model_path, patients_to_evaluate=evaluation_patients_df, 
                    n_replicates=n_replicates, updating_interval=updating_interval, results_path=results_path, 
                    verbose=verbose)
        # argDic = {'model_path': '..\\models\\test_currSizeOnly_CRBaseCase\\500_patients_test_currSizeOnly_CRBaseCase', 'patients_to_evaluate': {'n0': {0: 0.75}, 'rS': {0: 0.027}, 'rR': {0: 0.027}, 'dS': {0: 0.0}, 'dR': {0: 0.0}, 'dD': {0: 1.5}, 'k': {0: 1.0}, 'D': {0: 0}, 'theta': {0: 1}, 'DMax': {0: 1.0}, 'S0': {0: 0.74925}, 'R0': {0: 0.00075}, 'PatientId': {0: -100}}, 'n_replicates': 10, 'updating_interval': 7, 'results_path': './results/development', 'results_file_name': 'resultsDf_test_currSizeOnly_CRBaseCase_crBase_500_patients.csv', 'verbose': 0, 'tqdm_output': './results/development\\log_currentSizeOnly_baseCase.txt'}
        # run_evaluation(**argDic)