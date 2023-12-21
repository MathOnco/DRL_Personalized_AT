#
# Truncates all patient data to the first n cycles, to use for fitting.
#

import os
import sys
import glob
import pandas as pd

sys.path.append('../utils/')
from myUtils import mkdir

from bruchovsky_patients import LoadPatientData

data_dir = "patients/Bruchovsky_et_al/Complete_data"
output_dir = "patients/Bruchovsky_et_al/Truncated_data"

n_cycles = 2

def find_nth(string, substring, n):
    if (n == 1):
       return string.find(substring)
    else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

def truncate_df(dataframe, n, drug_name = 'Abi'):
    """Select the first n cycles from patient data.
    Done by selecting the nth occurance of drug turning
    on in the 'Abi' column."""
    drug_seq = ''.join([str(x) for x in dataframe[drug_name].values])
    end_ind = find_nth(drug_seq, '01', n)
    return dataframe.head(end_ind + 1)

def truncate_data(data_dir, output_dir, n_cycles):
    """Truncate all data files in a directory to the first n cycles"""
    mkdir(output_dir)
    for file in glob.glob(data_dir + "/*.csv"):
        df = pd.read_csv(file, index_col = 'Days')
        df_out = truncate_df(df, n_cycles)
        df_out.to_csv(os.path.join(output_dir, os.path.basename(file)), index='Days')

def truncate_txt_data(data_dir, output_dir, n_cycles):
    """Truncate all data files in a directory to the first n cycles"""
    mkdir(output_dir)
    for file in glob.glob(data_dir + "/*.txt"):
        patient_id = int(file.split('.')[-2][-3:])  # last three digits of name before file
        df = LoadPatientData(patient_id, data_dir)
        df_out = df.rename(columns={'Time': 'Days', 'DrugConcentration': 'Abi',
                                        'PSA': 'relPSA_Indi', 'PSA_raw':'PSA'})  # For compatibility with fitting code
        df_out.dropna(subset=['Days', 'Abi', 'PSA'], inplace=True)
        file_name = os.path.basename(file).split('.')[0] + ".csv"
        df_out.to_csv(os.path.join(output_dir, file_name), index='Days')

truncate_txt_data(data_dir, output_dir, n_cycles)
