# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import numpy as np
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from tqdm import tqdm
import string
import json
import datetime
import os

# ====================================================================================
# Functions for dealing with treatment schedules
# ====================================================================================
# Helper function to extract the treatment schedule from the data
def ConvertTDToTSFormat(timeVec,drugIntensityVec):
    treatmentScheduleList = [] # Time intervals in which we have the same amount of drug
    tStart = timeVec[0]
    currDrugIntensity = drugIntensityVec[0]
    for i,t in enumerate(timeVec):
        if drugIntensityVec[i]!=currDrugIntensity and not (np.all(np.isnan([drugIntensityVec[i],currDrugIntensity]))): # Check if amount of drug has changed
            treatmentScheduleList.append([tStart,t,currDrugIntensity])
            tStart = t
            currDrugIntensity = drugIntensityVec[i]
    treatmentScheduleList.append([tStart,timeVec[-1]+(tStart==timeVec[-1])*1,currDrugIntensity])
    return treatmentScheduleList

# Helper function to obtain treatment schedule from calibration data
def ExtractTreatmentFromDf(dataDf,treatmentColumn="DrugConcentration"):
    timeVec = dataDf['Time'].values
    nDaysPreTreatment = int(timeVec.min())
    if nDaysPreTreatment != 0: # Add the pretreatment phase if it's not already added
        timeVec = np.concatenate((np.arange(0, nDaysPreTreatment), timeVec), axis=0)
    drugIntensityVec = dataDf[treatmentColumn].values
    drugIntensityVec = np.concatenate((np.zeros((nDaysPreTreatment,)), drugIntensityVec), axis=0)
    return ConvertTDToTSFormat(timeVec, drugIntensityVec)

# Turns a treatment schedule in list format (i.e. [tStart, tEnd, DrugConcentration]) into a time series
def TreatmentListToTS(treatmentList,tVec):
    drugConcentrationVec = np.zeros_like(tVec)
    for drugInterval in treatmentList:
        drugConcentrationVec[(tVec>=drugInterval[0]) & (tVec<=drugInterval[1])] = drugInterval[2]
    return drugConcentrationVec

# Extract the date as a datetime object from a model or experiment data frame
def GetDateFromDataFrame(df):
    year, month, day, hour, minute = [df[key].values[0] for key in ['Year','Month','Day','Hour','Minute']]
    hour = 12 if np.isnan(hour) else hour
    minute = 0 if np.isnan(minute) else minute
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))

# ====================================================================================
# Misc
# ====================================================================================
def convert_ode_parameters(n0=0.75, rFrac=1e-3, cost=0, turnover=0, rS = 0.027, **kwargs):
    '''
    Converts parameters from a cost/turnover description (more interpretable, used in fitting model to 
    patient data) to the parameters used in the ODE model. Returns a dictionary that can be fed into the
    LotkaVolterraModel() class.
    
    :param n0: initial tumor density
    :param rFrac: initial resistance fraction (R0/(S0+R0))
    :param cost: resistance cost (rR = (1-cost)*rS)
    :param turnover: tumor cell death (dT = turnover*rS)
    :param rS: sensitive cell proliferation rate (used as time scale)
    '''
    return {'n0': n0, 'rS':rS, 'rR':(1-cost)*rS, 'dS':turnover*rS, 'dR':turnover*rS,
            'dD':1.5, 'k':1., 'D':0, 'theta':1, 'DMax':1.,
            'S0':n0*(1-rFrac), 'R0':n0*rFrac, 'cost': cost, 'turnover': turnover}

def obtain_architecture(model_name): 
    '''
    Returns dictionary of achitecture parameters from parameters file, to feed back into eval function.
    '''
    path = os.path.join("../models/", model_name, "paramDic_%s.txt"%(model_name))
    with open(path, 'r') as file:
        lines = [line.rstrip() for line in file]
        in_dict = {l.split(':')[0]:l.split(':')[1] for l in lines}  
        my_dict = {key: int(in_dict[key]) for key in ['n_inputs', 'n_values_size', 'n_values_delta', 'n_doseOptions']}
        my_dict['architecture'] = json.loads(in_dict['architecture'])
        return my_dict

def printTable(myDict, colList=None, printHeaderB=True, colSize=None, **kwargs):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList: colList = list(myDict[0].keys() if myDict else [])
    myList = [colList] if printHeaderB else [] # 1st row = header
    for item in myDict: myList.append([str('%.2e'%item[col] or '') for col in colList])
    colSize = [max(map(len,col)) for col in zip(*myList)] if not colSize else colSize
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    if printHeaderB: myList.insert(1, ['-' * i for i in colSize]) # Seperating line
    for item in myList: print(formatStr.format(*item))
    if kwargs.get('getColSizeB',False): return colSize

def mkdir(dirName):
    """
    Recursively generate a directory or list of directories. If directory already exists be silent. This is to replace
    the annyoing and cumbersome os.path.mkdir() which can't generate paths recursively and throws errors if paths
    already exist.
    :param dirName: if string: name of dir to be created; if list: list of names of dirs to be created
    :return: Boolean
    """
    dirToCreateList = [dirName] if type(dirName) is str else dirName
    for directory in dirToCreateList:
        currDir = ""
        for subdirectory in directory.split("/"):
            currDir = os.path.join(currDir, subdirectory)
            try:
                os.mkdir(currDir)
            except:
                pass
        return True
    