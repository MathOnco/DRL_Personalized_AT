# ====================================================================================
# Class to simulate simple 2-population Lotka-Volterra tumour model
# ====================================================================================
import sys
import numpy as np
from math import exp, log
sys.path.append("./")
from odeModelClass import ODEModel

class LotkaVolterraModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "LotkaVolterraModel"
        self.paramDic = {**self.paramDic,
                        'rS': 0.027,
                        'rR': 0.027,
                        'K': 1,
                        'dD': 1.5,
                        'dS': 0.,
                        'dR': 0.,
                        'S0': 0.74,
                        'R0':0.01,
                        'DMax':1}
        self.stateVars = ['S', 'R']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['rS'] * (1 - (S+R)/self.paramDic['K']) * (1-self.paramDic['dD']*D) * S - self.paramDic['dS']*S
        dudtVec[1] = self.paramDic['rR'] * (1 - (S+R)/self.paramDic['K']) * R - self.paramDic['dR']*R
        dudtVec[2] = 0
        return (dudtVec)

class ExponentialModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ExponentialModel"
        self.paramDic = {**self.paramDic,
                        'rS': 0.01365,
                        'rR': 0.00825,
                        'Ks': 1,
                        'Kr': 0.25,
                        'dDs': 2.3205,
                        'dDr': 1.3205,
                        'S0': 0.74,
                        'R0':0.01,
                        'DMax':1,
                        'alpha':1,
                        'gamma':0.27385}
        self.stateVars = ['S', 'R']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        try:
            dudtVec[0] = self.paramDic['rS'] * S * (1 - ((S+(R / (1+exp(self.paramDic['gamma']*t))))/self.paramDic['Ks'])**self.paramDic['alpha'] - self.paramDic['dDs']*D)
            dudtVec[1] = self.paramDic['rR'] * R * (1 - ((R+(S / (1+exp(self.paramDic['gamma']*t))))/self.paramDic['Kr'])**self.paramDic['alpha'] - self.paramDic['dDr']*D)
        except OverflowError:
            dudtVec[0] = self.paramDic['rS'] * S * (1 - ((S+(0))/self.paramDic['Ks'])**self.paramDic['alpha'] - self.paramDic['dDs']*D)
            dudtVec[1] = self.paramDic['rR'] * R * (1 - ((R+(0))/self.paramDic['Kr'])**self.paramDic['alpha'] - self.paramDic['dDr']*D)
        dudtVec[2] = 0
        return (dudtVec)  

class StemCellModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "StemCellModel"
        self.paramDic = {**self.paramDic,
                        'rR': log(2),
                        'beta': 1e-6,
                        'dR': 0.07,
                        'rho': 0.0001,
                        'phi': 0.01,
                        'S0': 1000,
                        'R0': 10,
                        'P0': 29,
                        'DMax': 1}
        self.stateVars = ['S', 'R', 'P']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, P, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = (1 - (R / (S + R)) * self.paramDic['beta']) * self.paramDic['rR'] * R - self.paramDic['dR'] * D * S  # Differentiated cells
        dudtVec[1] = (R / (S + R)) * self.paramDic['beta'] * self.paramDic['rR'] * R  # Stem-like (drug resistant) cells
        dudtVec[2] = self.paramDic['rho'] * S - self.paramDic['phi'] * P
        dudtVec[3] = 0
        return (dudtVec)
    