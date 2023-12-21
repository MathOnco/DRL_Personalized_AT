# Generate synthetic patient data from a range of data points
# Optional checks to ensure progression

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../utils")
from LotkaVolterraModel import LotkaVolterraModel


def random_params(paramRange):
    """Selects parameters randomly from output range"""
    output = {}
    for key in paramRange:
        if isinstance(paramRange[key], list):
            assert len(paramRange[key])==2, "Lists have upper and lower bound only"
            output[key] = np.random.uniform(*paramRange[key])
        else:
            output[key] = paramRange[key]
    return output

def complete_param_dict(cost, turnover, n0, r0, rs=0.027):
    """Pass unpacked dict to generate param dict in
    form required by model class.
    N.B rS is a defualt value from literatue"""
    return {"Cost": cost, "Turnover": turnover,
                "n0": n0, "R0": r0, "S0": (n0 - r0),
                "rS": rs, "rR": rs * (1-cost),
                "dS": rs * turnover, "dR": rs * turnover,
                'K': 1, 'DD': 1.5}

def test_progression(paramDic, t_end, day_interval=1):
    """Selects for patients that reach progression (20% growth
    from initial size) within t_end days, under rule of thumb 
    AT treatment.
    """
    model = LotkaVolterraModel(method='RK45', dt=1)
    model.SetParams(**paramDic)
    model.Simulate_AT(atThreshold=0.5, intervalLength=day_interval, t_end=t_end)
    output = max(model.resultsDf['Time']) + day_interval < t_end
    return int(output)


# Set allowed paramter ranges (use scalars for set values):
paramRange = {"n0": 0.1, "r0": 0.001,
              "cost": [0.2, 0.5], 
              "turnover": [0.5, 0.8]}


# Generate random patients
n_patients = 10
check_progression = True
t_end = 5000  # Ensure progression occurs in this period

for i in range(n_patients):
    while True:
        paramRand = random_params(paramRange)
        paramDic = complete_param_dict(**paramRand)
        
        progressed = test_progression(paramDic, t_end=5000)
        if progressed or not check_progression:
            print("Check passed!")
            break
        print("Check failed")

    paramDic["PatientId"] = i
    if i == 0:
        df = pd.DataFrame(paramDic, index=[0])
    else:
        df = pd.concat([df, pd.DataFrame(paramDic, index=[i])])

df.to_csv(f"patients/randSynthetic{n_patients}Patients.csv")
