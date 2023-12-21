# Parameters

This contains a list of parameters used in the model, alongisde their default values and meaning.

## DRL parameters

- `gamma` = 0.9999 - _(Discount rate for advantage estimation and reward discounting)_
- `n_doseOptions` = 2 - _(Agent can choose 2 different intensities of chemotherapy)_
- `base` = 0.1 - _(Base reward given per timestep survived)_
- `hol` = 0.05 - _(Additional reward for a treatment holiday)_
- `day_interval` = 1 - _(Interval between decision points (time step for DRL evaluation of ODE model))_
- `max_episode_length` = 3000 - _(Maximum length to run for during training (note: this currently is not functional))_
- `num_workers`\* = 4 - _(`Multiprocessing.cpu_count()`, sets workers to the number of available CPU threads)_
- `model_path` = "./models"
- `results_path` = "./results" - _(Directory to save output .csv in)_
- `logging_interval` = 100 - _(Save the state of the network every `logging_interval` patients)_
- `model_loaded_name` ='6260_patients_MonJul191846322021'
- `verbose` = 0 - _(Boolean to determine level of detail printed to terminal during runtime)_

## Tumour model parameters

- `n_replicates` = 5 - _(Number of times to repeat ODE model)_
- `n0` = 0.75 - _(Initial size of tumour)_
- `rFrac` = 0.001 - _(Initial proportion of susceptible cells)_
- `paramDic` = {} - _(See section below)_

### Parameter Dictionary

- `rS` = 0.027 - _(Birth rate of susceptible cells)_
- `rR` = 0.027 - _(Birth rate of resistant cells)_
- `dS` = 0.0 - _(Death rate of susceptible cells)_
- `dR` = 0.0 - _(Death rate of resistant cells)_
- `dD` = 1.5 - _(Drug induced death rate of susceptible cells)_
- `k` = 1.0 - _(Carrying capacity of tumour environment)_
- `D` = 0 - _(Drug concentration in environment)_
- `theta` = 1 - _(Scale factor for mapping cell counts to observed fluorescent area - assumes a cell radius of 10uM)_
- `DMax` = 1.0 - _(Max doseage given to patient)_
- `S0` = n0 * (1 - rFrac) - _(Initial size of susceptible portion of tumour)_
- `R0` = (n0 * rFrac) - _(Initial number of resistant portion of tumour)_
- `punish` = -0.1 - _(Punishment for exceeding 20% limit on tumour growth)_
- `learning_rate` = 1e-4 - _(Learning rate for Adam Optimiser)_

Also includes `n0`, `rFrac`, `base`, `hol`, and `day_interval` as `interval`, which are described above.

\* These parameters are used in training only.
