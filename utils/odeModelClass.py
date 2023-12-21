# ====================================================================================
# Abstract model class
# ====================================================================================
import numpy as np
import scipy.integrate
import pandas as pd
import os
import sys
import contextlib
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils

class ODEModel():
    def __init__(self, **kwargs):
        # Initialise parameters
        self.paramDic = {'DMax':100}
        self.stateVars = ['P1']
        self.resultsDf = None
        self.name = "ODEModel"

        # Set the parameters
        self.SetParams(**kwargs)

        # Configure the solver
        self.dt = kwargs.get('dt', 1e-3)  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', 1.0e-8)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', 1.0e-6)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', 'DOP853')  # ODE solver used
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          False)  # If true, suppress output of ODE solver (including warning messages)
        self.successB = False  # Indicate successful solution of the ODE system

    def __str__(self):
        return self.name

    # =========================================================================================
    # Function to set the parameters
    def SetParams(self, **kwargs):
        if len(self.paramDic.keys()) > 1:
            for key in self.paramDic.keys():
                self.paramDic[key] = float(kwargs.get(key, self.paramDic[key]))
            self.initialStateList = [self.paramDic[var + "0"] for var in self.stateVars]

    # =========================================================================================
    # Function to read in parameter set
    def GetParams(self, file_loc, method, id=0):
        df = pd.read_csv(file_loc)
        if method == "random":
            row = df.sample(1)
        elif method == "number":
            row = df.loc[df['PatientId'] == id]
        else:
            raise ValueError("Unknown parameter selection method")
        try:
            return row.to_dict('records')[0]
        except IndexError:
            raise ValueError("Patient %s not found" % id)

    # =========================================================================================
    # Function to simulate the model
    def Simulate(self, treatmentScheduleList, **kwargs):
        # Allow configuring the solver at this point as well
        self.dt = float(kwargs.get('dt', self.dt))  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', self.absErr)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', self.relErr)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', self.solverMethod)  # ODE solver used
        self.successB = False  # Indicate successful solution of the ODE system
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          self.suppressOutputB)  # If true, suppress output of ODE solver (including warning messages)

        # Solve
        self.treatmentScheduleList = treatmentScheduleList
        if self.resultsDf is None or treatmentScheduleList[0][0] == 0:
            currStateVec = self.initialStateList + [0]
            self.resultsDf = None
        else:
            currStateVec = [self.resultsDf[var].iloc[-1] for var in self.stateVars] + [self.resultsDf['DrugConcentration'].iloc[-1]]
        resultsDFList = []
        encounteredProblemB = False
        for intervalId, interval in enumerate(treatmentScheduleList):
            tVec = np.arange(interval[0], interval[1], self.dt)
            if intervalId == (len(treatmentScheduleList) - 1):
                tVec = np.arange(interval[0], interval[1] + self.dt, self.dt)
            currStateVec[-1] = interval[2]
            if self.suppressOutputB:
                with stdout_redirected():
                    solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                       t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                       method=self.solverMethod,
                                                       atol=self.absErr, rtol=self.relErr,
                                                       max_step=kwargs.get('max_step', 1))
            else:
                solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                   t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                   method=self.solverMethod,
                                                   atol=self.absErr, rtol=self.relErr,
                                                   max_step=kwargs.get('max_step', 1))
            # Check that the solver converged
            if not solObj.success or np.any(solObj.y < 0):
                self.errMessage = solObj.message
                encounteredProblemB = True
                if not self.suppressOutputB: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if not solObj.success:
                    if not self.suppressOutputB: print(self.errMessage)
                else:
                    if not self.suppressOutputB: print(
                        "Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver.")
                    if not self.suppressOutputB: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                self.solObj = solObj
                break
            # Save results
            resultsDFList.append(
                pd.DataFrame({"Time": tVec, "DrugConcentration": solObj.y[-1, :],
                              **dict(zip(self.stateVars,solObj.y))}))
            currStateVec = solObj.y[:, -1]
        # If the solver diverges in the first interval, it can't return any solution. Catch this here, and in this case
        # replace the solution with all zeros.
        if len(resultsDFList) > 0:
            resultsDf = pd.concat(resultsDFList)
        else:
            resultsDf = pd.DataFrame({"Time": tVec, "DrugConcentration": np.zeros_like(tVec),
                                     **dict(zip(self.stateVars,np.zeros_like(tVec)))})
        # Compute the fluorescent area that we'll see
        resultsDf['TumourSize'] = pd.Series(self.RunCellCountToTumourSizeModel(resultsDf),
                                            index=resultsDf.index)
        if self.resultsDf is not None:
            resultsDf = pd.concat([self.resultsDf, resultsDf])
        self.resultsDf = resultsDf
        self.successB = True if not encounteredProblemB else False

    # =========================================================================================
    # Define the model mapping cell counts to observed fluorescent area
    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        # Note: default scaleFactor value assumes a cell radius of 10uM. Volume is given in mm^3 -> r^3 = (10^-2 mm)^3 = 10^-6
        theta = self.paramDic.get('scaleFactor', 1)
        return theta * (np.sum(popModelSolDf[self.stateVars].values,axis=1))

    # =========================================================================================
    # Simulate adaptive therapy (Zhang et al algorithm)
    def Simulate_AT(self, refSize=None, atThreshold=0.5, atMethod = "Zhang", sigmoidWidth = 50, intervalLength=1.,
                    t_end=1000, nCycles=np.inf, tumourSizeWhenProgressed=1.2, t_span=None, reward_func=None, solver_kws={}):
        
        t_span = t_span if t_span is not None else (0, t_end)
        currInterval = [t_span[0], t_span[0] + intervalLength]
        refSize = self.paramDic.get('scaleFactor', 1) * np.sum(
            self.initialStateList) if refSize is None else refSize
        maxTolerableBurden = np.sum(self.initialStateList) * tumourSizeWhenProgressed
        dose = self.paramDic['DMax']
        currCycleId = 0
        if reward_func is not None:
            self.reward = 0
            

        progressed = False
        while (currInterval[1] <= t_end) and (currCycleId < nCycles) and (not progressed):
            # Simulate
            # print(currInterval,refSize)
            self.Simulate([[currInterval[0], currInterval[1], dose]], **solver_kws)

            # Update dose
            # print(self.resultsDf.TumourSize.iat[-1],(1-atThreshold)*refSize)
            if atMethod == "Zhang":
                if self.resultsDf.TumourSize.iat[-1] < (
                        1 - atThreshold) * refSize:  # Withdraw treatment below a certain size
                    dose = 0
                elif self.resultsDf.TumourSize.iat[-1] > refSize:
                    dose = self.paramDic['DMax']
                else:  # If size remains within a window of +- atThreshold, keep the same dose
                    dose = dose

            elif atMethod == "Sigmoid":
                sigmoid = lambda n : 1 / (1 + np.exp(- sigmoidWidth * (n - atThreshold)))
                norm_size = self.resultsDf.TumourSize.iat[-1] / refSize
                dose = self.paramDic['DMax'] * np.random.choice(2, 1, p = [1 - sigmoid(norm_size), sigmoid(norm_size)])[0]
            else:
                raise ValueError("Unknown method '%s', supported methods are 'Zhang' or 'Sigmoid'" % atMethod)
            # print(dose, intervalLength)

            # Update interval
            currInterval = [x + intervalLength for x in currInterval]
            currCycleId += 1
            progressed = np.any(self.resultsDf.TumourSize>maxTolerableBurden)

            if reward_func is not None:
                self.reward += reward_func(self.resultsDf, refSize, currInterval[1], dose)
                

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)
        self.ttp = self.resultsDf.Time[self.resultsDf.TumourSize>maxTolerableBurden].min()


        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)
        self.ttp = self.resultsDf.Time[self.resultsDf.TumourSize>maxTolerableBurden].min()

    # =========================================================================================
    # Interpolate to specific time resolution (e.g. for plotting)
    def Trim(self, t_eval=None, dt=1):
        t_eval = np.arange(0, self.resultsDf.Time.max(), dt) if t_eval is None else t_eval
        tmpDfList = []
        trimmedResultsDic = {'Time': t_eval}
        for variable in [*self.stateVars, 'TumourSize', 'DrugConcentration']:
            f = scipy.interpolate.interp1d(self.resultsDf.Time, self.resultsDf[variable])
            trimmedResultsDic = {**trimmedResultsDic, variable: f(t_eval)}
        tmpDfList.append(pd.DataFrame(trimmedResultsDic))
        self.resultsDf = pd.concat(tmpDfList)

    # =========================================================================================
    # Interpolate to specific time resolution (e.g. for plotting)
    def NormaliseToInitialSize(self, dataDf):
            dataDf['S'] /= dataDf.TumourSize.iloc[0]
            dataDf['R'] /= dataDf.TumourSize.iloc[0]
            dataDf['TumourSize'] /= dataDf.TumourSize.iloc[0]
            
    # =========================================================================================
    # Function to plot the model predictions
    def Plot(self, plotPops=False, n_boot=1,
             xmin=0, xlim=None, ymin=0, ylim=None, y2lim=1, palette=None,
             decorateAxes=True, legend=False, drug_label="Drug Concentration",
             decoratey2=True, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots(1, 1)
        varsToPlotList = ["TumourSize"]
        if plotPops: varsToPlotList += self.stateVars
        currModelPredictionDf = pd.melt(self.resultsDf, id_vars=['Time'], value_vars=varsToPlotList)
        sns.lineplot(x="Time", y="value", hue="variable", style="variable", n_boot=n_boot,
                     lw=5, palette=palette,
                     #legend=legend,
                     data=currModelPredictionDf, ax=ax)

        # Plot the drug concentration
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        drugConcentrationVec = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(self.resultsDf),
                                                       tVec=self.resultsDf['Time'])
        ax2.fill_between(self.resultsDf['Time'],
                         0, drugConcentrationVec, color="#8f59e0", alpha=0.2, label=drug_label)
        # Format the plot
        if xlim is not None: ax.set_xlim([xmin, xlim])
        if ylim is not None: ax.set_ylim([ymin, ylim])
        ax2.set_ylim([0, y2lim])
        ax.set_xlabel("Time in Days" if decorateAxes else "")
        ax.set_ylabel("Confluence" if decorateAxes else "")
        ax2.set_ylabel(r"Drug Concentration in $\mu M$" if decorateAxes else "")
        ax.set_title(kwargs.get('title', ''))
        if not decoratey2:
            ax2.set_yticklabels("")
            ax2.tick_params(color='w')
        if legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            h1.pop(l1.index('variable'))
            l1.remove("variable")
            leg = ax.legend(h1+h2, l1+l2, loc=2)#, framealpha=0.8, facecolor='w')
            for legobj in leg.legendHandles[:-1]:
                legobj.set_linewidth(4.5)

        plt.tight_layout()
        if kwargs.get('saveFigB', False):
            plt.savefig(kwargs.get('outName', 'modelPrediction.png'), orientation='portrait', format='png')
            plt.close()


# ====================================================================================
# Functions used to suppress output from odeint
# Taken from: https://stackoverflow.com/questions/31681946/disable-warnings-originating-from-scipy
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied