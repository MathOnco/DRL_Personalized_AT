# Code is property of Maximillian Strobl (2020), can be accessed at
# https://github.com/MathOnco/AT_costOfResistance_LVModel/blob/master/utils/fittingUtils.py

# ------------------------ Imports ------------------------
import pandas as pd
import numpy as np
import scipy
import sys
import os
from tqdm import tqdm
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import pickle
import re

sys.path.append('../../utils/')
import LotkaVolterraModel as lvm
import myUtils as utils

# Format plots
sns.set(style="white",
        font_scale=1.5,
        rc={'figure.figsize':(12,6)})

# ------------------------ Functions for fitting data ------------------------
def residual(params, x, data, eps_data, model, feature="TumourSize",solver_kws={}):
    model.SetParams(**params.valuesdict())
    converged = False
    max_step = solver_kws.get('max_step', 1)
    currSolver_kws = solver_kws.copy()
    while not converged:
        model.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(data), **currSolver_kws)
        converged = model.successB
        max_step = 0.75*max_step if max_step < np.inf else 100
        currSolver_kws['max_step'] = max_step
    # Interpolate to the data time grid
    t_eval = data.Time
    f = scipy.interpolate.interp1d(model.resultsDf.Time,model.resultsDf.TumourSize,fill_value="extrapolate")
    modelPrediction = f(t_eval)
    return (data[feature]-modelPrediction) / eps_data

def PerturbParams(params):
    params = params.copy()
    for p in params.keys():
        currParam = params[p]
        if currParam.vary:
            params[p].value = np.random.uniform(low=currParam.min, high=currParam.max)
    return params

def ComputeRSquared(fit,dataDf,feature="PSA"):
    tss = np.sum(np.square(dataDf[feature]-dataDf[feature].mean()))
    rss = np.sum(np.square(fit.residual))
    return 1-rss/tss

# ------------------------ Functions for analysing fits ------------------------
def PatientToOutcomeMap(patientId):
    patientsToExcludeList = [32, 46, 64, 83, 92]  # Exclude all patients with metastasis
    patientsWithRelapse = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]
    if patientId in patientsToExcludeList:
        return -1
    if patientId in patientsWithRelapse:
        return 1
    else:
        return 0

def LoadPatientData(patientId, dataDir="patients/eLife_Trial/at_patients"):
    patientDataDf = pd.read_csv(os.path.join(dataDir, "patient%.3d.csv" % patientId))
    patientDataDf.rename(columns={"Days": "Time", "PSA": "PSA_raw", "relPSA_Indi": "PSA",
                                  "Abi": "DrugConcentration"}, inplace=True)
    patientDataDf['PatientId'] = int(patientId)

    # patientDataDf['PSA'] /= patientDataDf.PSA.iloc[0]
    return patientDataDf

def PlotData(dataDf, feature='PSA', titleStr="", drugBarPosition=0.85,
             xlim=2e3, ylim=1.3, y2lim=1, decorateX=True, decorateY=True, decoratey2=False, markersize=10,
             ax=None, figsize=(10, 8), outName=None, **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the data
    ax.plot(dataDf.Time, dataDf[feature],
            linestyle="None", marker="x", markersize=markersize,
            color="black", markeredgewidth=2)

    # Plot the drug concentration
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    drugConcentrationVec = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(dataDf),
                                                   tVec=dataDf['Time'])
    drugConcentrationVec = drugConcentrationVec / (1 - drugBarPosition) + drugBarPosition
    ax2.fill_between(dataDf['Time'], drugBarPosition, drugConcentrationVec,
                     step="post", color="black", alpha=1., label="Drug Concentration")
    ax2.axis("off")

    # Format the plot
    ax.set_xlim(0,xlim)
    ax.set_ylim(0, ylim)
    ax2.set_ylim([0, y2lim])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(titleStr)
    ax.tick_params(labelsize=28)
    ax2.tick_params(labelsize=28)
    ax.legend().remove()
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)


def LoadFit(patientId, fitId, fitDir="./fits"):
    fitObj = pickle.load(
        open(os.path.join(fitDir, "patient%d" % patientId, "fitObj_patient_%d_fit_%d.p" % (patientId, fitId)), "rb"))
    fitObj.patientId = patientId
    return fitObj


def GetBestFit(patientId, criterion="AIC", fitDir="./fits"):
    summaryDf = GenerateFitSummaryDf(patientId, fitDir)
    if summaryDf.shape[0] > 0:
        bestFitId = summaryDf.FitId[summaryDf.AIC == summaryDf.AIC.min()].values[0]
        return LoadFit(patientId, bestFitId, fitDir)
    else:
        return -1


def GenerateFitSummaryDf(patientId, fitDir="./fits"):
    fitIdList = [int(re.findall(r'\d+', x)[1]) for x in os.listdir(os.path.join(fitDir, "patient%d" % patientId)) if
                 x.split("_")[0] == "fitObj"]
    tmpDicList = []
    for fitId in fitIdList:
        currFit = LoadFit(patientId, fitId, fitDir)
        tmpDicList.append({"PatientId": currFit.patientId, "FitId": currFit.fitId,
                           "AIC": currFit.aic, "BIC": currFit.bic, "RSquared": currFit.rSq,
                           **currFit.params.valuesdict()})
    return pd.DataFrame(tmpDicList)

def CountCycles(series):
    count = 0
    treating = False
    for s in series:
        if not treating and s > 0:
            # Start of treatment
            count += 1
            treating = True
        elif treating and s == 0:
            # End of treatment
            treating = False
    return count


def GenerateFitSummaryDf_AllPatients(patientIdList,fitDir="./fits",dataDir="patients/eLife_Trial/at_patients",progressBar=False):
    tmpDfList = []
    for patientId in tqdm(patientIdList,disable=progressBar==False):
        outcome = PatientToOutcomeMap(patientId)
        if outcome==-1: continue
        currDataDf = LoadPatientData(patientId,dataDir=dataDir)
        tmp = GenerateFitSummaryDf(patientId,fitDir=fitDir)
        if tmp.shape[0]==0: continue
        bestFit = GetBestFit(patientId=patientId,fitDir=fitDir)
        currRow = tmp[tmp.FitId==bestFit.fitId].copy()
        currRow['NSuccessfulFits'] = tmp.shape[0]
        currRow['NCycles'] = CountCycles(currDataDf.DrugConcentration)
        currRow['TimeInTrial'] = currDataDf.Time.max()
        currRow['Progression'] = outcome
        tmpDfList.append(currRow)
    return pd.concat(tmpDfList)


def SimulateFit(fitObj, dataDf, dt=1, solver_kws={}):
    myModel = lvm.LotkaVolterraModel()
    myModel.SetParams(**fitObj.params.valuesdict())
    myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),**solver_kws)
    # Interpolate to the desired time grid
    t_eval = np.arange(0, myModel.resultsDf.Time.max(), dt)
    trimmedResultsDic = {'Time': t_eval}
    for variable in ['S', 'R', 'TumourSize', 'DrugConcentration']:
        f = scipy.interpolate.interp1d(myModel.resultsDf.Time, myModel.resultsDf[variable])
        trimmedResultsDic = {**trimmedResultsDic, variable: f(t_eval)}
    myModel.resultsDf = pd.DataFrame(trimmedResultsDic)
    return myModel


def SimulateFits(fits, dataDf, fitDir="./fits", progressBar=False, solver_kws={}):
    if type(fits[0]) == int:
        patientId = dataDf.PatientId.unique()[0]
        fitObjList = [LoadFit(patientId, fitId, fitDir) for fitId in fits]
    else:
        fitObjList = fits
    tmpDfList = []
    for fitObj in tqdm(fitObjList, disable=progressBar == False):
        myModel = SimulateFit(fitObj, dataDf, solver_kws=solver_kws)
        myModel.resultsDf['FitId'] = fitObj.fitId
        tmpDfList.append(myModel.resultsDf)
    return pd.concat(tmpDfList, sort=False)


def PlotFit(fitObj, dataDf, titleStr="", ax=None, sim_kws={}):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    myModel = SimulateFit(fitObj, dataDf, **sim_kws)
    myModel.Plot(plotPops=True, ymin=0, title=titleStr, legendB=False, ax=ax)
    ax.plot(dataDf.Time, dataDf.PSA, linestyle='none', marker='x')


def PlotFits(fits, dataDf, fitDir="./fits", solver_kws={}, aggregateData=True, progressBar=False, drugBarPosition=0.85,
             xlim=None, ylim=1.3, y2lim=1, decorateX=True, decorateY=True, axisLabels=False, markersize=10, labelsize=28,
             titleStr="", ax=None, figsize=(10, 8), outName=None, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Simulate
    if type(fits) == list:
        predictionDf = SimulateFits(fits, dataDf, fitDir=fitDir, progressBar=progressBar, solver_kws=solver_kws)
    else:
        predictionDf = fits

    # Plot the size the we will see on the images
    sns.lineplot(x="Time", y="TumourSize", ci='sd',
                 lw=kwargs.get('linewidthA', kwargs.get('linewidth',7)),
                 color=kwargs.get('colorA', '#094486'),
                 estimator='mean' if aggregateData else None,
                 data=predictionDf, ax=ax)

    # Plot the individual populations
    sns.lineplot(x="Time", y="S", ci='sd',
                 lw=kwargs.get('linewidth', 7), color=kwargs.get('colorS', "#0F4C13"),
                 estimator='mean' if aggregateData else None,
                 data=predictionDf, ax=ax)
    ax.lines[1].set_linestyle(kwargs.get('linestyleS', '--'))
    sns.lineplot(x="Time", y="R", ci='sd',
                 lw=kwargs.get('linewidth', 7), color=kwargs.get('colorR', '#710303'),
                 estimator='mean' if aggregateData else None,
                 data=predictionDf, ax=ax)
    ax.lines[2].set_linestyle(kwargs.get('linestyleR', '-.'))

    # Plot the data
    ax.plot(dataDf.Time, dataDf.PSA,
            linestyle="None", marker="x", markersize=markersize,
            color="black", markeredgewidth=4)

    # Plot the drug concentration
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    exampleFitId = predictionDf.FitId.unique()[0]
    timeVec = predictionDf.Time[predictionDf.FitId == exampleFitId]
    drugConcentrationVec = predictionDf.DrugConcentration[predictionDf.FitId == exampleFitId]
    drugConcentrationVec = drugConcentrationVec / (1 - drugBarPosition) + drugBarPosition
    ax2.fill_between(timeVec,
                     drugBarPosition, drugConcentrationVec, color="black",
                     alpha=1., label="Drug Concentration")
    ax2.axis("off")

    # Format the plot
    if xlim is not None: ax.set_xlim(0,xlim)
    ax.set_ylim(0, ylim)
    ax2.set_ylim([0, y2lim])
    ax.set_xlabel("Time in Days" if axisLabels else "", fontdict={'fontsize':28})
    ax.set_ylabel("PSA (Normalised)" if axisLabels else "", fontdict={'fontsize':28})
    ax.set_title(titleStr)
    ax.tick_params(labelsize=labelsize)
    ax2.tick_params(labelsize=labelsize)
    # ax.legend().remove()
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)


def PlotParameterDistribution_SinglePatient(patientId, fitDir="./fits", showAll=False, nCols=5, figsize=(12, 4), ax=None):
    summaryDf = GenerateFitSummaryDf(patientId, fitDir)
    exampleFit = LoadFit(patientId, summaryDf.FitId.iloc[0], fitDir=fitDir)
    paramNamesList = list(exampleFit.params.keys()) if showAll else exampleFit.var_names
    nParams = len(paramNamesList)

    if ax is None: fig, axList = plt.subplots(nParams // nCols, nCols, figsize=figsize)
    for i, param in enumerate(paramNamesList):
        currAx = axList.flatten()[i]
        sns.stripplot(x="PatientId", y=param, data=summaryDf, ax=currAx)
        currAx.hlines(xmin=-1, xmax=2, y=exampleFit.params[param].min, linestyles='--')
        currAx.hlines(xmin=-1, xmax=2, y=exampleFit.params[param].max, linestyles='--')
        currAx.set_xlabel("")
        sns.despine(ax=currAx, offset=5, trim=True)
        currAx.tick_params(labelsize=24, rotation=45)
        currAx.set_xlabel("")
        currAx.set_ylabel("")
        currAx.set_title(param)
    plt.tight_layout()

def PlotParameterDistribution_PatientCohort(dataDf, paramList=["cost","turnover","n0","fR"], fitDir="./fits", showAll=False, plotBounds=True,
                                            palette={0:"teal",1:"orange"}, printTitle=True, figsize=(12, 6), outName=None, ax=None):
    if ax is None: fig, axList = plt.subplots(1, len(paramList), sharex=True, sharey=False, figsize=figsize)
    dataDf = dataDf.copy()
    exampleFit = GetBestFit(dataDf.PatientId.iloc[0], fitDir=fitDir)
    paramList = list(exampleFit.params.keys()) if showAll else paramList
    for i, param in enumerate(paramList):
        currAx = axList.flatten()[i]
        dataDf[param] *= 100
        sns.boxplot(x="Progression",y=param, width=0.5, palette=palette,
                    linewidth=3, data=dataDf, ax=currAx)
        sns.swarmplot(x="Progression", y=param, color='black', s=7, data=dataDf, ax=currAx)
        if plotBounds:
            currAx.hlines(xmin=-1, xmax=2, y=exampleFit.params[param].min*100, linestyles='--')
            currAx.hlines(xmin=-1, xmax=2, y=exampleFit.params[param].max*100, linestyles='--')
        currAx.set_xlabel("")
        sns.despine(ax=currAx, offset=5, trim=True)
        currAx.tick_params(labelsize=24, rotation=45)
        currAx.set_xlabel("")
        currAx.set_ylabel("")
        if printTitle: currAx.set_title(param)
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)


def QQPlot(dataDf, fit, feature="PSA", titleStr="", decorate=False, figsize=(7, 7), outName=None, ax=None):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    observedVals = dataDf[feature].dropna()
    predictedVals = observedVals - fit.eps_data * fit.residual
    sns.scatterplot(x=observedVals, y=predictedVals, ax=ax)
    # Add the 45 degree line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [0.75 * max(x0, y0) - 1e-5, 1.25 * min(x1, y1)]
    ax.plot(lims, lims, ':k')
    # Decorate
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(titleStr)
    ax.set_xlabel("Observed" if decorate else "")
    ax.set_ylabel("Predicted" if decorate else "")
    ax.tick_params(labelsize=28)
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)

def visualize_scatter_with_images(x, y, dataDf, hue=None, palette=None, imgDir="./", image_zoom=.4,
                                  outName=None, figsize=(65,65), showAxes=False, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for _,row in dataDf.iterrows():
        x0, y0 = row[[x,y]].values
        img = OffsetImage(plt.imread(os.path.join(imgDir,"patient%d.png"%row['PatientId'])), zoom=image_zoom)
        edgecolor = palette[row[hue]] if hue is not None else "black"
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=True,
                            bboxprops=dict(boxstyle="square,pad=0.3", fc="white", ec=edgecolor, lw=5))#dict(edgecolor=edgecolor,edgewidth=3))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(dataDf[[x,y]])
    ax.autoscale()
    if not showAxes: ax.axis("off")
    # plt.show()
    if outName is not None: plt.savefig(outName)
    return ax
