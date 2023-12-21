import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def LoadPatientData(patientId, dataDir):
    patientDataDf = pd.read_csv(os.path.join(dataDir, "patient%.3d.txt" % patientId), header=None)
    # patientDataDf = pd.read_csv(dataDir, header=None)
    patientDataDf.rename(columns={0: "PatientId", 1: "Date", 2: "CPA", 3: "LEU", 4: "PSA", 5: "Testosterone",
                                  6: "CycleId", 7: "DrugConcentration"}, inplace=True)
    patientDataDf['Date'] = pd.to_datetime(patientDataDf.Date)
    patientDataDf = patientDataDf.sort_values(by="Date")
    patientDataDf['Time'] = patientDataDf[8] - patientDataDf.iloc[0, 8]
    patientDataDf['PSA_raw'] = patientDataDf.PSA
    patientDataDf['PSA'] /= patientDataDf.PSA.iloc[0]
    return patientDataDf


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), *("data/Fig0a/dataTanaka/Bruchovsky_et_al/".split('/')))
    files = glob.glob(os.path.join(data_dir, "patient*.txt"))
    fig, axs = plt.subplots(8, 9, figsize=(30, 20), sharex=True, sharey=True)
    plt.rcParams['font.size'] = '24'

    for i, file in enumerate(sorted(files)):
        patient_id = file.split('.')[-2][-3:]
        dataDf = LoadPatientData(int(patient_id), data_dir)
        axs.flatten()[i].plot(dataDf.Time, dataDf.PSA / max(dataDf.PSA),
            linestyle="None", marker=".", color="black")#, markeredgewidth=4)
        axs.flatten()[i].text(0.65, 0.8, patient_id, transform=axs.flatten()[i].transAxes)

    fig.suptitle("Patients from Bruchovsky trial"); fig.supxlabel("Treatment progression (days)"); fig.supylabel("PSA")
    plt.savefig("../paper_figures/figures/all_bruchovsky_patients.png")
