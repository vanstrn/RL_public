import tensorflow as tf
from os import walk
import csv
import numpy as np
import matplotlib.pyplot as plt

def GetFileContents(path):
    """Reads a tensorflow summary file and extracts data into a structured dictionary """
    dataDict = {}
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag not in dataDict:
                dataDict[v.tag] = {}
                dataDict[v.tag]["data"] = []
                dataDict[v.tag]["step"] = []
            # print(dir(v))
            # print(dir(e))
            dataDict[v.tag]["data"].append(v.simple_value)
            dataDict[v.tag]["step"].append(e.step)

    return dataDict

def GetAverageSTD(dicts):
    length=0
    #Finding the longest array
    for dict in dicts:
        if len(dict["TotalReward"]["data"])>length:
            length = len(dict["TotalReward"]["data"])
    data = []
    for dict in dicts:
        if len(dict["TotalReward"]["data"])==length:
            data.append(dict["TotalReward"]["data"])
        else:
            for i in range(length-len(dict["TotalReward"]["data"])):
                dict["TotalReward"]["data"].append(np.nan)
    average = np.nanmean(np.stack(data),axis=0)
    std = 1.96*np.nanstd(np.stack(data),axis=0)/np.sqrt(len(dicts))
    return average,std


def PlotTensorflowData(dataName,dataSeparations,path="./logs",dataLabels=None,title="Effect of Sampling Methods"):
    if dataLabels is None:
        dataLabels = dataSeparations
    dataFiles = {}
    for name,label in zip(dataSeparations,dataLabels):
        dataFiles[label]=[]
        for (dirpath, dirnames, filenames) in walk(path):
            if len(filenames) != 0:
                if dataName in dirpath:
                    if name in dirpath:
                        dataFiles[label].append(GetFileContents(dirpath+"/"+filenames[0]))

    for label,data in dataFiles.items():
        ave,std=GetAverageSTD(data)
        x = np.arange(0,10*ave.shape[0],10)
        plt.plot(x,ave,label=label+" sampling - Trials:"+str(len(data)))
        plt.fill_between(x, ave-std, ave+std,alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":

    title = "Convex Hull - Effect of Coverage Percentage"
    dataName = "MG4R_SF_256_"
    dataSeparations = ["HC_1588","H_1588","F_1588","R_1588"]
    dataLabels = ["Clustering","Hull","First","Random"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels)

    title = "Effect of Coverage Percentage- Convex Hull"
    dataSeparations = ["_32H_1588","_48H_1588","_64H_1588"]
    dataLabels = ["30% coverage","45% coverage","60% coverage"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    title = "Effect of Coverage Percentage- Clustering"
    dataSeparations = ["_32HC","_48HC","_64HC"]
    dataLabels = ["30% coverage","45% coverage","60% coverage"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    title = "Effect of Coverage Percentage- First"
    dataSeparations = ["_32F","_48F","_64F"]
    dataLabels = ["30% coverage","45% coverage","60% coverage"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    title = "Effect of Coverage Percentage- Random"
    dataSeparations = ["_32R","_48R","_64R"]
    dataLabels = ["30% coverage","45% coverage","60% coverage"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
