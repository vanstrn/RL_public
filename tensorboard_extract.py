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

def GetAverageSTD(dicts,metric):
    length=0
    #Finding the longest array
    for dict in dicts:
        if len(dict[metric]["data"])>length:
            length = len(dict[metric]["data"])
    data = []
    for dict in dicts:
        if len(dict[metric]["data"])==length:
            data.append(dict[metric]["data"])
        else:
            for i in range(length-len(dict[metric]["data"])):
                dict[metric]["data"].append(np.nan)
            data.append(dict[metric]["data"])
    average = np.nanmean(np.stack(data),axis=0)
    std = 1.00*np.nanstd(np.stack(data),axis=0)/np.sqrt(len(dicts))
    return average,std


def PlotTensorflowData(dataName,dataSeparations,metric="TotalReward",path="./logs",dataLabels=None,title="Effect of Sampling Methods"):
    if dataLabels is None:
        dataLabels = dataSeparations
    dataFiles = {}
    for name,label in zip(dataSeparations,dataLabels):
        dataFiles[label]=[]
        for (dirpath, dirnames, filenames) in walk(path):
            if len(filenames) != 0:
                if dataName in dirpath:
                    if name in dirpath:
                        for filename in filenames:
                            if "events.out.tfevents" in filename:
                                dataFiles[label].append(GetFileContents(dirpath+"/"+filename))
                                continue

    for label,data in dataFiles.items():
        ave,std=GetAverageSTD(data,metric)
        x = np.arange(0,10*ave.shape[0],10)
        plt.plot(x,ave,label=label+" sampling - Trials:"+str(len(data)))
        plt.fill_between(x, ave-std, ave+std,alpha=0.5)
        plt.ylim([0.0,1.0])
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":

    #
    # title = "Effect of Different Sample Methods (N=128,32 Samples)"
    # dataName = "MG4R_SF_128_48"
    # dataSeparations = ["HC_1588","H_1588","F_1588","R_1588"]
    # dataLabels = ["Cluster","Hull","First","Random"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Different Sample Methods (N=256,32 Samples)"
    # dataName = "MG4R_SF_256_48"
    # dataSeparations = ["HC_1588","H_1588","F_1588","R_1588"]
    # dataLabels = ["Cluster","Hull","First","Random"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Different Sample Methods (N=512,32 Samples)"
    # dataName = "MG4R_SF_512_48"
    # dataSeparations = ["HC_1588","H_1588","F_1588","R_1588"]
    # dataLabels = ["Cluster","Hull","First","Random"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    #
    #Coverage percent effect on different sampling methods
    dataName = "MG4R_SF_"
    # title = "Effect of Coverage Percentage on Convex Hull (All N)"
    # dataSeparations = ["_32H_1588","_48H_1588","_64H_1588"]
    # dataLabels = ["30% coverage","45% coverage","60% coverage"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Coverage Percentage on Clustering (All N)"
    # dataSeparations = ["_32HC","_48HC","_64HC"]
    # dataLabels = ["30% coverage","45% coverage","60% coverage"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Coverage Percentage on First (All N)"
    # dataSeparations = ["_32F","_48F","_64F"]
    # dataLabels = ["30% coverage","45% coverage","60% coverage"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Coverage Percentage on Random (All N)"
    # dataSeparations = ["_32R","_48R","_64R"]
    # dataLabels = ["30% coverage","45% coverage","60% coverage"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Sample Method (All N)60% Coverage"
    # dataSeparations = ["_64H_1588","_64F","_64R","_64HC"]
    # dataLabels = ["Hull","First","Random","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Sample Method (All N)60% Coverage"
    # dataSeparations = ["_48H_1588","_48F","_48R","_48HC"]
    # dataLabels = ["Hull","First","Random","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Sample Method (All N)60% Coverage"
    # dataSeparations = ["_32H_1588","_32F","_32R","_32HC"]
    # dataLabels = ["Hull","First","Random","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)


    # title = "Effect of Different Sample Methods (N=512,64 Samples) SF1"
    # dataName = "CTF_"
    # dataSeparations = ["64HC","64HP","64F","64R","64HT"]
    # dataLabels = ["Cluster","Hull","First","Random","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Different Sample Methods (N=512,64 Samples) SF2"
    # dataName = "CTF_16_64"
    # dataSeparations = ["64HC_3","64HP_3","64F_3","64R_3","64HT_3"]
    # dataLabels = ["Cluster","Hull","First","Random","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Different Sample Methods (N=512,64 Samples) SF3"
    # dataName = "CTF_16_64"
    # dataSeparations = ["64HC_2","64HP_2","64F_2","64R_2","64HT_2"]
    # dataLabels = ["Cluster","Hull","First","Random","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # title = "Effect of Different Sample Methods (N=512,64 Samples) All Trials"
    # dataName = "CTF_16_64"
    # dataSeparations = ["64HC","64HP","64F","64R","64HT"]
    # dataLabels = ["Cluster","Hull","First","Random","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)



    # dataName = "MG4R_SF_"
    # title = "Effect of Sample Method 30% Coverage"
    # dataSeparations = ["32HP","32FS","32RE","32RN","32HC", "32HT"]
    # dataLabels = ["Hull","First","Random*","Random","Cluster","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,path="./logs_/MG4R_SF_2",dataLabels=dataLabels,title=title)
    # title = "Effect of Sample Method 45% Coverage"
    # dataSeparations = ["48HP","48FS","48RE","48RN","48HC", "48HT"]
    # dataLabels = ["Hull","First","Random*","Random","Cluster","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,path="./logs_/MG4R_SF_2",dataLabels=dataLabels,title=title)
    # title = "Effect of Sample Method 20% Coverage"
    # dataSeparations = ["20HP","20FS","20RE","20RN","20HC", "20HT"]
    # dataLabels = ["Hull","First","Random*","Random","Cluster","Hull TSNE"]
    # PlotTensorflowData(dataName,dataSeparations,path="./logs_/MG4R_SF_2",dataLabels=dataLabels,title=title)



    # dataName = "MG4R_v2_SFaH_128"
    # title = "Effect of Sample Method 50% Coverage"
    # dataSeparations = ["HP","PPO_FS_FS","RS","RN","HC"]
    # dataLabels = ["Hull","First","Random*","Random","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # dataName = "MG4R_v2_SFaH_96"
    # title = "Effect of Sample Method 37% Coverage"
    # dataSeparations = ["HP","PPO_FS_FS","RS","RN","HC"]
    # # dataLabels = ["Hull","First","Random*","Random","Cluster"]
    # dataLabels = ["Random","First","Hull","Random*","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # # dataName = "MG4R_v2_SFaH_64"
    # # title = "Effect of Sample Method 25% Coverage"
    # # dataSeparations = ["HP","PPO_FS_FS","RS","RN","HC"]
    # # dataLabels = ["Hull","First","Random*","Random","Cluster"]
    # # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)
    # dataName = "MG4R_v2_SFaH_48"
    # title = "Effect of Sample Method 18% Coverage"
    # dataSeparations = ["HP","PPO_FS_FS","RS","RN","HC"]
    # dataLabels = ["Random","First","Random*","Hull","Cluster"]
    # PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title)

    dataName = "2_3_SF_OPT"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["12","16","20","24","28","32","36"]
    dataLabels = ["12","16","20","24","28","32","36"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
    dataName = "2_3_SF_Gamma"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["2_3_SF_Gamma99E1","2_3_SF_Gamma98E1","2_3_SF_Gamma95E1","2_3_SF_Gamma9E1"]
    dataLabels = ["2_3_SF_Gamma99E1","2_3_SF_Gamma98E1","2_3_SF_Gamma95E1","2_3_SF_Gamma9E1"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
    dataName = "2_3_SF_FS"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["FS1","FS2","FS3","FS4","FS5","FS6","FS7"]
    dataLabels = ["FS1","FS2","FS3","FS4","FS5","FS6","FS7"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
    dataName = "2_3_SF_ENTLR"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["ENTLR1E7_BETA1E5","ENTLR1E7_BETA1E4","ENTLR1E7_BETA1E3","ENTLR1E7_BETA1E2","ENTLR1E7_BETA1E1","ENTLR1E6_BETA1E5","ENTLR1E6_BETA1E4","ENTLR1E6_BETA1E3","ENTLR1E6_BETA1E2","ENTLR1E6_BETA1E1"]
    dataLabels = ["ENTLR1E7_BETA1E5","ENTLR1E7_BETA1E4","ENTLR1E7_BETA1E3","ENTLR1E7_BETA1E2","ENTLR1E7_BETA1E1","ENTLR1E6_BETA1E5","ENTLR1E6_BETA1E4","ENTLR1E6_BETA1E3","ENTLR1E6_BETA1E2","ENTLR1E6_BETA1E1"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
    dataName = "2_3_SF_BS"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["2_3_SF_BS256_MB32","2_3_SF_BS256_MB64","2_3_SF_BS512_MB32","2_3_SF_BS512_MB64","2_3_SF_BS1024_MB32","2_3_SF_BS1024_MB64"]
    dataLabels = ["2_3_SF_BS256_MB32","2_3_SF_BS256_MB64","2_3_SF_BS512_MB32","2_3_SF_BS512_MB64","2_3_SF_BS1024_MB32","2_3_SF_BS1024_MB64"]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
    dataName = "2_3_SF_ALR"
    title = "Effect of Sample Method 18% Coverage"
    dataSeparations = ["2_3_SF_ALR1E4","2_3_SF_ALR1E5","2_3_SF_ALR1E6","2_3_SF_ALR5E5","2_3_SF_ALR5E6",]
    dataLabels = ["2_3_SF_ALR1E4","2_3_SF_ALR1E5","2_3_SF_ALR1E6","2_3_SF_ALR5E5","2_3_SF_ALR5E6",]
    PlotTensorflowData(dataName,dataSeparations,dataLabels=dataLabels,title=title,path="/home/neale/HP1")
