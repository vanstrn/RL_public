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

if True:
    mypath = "./logs"
    dictsUniform = []
    dicts = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        if len(filenames) != 0:
            if "MG4R_v1_SF_256_" in dirpath:
                if "Uniform" in dirpath:
                    dictsUniform.append(GetFileContents(dirpath+"/"+filenames[0]))
                else:
                    dicts.append(GetFileContents(dirpath+"/"+filenames[0]))
    ave1,std1=GetAverageSTD(dicts)
    ave2,std2=GetAverageSTD(dictsUniform)
    x = dicts[0]["TotalReward"]["step"]

    # plt.errorbar(y,average,yerr=std)
    # plt.errorbar(y,average2,yerr=std2)
    plt.plot(x,ave1,label="On Policy - Samples:"+str(len(dicts)))
    plt.plot(x,ave2,label="Uniform Sampling - Samples:"+str(len(dictsUniform)))
    plt.title("Effect of Sampling (N=256)")
    plt.fill_between(x, ave1-std1, ave1+std1,alpha=0.5)
    plt.fill_between(x, ave2-std2, ave2+std2,alpha=0.5)
    plt.legend()
    plt.show()

if False:
    mypath = "./logs"
    dictsUniform = {"64":[],"128":[],"256":[],"512":[],"1024":[]}
    dicts = {"64":[],"128":[],"256":[],"512":[],"1024":[]}
    for (dirpath, dirnames, filenames) in walk(mypath):
        if len(filenames) != 0:
            if "MG4R_v1_SF_" in dirpath:
                SFSize = dirpath.split("_")[3]
                if "Uniform" in dirpath:
                    dictsUniform[SFSize].append(GetFileContents(dirpath+"/"+filenames[0]))
                else:
                    dicts[SFSize].append(GetFileContents(dirpath+"/"+filenames[0]))

    ave1,std1=GetAverageSTD(dicts["64"])
    ave2,std2=GetAverageSTD(dicts["128"])
    ave3,std3=GetAverageSTD(dicts["256"])
    ave4,std4=GetAverageSTD(dicts["512"])
    ave5,std5=GetAverageSTD(dicts["1024"])
    x = dicts["1024"][0]["TotalReward"]["step"]

    plt.title("Effect of N=[64,128,256,512,1024] ")
    plt.plot(x,ave1,label="64 On Policy - Samples:"+str(len(dicts["64"])))
    plt.fill_between(x, ave1-std1, ave1+std1,alpha=0.3)
    plt.plot(x,ave2,label="128 On Policy - Samples:"+str(len(dicts["128"])))
    plt.fill_between(x, ave2-std2, ave2+std2,alpha=0.3)
    plt.plot(x,ave3,label="256 On Policy - Samples:"+str(len(dicts["256"])))
    plt.fill_between(x, ave3-std3, ave3+std3,alpha=0.3)
    plt.plot(x,ave4,label="512 On Policy - Samples:"+str(len(dicts["512"])))
    plt.fill_between(x, ave4-std4, ave4+std4,alpha=0.3)
    plt.plot(x,ave5,label="1024 On Policy - Samples:"+str(len(dicts["1024"])))
    plt.fill_between(x, ave5-std5, ave5+std5,alpha=0.3)
    plt.legend()
    plt.show()


    ave1,std1=GetAverageSTD(dictsUniform["64"])
    ave2,std2=GetAverageSTD(dictsUniform["128"])
    ave3,std3=GetAverageSTD(dictsUniform["256"])
    ave4,std4=GetAverageSTD(dictsUniform["512"])
    ave5,std5=GetAverageSTD(dictsUniform["1024"])
    x = dictsUniform["1024"][0]["TotalReward"]["step"]
    plt.title("Effect of N=[64,128,256,512,1024] ")
    plt.plot(x,ave1,label="64 Uniform - Samples:"+str(len(dictsUniform["64"])))
    plt.fill_between(x, ave1-std1, ave1+std1,alpha=0.3)
    plt.plot(x,ave2,label="128 Uniform - Samples:"+str(len(dictsUniform["128"])))
    plt.fill_between(x, ave2-std2, ave2+std2,alpha=0.3)
    plt.plot(x,ave3,label="256 Uniform - Samples:"+str(len(dictsUniform["256"])))
    plt.fill_between(x, ave3-std3, ave3+std3,alpha=0.3)
    plt.plot(x,ave4,label="512 Uniform - Samples:"+str(len(dictsUniform["512"])))
    plt.fill_between(x, ave4-std4, ave4+std4,alpha=0.3)
    plt.plot(x,ave5,label="1024 Uniform - Samples:"+str(len(dictsUniform["1024"])))
    plt.fill_between(x, ave5-std5, ave5+std5,alpha=0.3)
    plt.legend()
    plt.show()
