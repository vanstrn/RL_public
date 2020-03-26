"""
Attempt to replicate features of the Functional Model within a sub-classed model.

# TODO:
-Create a model that has dictionary inputs and outputs.
-Ability to run fit method with said dictionary outputs.
-Ability to control which variables are updated with the fit method.
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn
from .layers.approx_round import *
from .layers.inception import Inception
from .layers.reverse_inception import ReverseInception

import collections.abc
import os

from .common import *

def buildNetwork(configFile, actionSize, netConfigOverride={},debug=True, training=True, scope=None):
    """
    Reads a network config file and processes that into a netowrk with appropriate naming structure.

    This class only works on feed forward neural networks. Can only handle one input.

    Parameters
    ----------
    configFile : str
        Config file which points to the network description to be loaded.
    actionSize : int
        Output sizes of different network components. Assumes the environment is discrete action space.
    netConfigOverride : dict
        Dictionary of values which will override the default contents loaded from the config file.
    scope : str [opt]
        Defines a tensorflow scope for the network.

    Returns
    -------
    N/A
    """

    #Checking if JSON file is fully defined path or just a file name without path.
    #If just a name, then it will search in default directory.
    if "/" in configFile:
        if ".json" in configFile:
            pass
        else:
            configFile = configFile + ".json"
    else:
        for (dirpath, dirnames, filenames) in os.walk("configs/network"):
            for filename in filenames:
                if configFile in filename:
                    configFile = os.path.join(dirpath,filename)
                    break
        # raise
    with open(configFile) as json_file:
        data = json.load(json_file)
    data = UpdateNestedDictionary(data,netConfigOverride)

    # if data["NetworkBuilder"] != "network_v2":
    #     return

    layers = {}
    layerOutputs = {}

    #Creating Recursion sweep to go through dictionaries and lists in the networkConfig to insert user defined values.
    if "DefaultParams" in data.keys():
        data["NetworkStructure"] = UpdateStringValues(data["NetworkStructure"],data["DefaultParams"])

    #Creating Inputs for the network
    Inputs = {}
    for inputName,inputShape in data["Inputs"].items():
        Inputs[inputName] = KL.Input(shape=inputShape,name=inputName)

    #Creating layers of the network
    for sectionName,layerList in data["NetworkStructure"].items():
        for layerDict in layerList:
            if "units" in layerDict["Parameters"]:
                if layerDict["Parameters"]["units"] == "actionSize":
                    layerDict["Parameters"]["units"] = actionSize

            if "ReuseLayer" in layerDict:
                layer = layers["ReuseLayer"]
            else:
                layer =  GetLayer(layerDict)
            layers[layerDict["layerName"]] = layer


            if debug: print("Creating Layer:",layerDict["layerName"])
            if isinstance(layerDict["layerInput"], list): #Multi-input Layers
                layerIn = []
                for layerInput in layerDict["layerInput"]:
                    if "input" in layerInput:
                        _, input_name = layerInput.rsplit('.',1)
                        layerIn.append(Inputs[input_name])
                    else:
                        layerIn.append(layerOutputs[layerInput])
                if debug: print("\tLayer Inputs",layerIn)
                if layerType[layerName] == "Concatenate" or layerType[layerName] == "Multiply" or layerType[layerName] == "Add":
                    output = layer(layerIn)
                if layerType[layerName] == "GaussianNoise":
                    output = layer(layerIn,training=training)
                else:
                    output = layer(*layerIn)

            else: # Single input layers
                if "input" in layerDict["layerInput"]:
                    _, input_name = layerDict["layerInput"].rsplit('.',1)
                    if debug: print("\tLayer Input",Inputs[input_name])
                    output = layer(Inputs[input_name])
                else:
                    if debug: print("\tLayer Input",layerOutputs[layerDict["layerInput"]])
                    output = layer(layerOutputs[layerDict["layerInput"]])

            #If a layer has multiple outputs assign the outputs unique names. Otherwise just have output be the layername.
            if "multiOutput" in layerDict:
                for i,output_i in enumerate(output):
                    layerOutputs[multiOutput[layerDict["layerName"]][i]]=output_i
            else:
                layerOutputs[layerDict["layerName"]] = output
            if debug: print("\tLayer Output",layerOutputs[layerDict["layerName"]])


    #Creating the outputs for the model.
    outputs=[]
    for output,layerName in data["NetworkOutputs"].items():
        outputs.append(layerOutputs[layerName])
    if debug: print("Network Outputs:",outputs)
    network = tf.keras.models.Model(Inputs,outputs)

    return network


if __name__ == "__main__":
    pass
