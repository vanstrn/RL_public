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

def update(d, u):
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), u)
        elif isinstance(v,list):
            list_new = []
            for val in v:
                if isinstance(val, collections.abc.Mapping):
                    tmp_dict = update(val, u)
                    list_new.append(tmp_dict)
                elif isinstance(val, list):
                    pass
                elif val in u.keys():
                    list_new.append(u[val])
                else:
                    list_new.append(val)
            d[k] = list_new
        else:
            if v in u.keys():
                d[k] = u[v]
    return d
def Update(defaultSettings,overrides):
    for label,override in overrides.items():
        if isinstance(override, collections.abc.Mapping):
            Update(defaultSettings[label],override)
        else:
            defaultSettings[label] = override
    return defaultSettings

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
    data = Update(data,netConfigOverride)

    # if data["NetworkBuilder"] != "network_v2":
    #     return

    layers = {}
    layerOutputs = {}

    #Creating Recursion sweep to go through dictionaries and lists in the networkConfig to insert user defined values.
    if "DefaultParams" in data.keys():
        data["NetworkStructure"] = update(data["NetworkStructure"],data["DefaultParams"])

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


def GetLayer( dict):
    """Based on a dictionary input the function returns the appropriate layer for the NN."""

    if dict["layerType"] == "Dense":
        layer = KL.Dense( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Conv2D":
        layer = KL.Conv2D( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Conv2DTranspose":
        layer = KL.Conv2DTranspose( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "SeparableConv":
        layer = KL.SeparableConv2D( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Round":
        layer= RoundingSine(name=dict["layerName"])
    elif dict["layerType"] == "Flatten":
        layer= KL.Flatten()
    elif dict["layerType"] == "NonLocalNN":
        layer= Non_local_nn( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "LogSoftMax":
        layer = tf.nn.log_softmax
    elif dict["layerType"] == "SoftMax":
        layer = KL.Activation('softmax')
    elif dict["layerType"] == "Concatenate":
        layer = KL.Concatenate( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Multiply":
        layer = KL.Multiply( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Add":
        layer = KL.Add( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Reshape":
        layer = KL.Reshape( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "LSTM":
        layer = KL.LSTM(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "SimpleRNN":
        layer = KL.SimpleRNN(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Sum":
        layer = tf.keras.backend.sum
    elif dict["layerType"] == "Inception":
        layer = Inception(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "ReverseInception":
        layer = ReverseInception(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "UpSampling2D":
        layer = KL.UpSampling2D(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "GaussianNoise":
        layer = KL.GaussianNoise(**dict["Parameters"],name=dict["layerName"])

    return layer


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.25, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(config=config)
    test = buildNetwork(configFile="test.json",actionSize=4)
    test.compile(optimizer="adam", loss={"StatePrediction":"mse","Phi":"mse"})
    x = np.random.rand(2,19,19,2)
    y = np.random.rand(2,256)
    test.fit({"state":x},{"StatePrediction":x,"Phi":y})
