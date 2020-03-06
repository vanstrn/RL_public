"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

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
class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
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

        self.debug =debug
        self.actionSize = actionSize

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
        if scope is None:
            namespace = data["NetworkName"]
        else:
            namespace = scope
        super(Network,self).__init__(name=namespace)
        # Reading in the configFile
        if data["NetworkBuilder"] != "network":
            self.built = False
            return

        self.networkOutputs = data["NetworkOutputs"]

        self.scope=namespace

        self.layerList = {}
        self.layerInputs = {}
        self.layerType = {}
        self.multiOutput = {}
        self.layerGroupList = {}
        self.varGroupings = {}

        #Creating Recursion sweep to go through dictionaries and lists in the networkConfig to insert user defined values.
        if "DefaultParams" in data.keys():
            data["NetworkStructure"] = update(data["NetworkStructure"],data["DefaultParams"])

            #Creating all of the layers
        for sectionName,layerList in data["NetworkStructure"].items():
            self.varGroupings[sectionName] = []
            for layerDict in layerList:
                self.layerList[layerDict["layerName"]] = self.GetLayer(layerDict)
                self.layerInputs[layerDict["layerName"]] = layerDict["layerInput"]
                if "multiOutput" in layerDict:
                    self.multiOutput[layerDict["layerName"]] = layerDict["multiOutput"]
                else:
                    self.multiOutput[layerDict["layerName"]] = None
                self.layerGroupList[layerDict["layerName"]] = sectionName
        self.networkVariables=data["NetworkVariableGroups"]
        self.data=data

        self.testInputs = data["TestInput"]
        self.built = True

    def call(self,inputs):
        """Defines how the layers are called with a forward pass of the network.
        The methodology employed assumes sections and layers of the network are stictly forward pass.
        """
        if not self.built:
            return {}
        self.layerOutputs = {}
        for layerName,layer in self.layerList.items():
            if self.debug: print("Creating Layer:",layerName)
            if isinstance(self.layerInputs[layerName], list): #Multi-input Layers
                layerInputs = []
                for layerInput in self.layerInputs[layerName]:
                    if "input" in layerInput:
                        _, input_name = layerInput.rsplit('.',1)
                        layerInputs.append(inputs[input_name])
                    else:
                        layerInputs.append(self.layerOutputs[layerInput])
                if self.debug: print("\tLayer Inputs",layerInputs)
                if self.layerType[layerName] == "Concatenate" or self.layerType[layerName] == "Multiply" or self.layerType[layerName] == "Add":
                    output = layer(layerInputs)
                if self.layerType[layerName] == "GaussianNoise":
                    output = layer(layerInputs,training=training)
                else:
                    output = layer(*layerInputs)
            else: # Single input layers
                if "input" in self.layerInputs[layerName]:
                    _, input_name = self.layerInputs[layerName].rsplit('.',1)
                    if self.debug: print("\tLayer Input",inputs[input_name])
                    output = layer(inputs[input_name])
                else:
                    if self.debug: print("\tLayer Input",self.layerOutputs[self.layerInputs[layerName]])
                    output = layer(self.layerOutputs[self.layerInputs[layerName]])

            #If a layer has multiple outputs assign the outputs unique names. Otherwise just have output be the layername.
            if isinstance(self.multiOutput[layerName],list):
                for i,output_i in enumerate(output):
                    self.layerOutputs[self.multiOutput[layerName][i]]=output_i
            else:
                self.layerOutputs[layerName] = output
            if self.debug: print("\tLayer Output",self.layerOutputs[layerName])

            #Collecting Trainable Variable Weights
            # if self.debug: print("\tLayer Details:",layer.trainable_weights)
            try: self.varGroupings[self.layerGroupList[layerName]] += layer.variables
            except: pass

        results = {}
        for outputName,layerOutput in self.networkOutputs.items():
            results[outputName] = self.layerOutputs[layerOutput]
        return results

    def GetLayer(self, dict):
        """Based on a dictionary input the function returns the appropriate layer for the NN."""
        if "ReuseLayer" in dict:
            layer = self.layerList[dict["ReuseLayer"]]
        else:
            if dict["layerType"] == "Dense":
                if dict["Parameters"]["units"] == "actionSize":
                    dict["Parameters"]["units"] = self.actionSize
                layer = KL.Dense( **dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "Conv2D":
                layer = KL.Conv2D( **dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "Conv2DTranspose":
                layer = KL.Conv2DTranspose( **dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "SeparableConv":
                layer = KL.SeparableConv2D( **dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "Round":
                layer= RoundingSine(name=dict["layerName"])
                # layer= RoundingSine(**dict["Parameters"],name=dict["layerName"])
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
        self.layerType[dict["layerName"]] = dict["layerType"]

        return layer

    def getVars(self,scope=None):
        vars = []
        for name,var in self.varGroupings.items():
            vars += var
        return list(set(vars))

    def GetVariables(self,groupName):
        vars =[]
        for sectionName in self.networkVariables[groupName]:
            vars += self.varGroupings[sectionName]
        return vars

if __name__ == "__main__":
    # sess = tf.Session()
    # test = Network(configFile="test.json",actionSize=4)
    test = {"K1":1,
            "K2":2,
            "K3":"V1",
            "K4":2,
            "K5":{"test":["V1","V2"]},
            "K6":"V2",
            "K7":2,
            }
    test2 = {"V1":4,"V2":5}
    dict=update(test,test2)
    # ReplaceValues(test,test2)
    print(dict)
