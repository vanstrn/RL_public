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
from tensorflow.keras import backend as K


import collections.abc
import os

from .common import *

class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=False, training=True):
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
        data = UpdateNestedDictionary(data,netConfigOverride)
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
            data["NetworkStructure"] = UpdateStringValues(data["NetworkStructure"],data["DefaultParams"])

            #Creating all of the layers
        for sectionName,layerList in data["NetworkStructure"].items():
            self.varGroupings[sectionName] = []
            for layerDict in layerList:
                if layerDict["layerType"] == "Dense":
                    if "Parameters" in layerDict["layerType"]:
                        if layerDict["Parameters"]["units"] == "actionSize":
                            layerDict["Parameters"]["units"] = self.actionSize
                if "ReuseLayer" in layerDict:
                    self.layerList[layerDict["layerName"]] = self.layerList[layerDict["ReuseLayer"]]
                else:
                    self.layerList[layerDict["layerName"]] = GetLayer(layerDict)
                self.layerInputs[layerDict["layerName"]] = layerDict["layerInput"]
                self.layerType[layerDict["layerName"]] = layerDict["layerType"]
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
                elif self.layerType[layerName] == "GaussianNoise":
                    output = layer(layerInputs,training=training)
                else:
                    output = layer(*layerInputs)
            else: # Single input layers
                if self.layerType[layerName] == "InputNoise":
                    output = layer
                elif "input" in self.layerInputs[layerName]:
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
    pass
