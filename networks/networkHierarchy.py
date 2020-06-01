"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json

from .common import *

class HierarchicalNetwork(tf.keras.Model):
    def __init__(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True):
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
            config = json.load(json_file)
        config = UpdateNestedDictionary(config,netConfigOverride)
        if scope is None:
            namespace = config["NetworkName"]
        else:
            namespace = scope
        super(HierarchicalNetwork,self).__init__(name=namespace)
        # Reading in the configFile

        self.networkOutputs = config["NetworkOutputs"]
        self.networkVariableGroups = config["NetworkVariableGroups"]

        self.scope=namespace

        #Creating Recursion sweep to go through dictionaries and lists in the networkConfig to insert user defined values.
        if "DefaultParams" in config.keys():
            config["NetworkStructure"] = UpdateStringValues(config["NetworkStructure"],config["DefaultParams"])
        config["NetworkStructure"] = UpdateStringValues(config["NetworkStructure"],{"actionSize":self.actionSize})


        self.layerList = {}
        self.layerInputs = {}
        self.layerGroupings = {}
        #Creating all of the layers
        for groupName,groupDict in config["NetworkStructure"].items():
            if groupName == "SubNetworkStructure":
                for option in range(config["DefaultParams"]["NumOptions"]):
                    self.layerGroupings["SubNetworkStructure"+str(option)] = []
                    for sectionName,layerList in groupDict.items():
                        for layerDict in layerList:
                            dict = layerDict.copy()
                            dict["layerName"]= dict["layerName"]+"_option"+str(option)
                            if self.debug: print("Creating Layer: ", dict["layerName"])
                            layer = GetLayer(dict)
                            self.layerList[dict["layerName"]] = layer
                            self.layerGroupings["SubNetworkStructure"+str(option)].append(layer)

                            if self.debug: print("  Layer Output: ", layerDict["layerInput"])
                            if "Sub" in dict["layerInput"]:
                                self.layerInputs[dict["layerName"]] = dict["layerInput"]+"_option"+str(option)
                            else:
                                self.layerInputs[dict["layerName"]] = dict["layerInput"]
                                # self.layerInputs[layerDict["layerName"]+"_option"+str(option)] = layerDict["layerInput"]

            else:
                self.layerGroupings[groupName]=[]
                for sectionName,layerList in groupDict.items():
                    for layerDict in layerList:
                        layer = GetLayer(layerDict)
                        self.layerList[layerDict["layerName"]] = layer
                        self.layerInputs[layerDict["layerName"]] = layerDict["layerInput"]
                        self.layerGroupings[groupName].append(layer)

        self.networkVariables=config["NetworkVariableGroups"]
        self.networkOutputs = config["NetworkOutputs"]
        self.numOptions = config["DefaultParams"]["NumOptions"]

    def call(self,inputs):
        """Defines how the layers are called with a forward pass of the network.
        The methodology employed assumes sections and layers of the network are stictly forward pass.
         """
        self.layerOutputs = {}
        for layerName,layer in self.layerList.items():
            if isinstance(self.layerInputs[layerName], list): #Multi-input Layers
                layerInputs = []
                for layerInput in self.layerInputs[layerName]:
                    if "input" in self.layerInputs[layerName]:
                        _, input_name = self.layerInputs[layerName].rsplit('.',1)
                        layerInputs.append(inputs[input_name])
                    else:
                        layerInputs.append(self.layerOutputs[self.layerInputs[layerName]])
                self.layerOutputs[layerName] = layer(layerInputs)
            else: # Single input layers
                if "input" in self.layerInputs[layerName]:
                    _, input_name = self.layerInputs[layerName].rsplit('.',1)
                    self.layerOutputs[layerName] = layer(inputs[input_name])
                else:
                    self.layerOutputs[layerName] = layer(self.layerOutputs[self.layerInputs[layerName]])

        results = {}
        for outputName,layerOutput in self.networkOutputs.items():
            if "sub" in outputName:
                results[outputName] = []
                for option in range(self.numOptions):
                    results[outputName].append(self.layerOutputs[layerOutput+"_option"+str(option)])
            else:
                results[outputName] = self.layerOutputs[layerOutput]
        return results

    def getVars(self,scope=None):
        if scope is None:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/" + self.scope)

    def getHierarchyVariables(self):
        vars = []
        for group in self.networkVariableGroups["Hierarchy"]:
            for layer in self.layerGroupings[group]:
                if hasattr(layer, 'variables'):
                    vars.extend(layer.variables)
        return vars

    def getSubpolicyVariables(self,option):
        vars = []
        for group in self.networkVariableGroups["SubPolicy"]:
            if group == "SubNetworkStructure":
                for layer in self.layerGroupings[group+str(option)]:
                    if hasattr(layer, 'variables'):
                        vars.extend(layer.variables)
            else:
                for layer in self.layerGroupings[group]:
                    if hasattr(layer, 'variables'):
                        vars.extend(layer.variables)
        return vars


if __name__ == "__main__":
    import numpy as np
    sess = tf.Session()
    test = HierarchicalNetwork(configFile="hierarchyTest.json",actionSize=4)
    s = tf.placeholder(tf.float32, [None,39,39,6], 'S')
    state={"state":s}
    out = test(state)
