"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn
from .layers.approx_round import *

class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=False):
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

        with open(configFile) as json_file:
            data = json.load(json_file)
        data.update(netConfigOverride)
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
                if self.layerType[layerName] == "Concatenate":
                    output = layer(layerInputs)
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
                layer= RoundingSine()
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
            elif dict["layerType"] == "Reshape":
                layer = KL.Reshape( **dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "LSTM":
                layer = KL.LSTM(**dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "SimpleRNN":
                layer = KL.SimpleRNN(**dict["Parameters"],name=dict["layerName"])
            elif dict["layerType"] == "Sum":
                layer = tf.keras.backend.sum
        self.layerType[dict["layerName"]] = dict["layerType"]

        return layer

    def getVars(self,scope=None):
        if scope is None:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/" + self.scope)

    def GetVariables(self,groupName):
        vars =[]
        for sectionName in self.networkVariables[groupName]:
            vars += self.varGroupings[sectionName]
        return vars

if __name__ == "__main__":
    sess = tf.Session()
    test = Network(configFile="test.json",actionSize=4)
