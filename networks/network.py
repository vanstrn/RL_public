"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn
from .layers.approx_round import *

class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize, netConfigOverride, scope=None):
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
        self.networkOutputs = data["NetworkOutputs"]

        self.scope=namespace

        self.layerList = {}
        self.layerInputs = {}
            #Creating all of the layers
        for sectionName,layerList in data["NetworkStructure"].items():
            for layerDict in layerList:
                self.layerList[layerDict["layerName"]] = self.GetLayer(layerDict)
                self.layerInputs[layerDict["layerName"]] = layerDict["layerInput"]

        self.networkVariables=data["NetworkVariableGroups"]

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
            results[outputName] = self.layerOutputs[layerOutput]
        return results

    def GetLayer(self, dict):
        """Based on a dictionary input the function returns the appropriate layer for the NN."""
        if "ReuseLayer" in dict:
            layer = self.layerList[dict["ReuseLayer"]]
        else:
            if dict["layerType"] == "Dense":
                if dict["outputSize"] == "actionSize":
                    output = self.actionSize
                else:
                    output = dict["outputSize"]
                layer = KL.Dense(   output,
                                    activation=dict["activation"],
                                    # kernel_initializer=dict["kernel_initializer"],  # weights
                                    # bias_initializer=dict["bias_initializer"],  # biases
                                    name=dict["layerName"])

            elif dict["layerType"] == "Conv2D":
                layer = KL.Conv2D( filters=dict["filters"],
                                kernel_size=dict["kernel_size"],
                                strides=dict["strides"],
                                activation=dict["activation"],
                                name=dict["layerName"])
            elif dict["layerType"] == "Conv2D":
                layer = KL.Conv2DTranspose( filters=dict["filters"],
                                kernel_size=dict["kernel_size"],
                                strides=dict["strides"],
                                activation=dict["activation"],
                                name=dict["layerName"])
            elif dict["layerType"] == "SeparableConv":
                if "padding" in dict:
                    padding = dict["padding"]
                else:
                    padding="valid"
                layer = KL.SeparableConv2D( filters=dict["filters"],
                                            kernel_size=dict["kernel_size"],
                                            strides=dict["strides"],
                                            padding=dict["padding"],
                                            depth_multiplier=dict["depth_multiplier"],
                                            name=dict["layerName"],
                                            padding=padding)
            elif dict["layerType"] == "Round":
                layer= KL.RoundingSine()
            elif dict["layerType"] == "Flatten":
                layer= KL.Flatten()
            elif dict["layerType"] == "NonLocalNN":
                layer= Non_local_nn(channels=dict["channels"])
            elif dict["layerType"] == "LogSoftMax":
                layer = tf.nn.log_softmax
            elif dict["layerType"] == "SoftMax":
                layer = KL.Activation('softmax')
            elif dict["layerType"] == "Concatenate":
                layer = KL.Concatenate(axis=dict["axis"])
            elif dict["layerType"] == "Reshape":
                layer = KL.Concatenate(target_shape=dict["target_shape"])

        return layer

    def getVars(self,scope=None):
        if scope is None:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/" + self.scope)

    def getVariables(self,name,scope):
        vars = []
        for section in self.networkVariables[name]:
            vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/" + self.scope + "/" + section))
        return vars

if __name__ == "__main__":
    sess = tf.Session()
    test = Network(configFile="test.json",actionSize=4)
