"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn

class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize,netConfigOverride,scope=None):
        """
        Reads a network config file and processes that into a netowrk with appropriate naming structure.

        This class only works on feed forward neural networks.
        Can only handle one input.
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

    def call(self,input):
        """Defines how the layers are called with a forward pass of the network.
        The methodology employed assumes sections and layers of the network are stictly forward pass.
         """
        self.layerOutputs = {}
        for layerName,layer in self.layerList.items():
            if self.layerInputs[layerName] == "input":
                self.layerOutputs[layerName] = layer(input)
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
            elif dict["layerType"] == "SeparableConv":
                layer = KL.SeparableConv2D( filters=dict["filters"],
                                            kernel_size=dict["kernel_size"],
                                            strides=dict["strides"],
                                            padding=dict["padding"],
                                            depth_multiplier=dict["depth_multiplier"],
                                            name=dict["layerName"])

            elif dict["layerType"] == "Flatten":
                layer= KL.Flatten()
            elif dict["layerType"] == "NonLocalNN":
                layer= Non_local_nn(channels=dict["channels"])
            elif dict["layerType"] == "LogSoftMax":
                layer = tf.nn.log_softmax
            elif dict["layerType"] == "SoftMax":
                layer = KL.Activation('softmax')

        return layer

    @property
    def getVars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

if __name__ == "__main__":
    sess = tf.Session()
    test = Network(configFile="test.json",actionSize=4)
    # s = tf.placeholder(tf.float32, [None, 4], 'S')
    # x = test(s)
