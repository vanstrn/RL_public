"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json


class Network(tf.keras.Model):
    def __init__(self, configFile, actionSize):
        """
        Reads a network config file and processes that into a netowrk with appropriate naming structure.

        This class only works on feed forward neural networks.
        Can only handle one input.
        """
        self.actionSize = actionSize

        with open(configFile) as json_file:
            data = json.load(json_file)
        namespace =data["NetworkName"]
        super(Network,self).__init__(name=namespace)
        # Reading in the configFile
        self.networkOutputs = data["NetworkOutputs"]

        self.scope=namespace

        self.layerList = {}
        self.layerInputs = {}
        with tf.variable_scope(self.scope):
            #Creating all of the layers
            for sectionName,layerList in data["NetworkStructure"].items():
                for layerDict in layerList:
                    self.layerList[layerDict["layerName"]] = self.GetLayer(layerDict)
                    self.layerInputs[layerDict["layerName"]] = layerDict["layerInput"]


        #Creating the Output Dictionary based on the config file

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

        elif dict["layerType"] is "Conv":
            layer = Conv2D(     filters=dict["filters"],
                                kernel_size=dict["kernel_size"],
                                strides=dict["strides"],
                                activation=dict["activation"])

        return layer


    @property
    def getVars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

if __name__ == "__main__":
    sess = tf.Session()
    test = Network(configFile="test.json",actionSize=4)
    s = tf.placeholder(tf.float32, [None, 4], 'S')
    x = test(s)
