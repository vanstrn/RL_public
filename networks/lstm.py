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

        self.conv1 = KL.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = KL.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.flatten = KL.Flatten()
        self.reshape = KL.Reshape(target_shape=[1,32])
        self.LSTM = KL.LSTM(256, return_state=True)
        self.dense = KL.Dense(256)
        self.actor_dense1 = KL.Dense(128)
        self.actor_dense2 = KL.Dense(actionSize)
        self.softmax = KL.Activation('softmax')
        self.critic_dense = KL.Dense(1)


    def call(self,inputs):
        """Defines how the layers are called with a forward pass of the network.
        The methodology employed assumes sections and layers of the network are stictly forward pass.
         """
        inputState = inputs["state"]
        hiddenInputState = inputs["hiddenState"]

        x = self.conv1(inputState)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x,h_c,h_s = self.LSTM(x,initial_state=hiddenInputState)
        x = self.dense(x)

        actor = self.actor_dense1(x)
        actor = self.actor_dense2(actor)
        log_logits = tf.nn.log_softmax(actor)
        actor = self.softmax(actor)

        critic = self.critic_dense(x)


        results= {"critic":critic,
              "actor":actor,
              "log_logits":log_logits,
              "hiddenState":[h_c,h_s]}
        return results

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
