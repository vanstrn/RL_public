"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn
from .layers.approx_round import *

class ACNetwork(tf.keras.Model):
    def __init__(self, actionSize, scope="ActorCritic"):
        super(ACNetwork,self).__init__(name=scope)
        self.scope=scope

        #Defining all of the Layers of the network.
        self.conv1 = KL.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = KL.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.flatten = KL.Flatten()
        self.dense = KL.Dense(256)

        self.actor_dense1 = KL.Dense(128)
        self.actor_dense2 = KL.Dense(actionSize)
        self.softmax = KL.Activation('softmax')

        self.critic_dense = KL.Dense(1)


    def call(self,state):
        """Defines how the layers are called with a forward pass of the network.
        The methodology employed assumes sections and layers of the network are stictly forward pass.
        """
        #Defining how the common layers of the network are used.
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)

        #Defining use of actor only layers
        actor = self.actor_dense1(x)
        actor = self.actor_dense2(actor)
        # log_logits = tf.nn.log_softmax(actor)
        actor = self.softmax(actor)

        #Defining use of Critic Parameters.
        critic = self.critic_dense(x)

        return actor, critic

    @property
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
