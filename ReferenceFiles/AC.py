"""
Actor Critic Models which can be used for Reinforcement Learning

"""

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

class ActorCritic_DNN(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic_DNN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


class ActorCritic_CNN(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic_CNN, self).__init__()
        self.state_size = state_size
        # print(state_size)
        self.action_size = action_size
        self.conv1 = layers.Conv2D(input_shape=state_size,filters=32,kernel_size=[3,3],strides=(1,1),padding='SAME',activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=[2, 2])
        self.conv2 = layers.Conv2D(filters=64,kernel_size=[3,3],strides=(1,1),padding='SAME',activation='relu')
        self.flat1 = layers.Flatten()
        self.dens1 = layers.Dense(100, activation='relu')
        self.dens2 = layers.Dense(action_size, activation='relu')
        self.values = layers.Dense(1, activation='relu')


    def call(self, inputs):
        #Combined Calculations
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flat1(x)

        #Actor Calculations
        a = self.dens1(x)
        logits = self.dens2(a)

        #Critic Calculations
        values=self.values(x)

        return logits,values

if __name__ == '__main__':
    # ac = ActorCritic_CNN((20,20,6),625)
    ac2 = ActorCritic_DNN(4,2)
