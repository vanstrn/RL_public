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


def FFNetwork(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
    input_img = KL.Input(shape=(19,19,2))
    flat = KL.Flatten()(input_img)
    l1 = KL.Dense(400,activation="relu")(flat)
    l2 = KL.Dense(200,activation="relu")(l1)
    encoded = KL.Dense(256,activation="relu")(l2)
    l2_ = KL.Dense(200,activation="relu")(encoded)
    l1_ = KL.Dense(400,activation="relu")(l2_)
    out_ = KL.Dense(722,activation=None)(l1_)
    output = KL.Reshape((19,19,2))(out_)


    network = tf.keras.models.Model(input_img,output)
    return network

def FFNetwork2(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
    input_img = KL.Input(shape=(19,19,2))
    conv1 = KL.Conv2D(filters=16,kernel_size=2,strides=1,activation="elu")(input_img)
    conv2 = KL.Conv2D(filters=32,kernel_size=2,strides=1,activation="elu")(conv1)
    flat = KL.Flatten()(conv2)
    l1 = KL.Dense(200,activation="relu")(flat)
    encoded = KL.Dense(256,activation="relu")(l1)
    l1_ = KL.Dense(144,activation="relu")(encoded)
    reshape = KL.Reshape((3,3,16))(l1_)
    convT1 = KL.Conv2DTranspose(filters=16,kernel_size=3,strides=3,activation="elu")(reshape)
    convT2 = KL.Conv2DTranspose(filters=8,kernel_size=2,strides=2,activation="elu")(convT1)
    convT3 = KL.Conv2DTranspose(filters=2,kernel_size=2,strides=1,activation="elu")(convT2)

    network = tf.keras.models.Model(input_img,convT3)
    return network

def FFNetwork3(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
    input_img = KL.Input(shape=(19,19,2))
    conv1 = KL.Conv2D(filters=16,kernel_size=3,strides=1,activation="elu")(input_img)
    conv2 = KL.Conv2D(filters=32,kernel_size=3,strides=1,activation="elu")(conv1)
    conv3 = KL.Conv2D(filters=64,kernel_size=3,strides=1,activation="elu")(conv2)
    conv4 = KL.Conv2D(filters=64,kernel_size=3,strides=1,activation="elu")(conv3)
    flat = KL.Flatten()(conv4)
    l1 = KL.Dense(200,activation="relu")(flat)
    encoded = KL.Dense(256,activation="relu")(l1)
    l1_ = KL.Dense(968,activation="relu")(encoded)
    reshape = KL.Reshape((11,11,8))(l1_)
    convT1 = KL.Conv2DTranspose(filters=16,kernel_size=3,strides=1,activation="elu")(reshape)
    convT2 = KL.Conv2DTranspose(filters=16,kernel_size=3,strides=1,activation="elu")(convT1)
    convT3 = KL.Conv2DTranspose(filters=16,kernel_size=3,strides=1,activation="elu")(convT2)
    convT4 = KL.Conv2DTranspose(filters=2,kernel_size=3,strides=1,activation=None)(convT3)

    network = tf.keras.models.Model(input_img,convT4)
    return network

def FFNetworkInception(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
    input_img = KL.Input(shape=(19,19,2))
    conv1 = Inception(filters=[4,4,2,2])(input_img)
    conv2 = Inception(filters=[8,8,4,4])(conv1)
    conv3 = Inception(filters=[16,16,8,8])(conv2)
    conv4 = Inception(filters=[32,32,16,16])(conv3)
    flat = KL.Flatten()(conv4)
    l1 = KL.Dense(200,activation="relu")(flat)
    encoded = KL.Dense(256,activation="relu")(l1)
    l1_ = KL.Dense(968,activation="relu")(encoded)
    reshape = KL.Reshape((11,11,8))(l1_)
    convT1 = ReverseInception(filters=[32,32,16,16])(reshape)
    convT2 = ReverseInception(filters=[16,16,8,8])(convT1)
    convT3 = ReverseInception(filters=[8,8,4,4])(convT2)
    convT4 = KL.Conv2DTranspose(filters=2,kernel_size=3,strides=1,activation=None)(convT3)

    network = tf.keras.models.Model(input_img,convT4)
    return network

def SFNetwork(self, configFile, actionSize, netConfigOverride={}, scope=None,debug=True, training=True):
    input_img = KL.Input(shape=(19,19,2))
    conv1 = KL.Conv2D(filters=16,kernel_size=2,strides=1,activation="elu")(input_img)
    conv2 = KL.Conv2D(filters=32,kernel_size=2,strides=1,activation="elu")(conv1)
    flat = KL.Flatten()(conv2)
    l1 = KL.Dense(200,activation="relu")(flat)
    encoded = KL.Dense(256,activation="relu")(l1)
    l1_ = KL.Dense(144,activation="relu")(encoded)
    reshape = KL.Reshape((3,3,16))(l1_)
    convT1 = KL.Conv2DTranspose(filters=16,kernel_size=3,strides=3,activation="elu")(reshape)
    convT2 = KL.Conv2DTranspose(filters=8,kernel_size=2,strides=2,activation="elu")(convT1)
    convT3 = KL.Conv2DTranspose(filters=2,kernel_size=2,strides=1,activation=None)(convT2)

    reward_pred = KL.Dense(1,activation=None)(encoded)

    network = tf.keras.models.Model(input_img,[convT3,reward_pred])
    return network



if __name__ == "__main__":
    # sess = tf.Session()
    # test = Network(configFile="test.json",actionSize=4)
    pass
