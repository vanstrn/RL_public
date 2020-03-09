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





if __name__ == "__main__":
    # sess = tf.Session()
    # test = Network(configFile="test.json",actionSize=4)
    pass
