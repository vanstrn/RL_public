"""
Methods to create a Keras Functional Model.

"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import json
from .layers.non_local import Non_local_nn
from .layers.approx_round import *
from .layers.inception import Inception
from .layers.reverse_inception import ReverseInception
import tensorflow.keras.backend as K

import collections.abc
import os

def UpdateStringValues(d, u):
    """
    Updates values of a nested dictionary/list structure. The method searches
    for a keys provided by the override dictionary and replaces them with
    associated values.
    """
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = UpdateStringValues(d.get(k, {}), u)
        elif isinstance(v,list):
            list_new = []
            for val in v:
                if isinstance(val, collections.abc.Mapping):
                    tmp_dict = UpdateStringValues(val, u)
                    list_new.append(tmp_dict)
                elif isinstance(val, list):
                    pass
                elif val in u.keys():
                    list_new.append(u[val])
                else:
                    list_new.append(val)
            d[k] = list_new
        else:
            for key in u.keys():
                if isinstance(v, str):
                    if key in v:
                        d[k] = EvalString(v,u)
                        break
    return d

def EvalString(string,updateDict):
    for key,value in updateDict.items():
        string = string.replace(key,str(value))
    return eval(string)

def UpdateNestedDictionary(defaultSettings,overrides):
    for label,override in overrides.items():
        if isinstance(override, collections.abc.Mapping):
            UpdateNestedDictionary(defaultSettings[label],override)
        else:
            defaultSettings[label] = override
    return defaultSettings

def GetLayer( dict):
    """Based on a dictionary input the function returns the appropriate layer for the NN."""

    if dict["layerType"] == "Dense":
        layer = KL.Dense( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Conv2D":
        layer = KL.Conv2D( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Conv2DTranspose":
        layer = KL.Conv2DTranspose( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "SeparableConv":
        layer = KL.SeparableConv2D( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Round":
        layer= RoundingSine(name=dict["layerName"])
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
    elif dict["layerType"] == "Multiply":
        layer = KL.Multiply( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Add":
        layer = KL.Add( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Reshape":
        layer = KL.Reshape( **dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "LSTM":
        layer = KL.LSTM(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "SimpleRNN":
        layer = KL.SimpleRNN(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "Sum":
        layer = tf.keras.backend.sum
    elif dict["layerType"] == "Inception":
        layer = Inception(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "ReverseInception":
        layer = ReverseInception(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "UpSampling2D":
        layer = KL.UpSampling2D(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "GaussianNoise":
        layer = KL.GaussianNoise(**dict["Parameters"],name=dict["layerName"])
    elif dict["layerType"] == "StopGradient":
        layer = KL.Lambda(lambda x: K.stop_gradient(x))

    return layer


if __name__ == "__main__":
    # test = Network(configFile="test.json",actionSize=4)
    test = {"K1":1,
            "K2":2,
            "K3":"V1",
            "K4":2,
            "K5":{"test":["V1","V2"]},
            "K6":"V2",
            "K7":2,
            }
    test2 = {"V1":4,"V2":5}
    dict=UpdateStringValues(test,test2)
    # ReplaceValues(test,test2)
    print(dict)
