"""
Preliminary Network Architecture for a DNN that can be used in as is or in more
complicated networks.

Dependencies: Tensorflow

To Do: Make the input to the DNN a list with a variable length.
"""

import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam

def CreateDNN(inputShape,outputShape,optimizer=None,layerSizes=[100,100,100]):
    """
    Creates a DNN, with a specified optimizer.
    """
    model   = Sequential()
    for i,size in enumerate(layerSizes):
        if i==0:
            model.add(Dense(24, input_dim=inputShape[0], activation="relu"))
        else:
            model.add(Dense(size, activation="relu"))
    model.add(Dense(outputShape))


    if optimizer == None:
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=0.005))
    else:
        model.compile(loss="mse", optimizer=optimizer)

    return model



def Create2InDNN(inputShape1,inputShape2,optimizer,layerSizes1=[100,100,100],
    layerSizes2=[100],layerSizes3=[100]):
    """
    Creates a DNN, with a specified optimizer.
    """
    input1 = Input(shape=inputShape1)
    for i,size in enumerate(layerSizes1):
        if i==0:
            h1 = Dense(size, activation='relu')(input1)
        else:
            h1 = Dense(size, activation='relu')(h1)

    input2 = Input(shape=inputShape2)
    action_h1    = Dense(48)(input2)
    for i,size in enumerate(layerSizes2):
        if i==0:
            h2 = Dense(size, activation='relu')(input2)
        else:
            h2 = Dense(size, activation='relu')(h2)

    merged    = Add()([h1, h2])

    for i,size in enumerate(layerSizes2):
        if i==0:
            h3 = Dense(size, activation='relu')(merged)
        else:
            h3 = Dense(size, activation='relu')(h3)

    output = Dense(1, activation='relu')(h3)
    model  = Model(input=[input1,input2], output=output)

    adam  = Adam(lr=0.001)
    model.compile(loss="mse", optimizer=adam)
    return model


if __name__ == "__main__":
    opt  = Adam(lr=0.001)
    opt2  = Adam(lr=0.001)
    dnn = CreateDNN((20,),(2,),opt)
    dnn = Create2InDNN((20,),(2,),opt2)
