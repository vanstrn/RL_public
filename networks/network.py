"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""

import numpy as np
import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self):
        """
        """
        super(Network,self).__init__()
