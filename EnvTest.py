"""
Asynchonous Advantage Actor Critic implementation in Cart-Pole.

"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym,gym_cap
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from AC import ActorCritic_CNN
from Utils import record,Memory,one_hot_encoder

# from policy.random import Random
# from policy.roomba import Roomba
from policy import Patrol
# from policy.roomba import Roomba

tf.enable_eager_execution()


visionRange = 5


if __name__ == "__main__":

    env = gym.make("cap-v0")
    A=1
    obs = env.reset(policy_red=Patrol())
