import gym
import gym_cap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# x = np.matrix([
#     [1,2,3,4,5,6,7,8],
#     [1,2,3,4,5,6,7,8],
#     [2,3,2,3,2,3,2,3],
#     [2,3,2,3,2,3,2,3],
#     [2,3,2,3,2,3,2,3],
#     [2,3,2,3,2,3,2,3],
#     [2,3,2,3,2,3,2,3],
#     [2,3,2,3,2,3,2,3],
#     ])
# noise = np.random.rand(8,8)
#
# res = x +noise*0
#
# w,v = np.linalg.eig(res)
# print(v)




import tensorflow as tf
import numpy as np
import gym, gym_minigrid, gym_cap

from utils.RL_Wrapper import TrainedNetwork
from utils.utils import InitializeVariables

"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import gym_minigrid,gym_cap
import tensorflow as tf
import argparse
from urllib.parse import unquote

from networks.networkAE import *
from networks.network_v3 import buildNetwork
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import json
from utils.worker import Worker as Worker
from utils.utils import MovingAverage
import threading
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.backend as K
from random import randint
from environments.Common import CreateEnvironment


#Input arguments to override the default Config Files
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True,
                    help="File for specific run. Located in ./configs/run")
parser.add_argument("-c", "--config", required=False,
                    help="JSON configuration string to override runtime configs of the script.")
parser.add_argument("-e", "--environment", required=False,
                    help="JSON configuration string to override environment parameters")
parser.add_argument("-n", "--network", required=False,
                    help="JSON configuration string to override network parameters")
args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
if args.environment is not None: envConfigOverride = json.loads(unquote(args.environment))
else: envConfigOverride = {}
if args.network is not None: netConfigOverride = json.loads(unquote(args.network))
else: netConfigOverride = {}

#Defining parameters and Hyperparameters for the run.
with open("configs/run/"+args.file) as json_file:
    settings = json.load(json_file)
    settings.update(configOverride)
with open("configs/environment/"+settings["EnvConfig"]) as json_file:
    envSettings = json.load(json_file)
    envSettings.update(envConfigOverride)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME+ '/'
LOG_PATH = './images/SF/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment
env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)

#Creating the Networks and Methods of the Run.

def GetAction(state):
    """
    Contains the code to run the network based on an input.
    """
    p = 1/nActions
    if len(state.shape)==3:
        probs =np.full((1,nActions),p)
    else:
        probs =np.full((state.shape[0],nActions),p)
    actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
    return actions

s = []
s_next = []
r_store = []
for i in range(settings["SampleEpisodes"]):
    s0 = env.reset()

    for j in range(settings["MAX_EP_STEPS"]+1):

        a = GetAction(state=s0)

        s1,r,done,_ = env.step(a)

        s.append(s0)
        s_next.append(s1)
        r_store.append(r)

        s0 = s1
        if done:
            break

np.savez_compressed("./data/SF_TRIAL2.npz", s=s, s_next=s_next, r_store=r_store)
