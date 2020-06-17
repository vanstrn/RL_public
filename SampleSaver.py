
"""
Code to get a static training set from an environment with random actions.
To Do:  -Implement an On-Policy Option.
        -Help, which will show required JSON entries
"""

import numpy as np
import tensorflow as tf
import argparse
from urllib.parse import unquote
import os

from utils.utils import CreatePath
import json
import environments as e
import gym,gym_cap


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
for (dirpath, dirnames, filenames) in os.walk("configs/run"):
    for filename in filenames:
        if args.file == filename:
            runConfigFile = os.path.join(dirpath,filename)
            break
with open(runConfigFile) as json_file:
    settings = json.load(json_file)
    settings.update(configOverride)

for (dirpath, dirnames, filenames) in os.walk("configs/environment"):
    for filename in filenames:
        if settings["EnvConfig"] == filename:
            envConfigFile = os.path.join(dirpath,filename)
            break
with open(envConfigFile) as json_file:
    envSettings = json.load(json_file)
    envSettings.update(envConfigOverride)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME+ '/'
LOG_PATH = './images/SF/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment
env,dFeatures,nActions,nTrajs = e.CreateEnvironment(envSettings)

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

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

s = []
s_next = []
r_store = []
for i in range(200):
    s0 = env.reset()

    for j in range(settings["MaxEpisodeSteps"]+1):

        a = GetAction(state=s0)

        s1,r,done,_ = env.step(a)

        if arreq_in_list(s0,s):
            pass
        else:
            s.append(s0)
            s_next.append(s1)
            r_store.append(r)

        s0 = s1
        if done:
            break

np.savez_compressed("./data/SF_SampleData10Map_2.npz", s=s, s_next=s_next, r_store=r_store)
