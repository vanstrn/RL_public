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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device('/gpu:0'):
    netConfigOverride["DefaultParams"]["Trainable"] = False
    SF1,SF2,SF3,SF4 = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    try:SF1.load_weights(MODEL_PATH+"/model_phi.h5")
    except: print("Did not load weights")
    try:SF2.load_weights(MODEL_PATH+"/model_psi.h5")
    except: print("Did not load weights")

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
    return actions   # return a int and extra data that needs to be fed to buffer.
def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5: #Wall
    return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

if True:
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

#####Evaluating using random sampling ########
#Processing state samples into Psi.
    psiSamples = SF2.predict(np.stack(s)) # [X,SF Dim]

    ##-Repeat M times to evaluate the effect of sampling.
    M = 3
    count = 0
    dim = psiSamples.shape[1]
    for replicate in range(M)
        #Taking Eigenvalues and Eigenvectors of the environment,
        w_g,v_g = np.linalg.eig(psiSamples[count:count+dim,:])
        count += dim

        #Creating Eigenpurposes of the N highest Eigenvectors and saving images
        N = 5
        for sample in range(N)
            v_option=np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                phi= SF1.predict(np.expand_dims(grid,0))
                v_option[i,j]= phi*v_g(:,sample)
            imgplot = plt.imshow(v_option)
            plt.title("Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample]))
            plt.savefig(LOG_PATH+"/option"+str(sample)+"replicate"+str(replicate)+".png")


if True: #Calculating the MSE in the state,reward and value prediction.

    #Reward Prediction Error. Performed over the reward map of entire environment.
    env.reset()
    rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
    for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        grid = ConstructSample(env,[i,j])
        if grid is None: continue
        [_,reward] = SF1.predict(np.expand_dims(grid,0))
        rewardMap[i,j] = reward
    rewardMapReal = np.zeros((dFeatures[0],dFeatures[1]))
    rewardMapReal[8,14] = 0.25
    rewardMapReal[10,14] = 0.25

    rewardError = np.sum((rewardMapReal-rewardMap)**2)
    print(rewardError)

    #Value Prediction Error. Performed over the value map of entire environment.
    from DecompositionVisual import ObstacleVisualization
    valueMap = np.zeros((dFeatures[0],dFeatures[1]))
    for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        grid = ConstructSample(env,[i,j])
        if grid is None: continue
        [value] = SF4.predict(np.expand_dims(grid,0))
        valueMap[i,j] = value
    valueMapReal = ObstacleVisualization()

    valueError = np.sum((valueMapReal-valueMap)**2)
    print(valueError)
