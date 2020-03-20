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
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)

for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    env,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings,sess)

#Creating the Networks and Methods of the Run.

with tf.device('/gpu:0'):
    netConfigOverride["DefaultParams"]["Trainable"] = False
    SF,SF2,SF3,SF4 = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    try:SF.load_weights(MODEL_PATH+"/model_phi.h5")
    except: print("Did not load weights")

def GetAction(state,episode=0,step=0,deterministic=False,debug=False):
    """
    Contains the code to run the network based on an input.
    """
    p = 1/nActions
    if len(state.shape)==3:
        probs =np.full((1,nActions),p)
    else:
        probs =np.full((state.shape[0],nActions),p)
    actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
    if debug: print(probs)
    return actions , []  # return a int and extra data that needs to be fed to buffer.

s = []
s_next = []
r_store = []
for i in range(settings["EnvHPs"]["SampleEpisodes"]):
    for functionString in envSettings["BootstrapFunctions"]:
        env.seed(34)
        BootstrapFunctions = GetFunction(functionString)
        s0, loggingDict = BootstrapFunctions(env,settings,envSettings,sess)

    for functionString in envSettings["StateProcessingFunctions"]:
        StateProcessing = GetFunction(functionString)
        s0 = StateProcessing(s0,env,envSettings,sess)

    for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):

        a, networkData = GetAction(state=s0,episode=0,step=j)

        for functionString in envSettings["ActionProcessingFunctions"]:
            ActionProcessing = GetFunction(functionString)
            a = ActionProcessing(a,env,envSettings,sess)

        s1,r,done,_ = env.step(a)
        for functionString in envSettings["StateProcessingFunctions"]:
            StateProcessing = GetFunction(functionString)
            s1 = StateProcessing(s1,env,envSettings,sess)

        for functionString in envSettings["RewardProcessingFunctions"]:
            RewardProcessing = GetFunction(functionString)
            r,done = RewardProcessing(s1,r,done,env,envSettings,sess)
        s.append(s0)
        s_next.append(s1)
        r_store.append(r)
        s0 = s1

        if done.all():
            break

LOG_PATH = './images/SF/'+EXP_NAME
CreatePath(LOG_PATH)

class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%250 == 0:
            model_json = SF2.to_json()
            with open(MODEL_PATH+"model_psi.json", "w") as json_file:
                json_file.write(model_json)
            SF2.save_weights(MODEL_PATH+"model_psi.h5")

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

counter = 0
class ValueTest(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch != 0 and epoch%29 == 0:
            global counter
            env.seed(1337)
            env.reset()
            rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                [value] = SF4.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(env.grid.encode()[:,:,0], vmin=0, vmax=10)
            fig.add_subplot(2,1,2)
            plt.title("Value Prediction")
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(LOG_PATH+"/ValuePred"+str(counter)+".png")
            plt.close()
            counter +=1



SF2.compile(optimizer="adam", loss="mse")
phi = SF3.predict(np.stack(s))
gamma=0.99
for i in range(50):

    psi_next = SF2.predict(np.stack(s_next))

    labels = phi+gamma*psi_next
    SF2.fit(
        np.stack(s),
        [np.stack(labels)],
        epochs=30,
        batch_size=512,
        shuffle=True,
        callbacks=[ValueTest(),SaveModel()])
