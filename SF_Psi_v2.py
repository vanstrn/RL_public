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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device('/gpu:0'):
    try:
        netConfigOverride["DefaultParams"]["Trainable"] = False
    except:
        netConfigOverride["DefaultParams"] = {}
        netConfigOverride["DefaultParams"]["Trainable"] = False
    SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    try:SF5.load_weights(MODEL_PATH+"/model.h5")
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



class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%250 == 0:
            model_json = SF2.to_json()
            with open(MODEL_PATH+"model_psi.json", "w") as json_file:
                json_file.write(model_json)
            SF2.save_weights(MODEL_PATH+"model_psi.h5")
            SF5.save_weights(MODEL_PATH+"model.h5")

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

counter = 0
class ValueTest(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if (epoch+1)%settings["FitIterations"] == 0:
            global counter
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


opt = tf.keras.optimizers.Adam(learning_rate=settings["LearningRate"])
SF2.compile(optimizer=opt, loss="mse")
phi = SF3.predict(np.stack(s))
gamma=settings["Gamma"]
for i in range(settings["Epochs"]):

    psi_next = SF2.predict(np.stack(s_next))

    labels = phi+gamma*psi_next
    SF2.fit(
        np.stack(s),
        [np.stack(labels)],
        epochs=settings["FitIterations"],
        batch_size=settings["BatchSize"],
        shuffle=True,
        callbacks=[ValueTest(),SaveModel()])
