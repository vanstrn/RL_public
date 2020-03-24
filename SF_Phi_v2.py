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

def M4E(y_true,y_pred):
    return K.mean(K.pow(y_pred-y_true,4))

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device('/gpu:0'):
    SF,_,_,_ = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    try:SF.load_weights(MODEL_PATH+"/model.h5")
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
    return actions

#Collecting samples
s = []
s_next = []
r_store = []
for i in range(settings["SampleEpisodes"]):
    s0 = env.reset()

    for j in range(settings["MAX_EP_STEPS"]+1):

        a = GetAction(state=s0,episode=0,step=j)

        s1,r,done,_ = env.step(a)

        s.append(s0)
        s_next.append(s1)
        r_store.append(r)

        s0 = s1
        if done:
            break


class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%50 == 0:
            model_json = SF.to_json()
            with open(MODEL_PATH+"model_phi.json", "w") as json_file:
                json_file.write(model_json)
            SF.save_weights(MODEL_PATH+"model_phi_"+str(epoch)+".h5")

class ImageGenerator(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%50 == 0:
            for i in range(5):
                state = s[i*100+randint(0,200)]
                [state_new,reward] = SF.predict(np.expand_dims(state,0))
                fig=plt.figure(figsize=(5.5, 8))
                fig.add_subplot(2,1,1)
                plt.title("State")
                imgplot = plt.imshow(state[:,:,0], vmin=0, vmax=10)
                fig.add_subplot(2,1,2)
                plt.title("Predicted Next State - Reward: " + str(reward))
                imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
                if i == 0:
                    plt.savefig(LOG_PATH+"/StatePredEpoch"+str(epoch)+".png")
                else:
                    plt.savefig(LOG_PATH+"/StatePred"+str(i)+".png")
                plt.close()

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

class RewardTest(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%50 == 0:
            for num in range(1):
                env.reset()
                rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
                for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                    grid = ConstructSample(env,[i,j])
                    if grid is None: continue
                    [state_new,reward] = SF.predict(np.expand_dims(grid,0))
                    rewardMap[i,j] = reward
                fig=plt.figure(figsize=(5.5, 8))
                fig.add_subplot(2,1,1)
                plt.title("State")
                imgplot = plt.imshow(env.grid.encode()[:,:,0], vmin=0, vmax=10)
                fig.add_subplot(2,1,2)
                plt.title("Reward Prediction Epoch "+str(epoch))
                imgplot = plt.imshow(rewardMap)
                fig.colorbar(imgplot)
                plt.savefig(LOG_PATH+"/RewardPred"+str(epoch)+".png")
                plt.close()

opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
SF.compile(optimizer=opt, loss=[M4E,"mse"], loss_weights = [1.0,1.0])
SF.fit(
    np.stack(s),
    [np.stack(s_next),np.stack(r_store)],
    epochs=1001,
<<<<<<< HEAD:SF_Phi.py
    batch_size=settings["BatchSize"],
=======
    batch_size=2048,
>>>>>>> f12bfbeb21c2dd457b102412a5cdafd44fac2695:SF_Phi_v2.py
    shuffle=True,
    callbacks=[ImageGenerator(),SaveModel(),RewardTest()])
