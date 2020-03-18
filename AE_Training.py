import tensorflow as tf
import numpy as np
import gym, gym_minigrid, gym_cap

from utils.RL_Wrapper import TrainedNetwork
from utils.utils import InitializeVariables

# net = TrainedNetwork("models/MG_A3C_SF_Testing/",
#     input_tensor="S:0",
#     output_tensor="Global/activation/Softmax:0",
#     device='/cpu:0'
#     )
#
# # session = tf.keras.backend.get_session()
# # init = tf.global_variables_initializer()
# # session.run(init)
#
# InitializeVariables(net.sess)
# x=np.random.random([1,7,7,3])
#
# out = net.get_action(x)
# print(out)


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
from networks.network_v2 import buildNetwork
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
# sess = tf.Session()

for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    env,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings,sess)

#Creating the Networks and Methods of the Run.
def M4E(y_true,y_pred):
    return K.mean(K.pow(y_pred-y_true,4))

with tf.device('/gpu:0'):
    AE = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    AE.summary()
    AE.compile(optimizer="adadelta", loss=M4E)
    try:AE.load_weights(MODEL_PATH+"/model.h5")
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
# for i in range(15):
for i in range(settings["EnvHPs"]["SampleEpisodes"]):
    for functionString in envSettings["BootstrapFunctions"]:
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
        s0 = s1

        if done.all():
            break
LOG_PATH = './images/AE/'+EXP_NAME
CreatePath(LOG_PATH)
class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%250 == 0:
            model_json = AE.to_json()
            with open(MODEL_PATH+"model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            AE.save_weights(MODEL_PATH+"model.h5")
            print("Saved model to disk")

class ImageGenerator(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%250 == 0:
            for i in range(15):
                if len(s[0].shape) == 4:
                    state = s[i*300+randint(0,200)][0,:,:,:]
                else:
                    state = s[i*300+randint(0,200)]
                state_new = AE.predict(np.expand_dims(state,0))
                fig=plt.figure(figsize=(5.5, 8))
                fig.add_subplot(2,1,1)
                plt.title("State")
                imgplot = plt.imshow(state[:,:,0], vmin=0, vmax=10)
                fig.add_subplot(2,1,2)
                plt.title("Predicted Next State")
                imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
                plt.savefig(LOG_PATH+"/test"+str(i)+".png")
                plt.close()
class ImageGenerator2(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%250 == 0:
            for i in range(20):
                if len(s[0].shape) == 4:
                    state = s[i*300+randint(0,200)][0,:,:,:]
                else:
                    state = s[i*300+randint(0,200)]
                state_new = AE.predict(np.expand_dims(state,0))
                fig=plt.figure(figsize=(16, 8))
                fig.add_subplot(2,4,1)
                plt.title("State (Background)")
                imgplot = plt.imshow(state[:,:,1], vmin=-2, vmax=2)
                fig.add_subplot(2,4,5)
                plt.title("Predicted Next State (Background)")
                imgplot = plt.imshow(state_new[0,:,:,1],vmin=-2, vmax=2)
                fig.add_subplot(2,4,2)
                plt.title("State (Flag)")
                imgplot = plt.imshow(state[:,:,2], vmin=-2, vmax=2)
                fig.add_subplot(2,4,6)
                plt.title("Predicted Next State (Flag)")
                imgplot = plt.imshow(state_new[0,:,:,2],vmin=-2, vmax=2)
                fig.add_subplot(2,4,3)
                plt.title("State (Obstacles)")
                imgplot = plt.imshow(state[:,:,3], vmin=-2, vmax=2)
                fig.add_subplot(2,4,7)
                plt.title("Predicted Next State (Obstacles)")
                imgplot = plt.imshow(state_new[0,:,:,3],vmin=-2, vmax=2)
                fig.add_subplot(2,4,4)
                plt.title("State (Ground Robots)")
                imgplot = plt.imshow(state[:,:,4], vmin=-2, vmax=2)
                fig.add_subplot(2,4,8)
                plt.title("Predicted Next State (Ground Robots)")
                imgplot = plt.imshow(state_new[0,:,:,4],vmin=-2, vmax=2)
                plt.savefig(LOG_PATH+"/test"+str(i)+".png")
                plt.close()
if len(s[0].shape) == 4:
    state = np.vstack(s)
    state_ = np.vstack(s_next)
else:
    state = np.stack(s)
    state_ = np.stack(s_next)

del env
AE.fit(
    {"state":state},
    {"StatePrediction":state_},
    epochs=5000,
    batch_size=512,
    shuffle=True,
    callbacks=[ImageGenerator2(),SaveModel()])
