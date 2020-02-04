"""
Modular Framework for setting up singular environment sampling
based on a runtime config file.
"""

import numpy as np
import gym
import gym_minigrid,gym_cap
import tensorflow as tf
import argparse
from urllib.parse import unquote

from networks.network import Network
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import json

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

#Creating the Environment and Network to be used in training
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=250)])
for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    env,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME+'/'
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

with tf.device('/gpu:0'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride)
    Method = GetFunction(settings["Method"])
    net = Method(network,stateShape=dFeatures,actionSize=nActions,HPs=settings["NetworkHPs"],nTrajs=nTrajs)

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.create_file_writer(LOG_PATH)
net.InitializeVariablesFromFile(MODEL_PATH)

progbar = tf.keras.utils.Progbar(None, unit_name='Training')
#Running the Simulation
for i in range(settings["EnvHPs"]["MAX_EP"]):

    global_step+= 1

    logging = interval_flag(int(global_step), settings["EnvHPs"]["LOG_FREQ"], 'log')
    saving = interval_flag(int(global_step), settings["EnvHPs"]["SAVE_FREQ"], 'save')

    for functionString in envSettings["BootstrapFunctions"]:
        BootstrapFunctions = GetFunction(functionString)
        s0, loggingDict = BootstrapFunctions(env,settings,envSettings)
    for functionString in envSettings["StateProcessingFunctions"]:
        StateProcessing = GetFunction(functionString)
        s0 = StateProcessing(s0,env,envSettings)

    for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):
        updating = interval_flag(j, settings["EnvHPs"]['UPDATE_GLOBAL_ITER'], 'update')

        a, networkData = net.GetAction(state=s0)

        for functionString in envSettings["ActionProcessingFunctions"]:
            ActionProcessing = GetFunction(functionString)
            a = ActionProcessing(a,env,envSettings)

        s1,r,done,_ = env.step(a)
        for functionString in envSettings["StateProcessingFunctions"]:
            StateProcessing = GetFunction(functionString)
            s1 = StateProcessing(s1,env,envSettings)

        for functionString in envSettings["RewardProcessingFunctions"]:
            RewardProcessing = GetFunction(functionString)
            r,done = RewardProcessing(s1,r,done,env,envSettings)

        #Update Step
        net.AddToTrajectory([s0,a,r,s1,done]+networkData)


        for functionString in envSettings["LoggingFunctions"]:
            LoggingFunctions = GetFunction(functionString)
            loggingDict = LoggingFunctions(loggingDict,s1,r,done,env,envSettings)

        s0 = s1

        if updating or done.all():   # update global and assign to local net
            net.Update(settings["NetworkHPs"])
        if done.all() or j == settings["EnvHPs"]["MAX_EP_STEPS"]:
            net.Update(settings["NetworkHPs"])
            net.ClearTrajectory()
        if done.all():
            break

    #Closing Functions that will be executed after every episode.
    for functionString in envSettings["EpisodeClosingFunctions"]:
        EpisodeClosingFunction = GetFunction(functionString)
        finalDict = EpisodeClosingFunction(loggingDict,env,settings,envSettings)

    progbar.update(int(global_step))

    if saving:
        net.SaveModel(MODEL_PATH,global_step)

    if logging:
        Record(finalDict, writer, int(global_step))
