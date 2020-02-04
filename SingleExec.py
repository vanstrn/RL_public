"""
Modular Framework for setting up singular environment sampling
based on a runtime config file.
"""

import numpy as np
import gym
import gym_minigrid
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    env,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings,sess)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

with tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride)
    Method = GetFunction(settings["Method"])
    net = Method(network,sess,stateShape=dFeatures,actionSize=nActions,HPs=settings["NetworkHPs"],nTrajs=nTrajs)

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,MODEL_PATH)
InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
#Running the Simulation
for i in range(settings["EnvHPs"]["MAX_EP"]):

    sess.run(global_step_next)
    logging = interval_flag(sess.run(global_step), settings["EnvHPs"]["LOG_FREQ"], 'log')
    saving = interval_flag(sess.run(global_step), settings["EnvHPs"]["SAVE_FREQ"], 'save')

    for functionString in envSettings["BootstrapFunctions"]:
        BootstrapFunctions = GetFunction(functionString)
        s0, loggingDict = BootstrapFunctions(env,settings,envSettings,sess)
    for functionString in envSettings["StateProcessingFunctions"]:
        StateProcessing = GetFunction(functionString)
        s0 = StateProcessing(s0,env,envSettings,sess)

    for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):
        updating = interval_flag(j, settings["EnvHPs"]['UPDATE_GLOBAL_ITER'], 'update')


        a, networkData = net.GetAction(state=s0,global_step=global_step,step=j)

        for functionString in envSettings["ActionProcessingFunctions"]:
            ActionProcessing = GetFunction(functionString)
            a = ActionProcessing(a,env,envSettings,sess)
        # env.render()

        s1,r,done,_ = env.step(a)
        for functionString in envSettings["StateProcessingFunctions"]:
            StateProcessing = GetFunction(functionString)
            s1 = StateProcessing(s1,env,envSettings,sess)

        for functionString in envSettings["RewardProcessingFunctions"]:
            RewardProcessing = GetFunction(functionString)
            r,done = RewardProcessing(s1,r,done,env,envSettings,sess)

        #Update Step
        net.AddToTrajectory([s0,a,r,s1,done]+networkData)

        if updating:   # update global and assign to local net
            net.Update(settings["NetworkHPs"],logging,writer)

        for functionString in envSettings["LoggingFunctions"]:
            LoggingFunctions = GetFunction(functionString)
            loggingDict = LoggingFunctions(loggingDict,s1,r,done,env,envSettings,sess)

        s0 = s1
        if done.all() or j == settings["EnvHPs"]["MAX_EP_STEPS"]:
            net.Update(settings["NetworkHPs"],logging,writer)
            net.ClearTrajectory()
        if done.all():
            break
    #Closing Functions that will be executed after every episode.
    for functionString in envSettings["EpisodeClosingFunctions"]:
        EpisodeClosingFunction = GetFunction(functionString)
        finalDict = EpisodeClosingFunction(loggingDict,env,settings,envSettings,sess,progbar)

    if logging:
        saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))

    if saving:
        Record(finalDict, writer, i)
