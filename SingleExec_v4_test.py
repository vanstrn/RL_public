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
import os
import json
import time
import psutil

from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction, UpdateNestedDictionary
from utils.record import Record,SaveHyperparams
from environments.Common import CreateEnvironment

#Input arguments to override the default Config Files
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True,
                    help="File for specific run. Located in ./configs/run")
parser.add_argument("-c", "--config", required=False,
                    help="JSON configuration string to override runtime configs of the script.")
parser.add_argument("-n", "--network", required=False,
                    help="JSON configuration string to override network parameters")
parser.add_argument("-p", "--processor", required=False, default="/gpu:0",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")
parser.add_argument("-r", "--render", required=False, default=False, action="store_true",
                    help="Whether or not to render the environment")
parser.add_argument("-t", "--cont", required=False, default=False, action="store_true",
                    help="Defining if this a continuation of training")
args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
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
    settings = UpdateNestedDictionary(settings,configOverride)

for (dirpath, dirnames, filenames) in os.walk("configs/environment"):
    for filename in filenames:
        if settings["EnvConfig"] == filename:
            envConfigFile = os.path.join(dirpath,filename)
            break
with open(envConfigFile) as json_file:
    envSettings = json.load(json_file)

print(settings["NetworkHPs"])
env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)
EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
IMAGE_PATH = './images/'+EXP_NAME

if "LoadName" in settings:
    LOAD_PATH = './models/'+settings["LoadName"]
else:
    LOAD_PATH = MODEL_PATH

CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)
# CreatePath(IMAGE_PATH)

#Saving config files in the model directory
with open(LOG_PATH+'/runSettings.json', 'w') as outfile:
    json.dump(settings, outfile)
with open(MODEL_PATH+'/netConfigOverride.json', 'w') as outfile:
    json.dump(netConfigOverride, outfile)

#Creating the Environment and Network to be used in training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device(args.processor):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    Method = GetFunction(settings["Method"])
    net = Method(sess,settings,netConfigOverride,stateShape=dFeatures,actionSize=nActions,nTrajs=nTrajs,env=env)

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,LOAD_PATH)

#If not continuing training reset step counter to 0.
if args.cont:
    global_step_next = tf.assign(global_step,0)

InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])

if "LoggingFunctions" in settings:
    loggingFunctions=[]
    for loggingFunc in settings["LoggingFunctions"]:
        func = GetFunction(loggingFunc)
        loggingFunctions.append(func(env,net,IMAGE_PATH))

for i in range(settings["MaxEpisodes"]):

    sess.run(global_step_next)
    logging = interval_flag(sess.run(global_step), settings["LogFreq"], 'log')
    saving = interval_flag(sess.run(global_step), settings["SaveFreq"], 'save')

    s0 = env.reset()

    for j in range(settings["MaxEpisodeSteps"]+1):

        a, networkData = net.GetAction(state=s0,episode=sess.run(global_step),step=j)

        s1,r,done,info = env.step(action=a)
        net.AddToTrajectory([s0,a,r,s1,done]+networkData)
        if args.render:
            env.render()
        s0 = s1
        if done or j == settings["MaxEpisodeSteps"]:
            net.Update(sess.run(global_step))
            break

    loggingDict = env.getLogging()
    if logging:
        dict = net.GetStatistics()
        loggingDict.update(dict)
        loggingDict["Memory Use"] = psutil.virtual_memory().used*1E-9
        loggingDict["Memory Percent"] = psutil.virtual_memory().percent
        loggingDict["CPU Use"] = psutil.cpu_percent()
        Record(loggingDict, writer, sess.run(global_step))
        if "LoggingFunctions" in settings:
            for func in loggingFunctions:
                func(sess.run(global_step))

    if saving:
        saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))
    progbar.update(i)
