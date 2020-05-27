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
import json
import time

from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
from environments.Common import CreateEnvironment

#######Change here to use different method or network.####################
from networks.ppoNetwork import PPONetwork as Network
from methods.PPO import PPO as TrainingMethod
#This is hardcoded If you want modular version look at code in master branch.
##########################################################################


#Input arguments to create the training run.
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True,
                    help="File for specific run. Located in ./configs/run")
parser.add_argument("-c", "--config", required=False,
                    help="JSON configuration string to override runtime configs of the script.")
parser.add_argument("-p", "--processor", required=False, default="/gpu:0",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")

args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}

#Defining parameters and Hyperparameters for the run.
with open("configs/run/"+args.file) as json_file:
    settings = json.load(json_file)
    settings.update(configOverride)

#Creating directories for logging and saving models.
EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment based on wrapper structure.
with open("configs/environment/"+settings["EnvConfig"]) as json_file:
    envSettings = json.load(json_file)
env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)
#Initializing the Hardware for the run, based on the input arguments and config file.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)

#Setting up the Network.
with tf.device(args.processor):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network(nActions,scope="Training")
    net = TrainingMethod(network,sess,stateShape=dFeatures,actionSize=nActions,HPs=settings["NetworkHPs"],nTrajs=nTrajs,scope="Training")

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,MODEL_PATH)
InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])

for i in range(settings["MaxEpisodes"]):

    sess.run(global_step_next)

    s0 = env.reset()

    for j in range(settings["MaxEpisodeSteps"]+1):

        a, networkData = net.GetAction(state=s0,episode=sess.run(global_step),step=j)

        s1,r,done,_ = env.step(action=a)

        net.AddToTrajectory([s0,a,r,s1,done]+networkData)

        s0 = s1

        if done or j == settings["MaxEpisodeSteps"]:
            net.Update(sess.run(global_step))
            break

    loggingDict = env.getLogging() #This method needs to be defined in a wrapper. (Have it outside loop because it does other things besides logging.)
    if interval_flag(sess.run(global_step), settings["LogFreq"], 'log'):
        dict = net.GetStatistics()
        loggingDict.update(dict)
        Record(loggingDict, writer, sess.run(global_step))

    if interval_flag(sess.run(global_step), settings["SaveFreq"], 'save'):
        saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))

    progbar.update(i)
