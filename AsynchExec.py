"""
Framework for setting up an experiment.
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
from utils.worker import Worker as Worker
from utils.utils import MovingAverage
import threading


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
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)

for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    _,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings,sess)

GLOBAL_RUNNING_R = MovingAverage(400)
GLOBAL_EP_LEN = MovingAverage(400)

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
#Creating the Networks and Methods of the Run.
with tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    GLOBAL_AC = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"])
    GLOBAL_AC.Model.summary()
    saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.getVars+[global_step])
    GLOBAL_AC.InitializeVariablesFromFile(saver,MODEL_PATH)

# Create worker
    workers = []
    for i in range(settings["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride,scope=i_name)
        Method = GetFunction(settings["Method"])
        localNetwork = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalAC=GLOBAL_AC,nTrajs=nTrajs)
        localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
        workers.append(Worker(i_name,localNetwork,sess, settings["EnvHPs"],global_step,global_step_next,settings,envSettings,progbar))

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
InitializeVariables(sess) #Included to catch if there are any uninitalized variables.


COORD = tf.train.Coordinator()
worker_threads = []
for worker in workers:
    job = lambda: worker.work(COORD,writer,MODEL_PATH,settings,envSettings,saver,GLOBAL_RUNNING_R,GLOBAL_EP_LEN)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)
