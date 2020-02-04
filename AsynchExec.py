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
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=250)])

for functionString in envSettings["StartingFunctions"]:
    StartingFunction = GetFunction(functionString)
    _,dFeatures,nActions,,nTrajs = StartingFunction(settings,envSettings)

GLOBAL_RUNNING_R = MovingAverage(400)

#Creating the Networks and Methods of the Run.
with tf.device('/gpu:0'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    GLOBAL_AC = A3C(network,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"])

# Create worker
    workers = []
    for i in range(settings["EnvHPs"]["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride,scope=i_name)
        localNetwork = A3C(network,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalAC=GLOBAL_AC,nTrajs=nTrajs)
        workers.append(Worker(i_name,localNetwork,, settings["EnvHPs"],global_step))


#Creating Auxilary Functions for logging and saving.
writer = tf.summary.create_file_writer(LOG_PATH)
GLOBAL_AC.InitializeVariablesFromFile(MODEL_PATH)

COORD = tf.train.Coordinator()
worker_threads = []
for worker in workers:
    job = lambda: worker.work(COORD,writer,MODEL_PATH,settings,envSettings)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)
