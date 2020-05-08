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
import collections.abc
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
parser.add_argument("-p", "--processor", required=False, default="/gpu:0",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")
parser.add_argument("-r", "--render", default=False, action="store_true",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")
args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
if args.environment is not None: envConfigOverride = json.loads(unquote(args.environment))
else: envConfigOverride = {}
if args.network is not None: netConfigOverride = json.loads(unquote(args.network))
else: netConfigOverride = {}

def Update(defaultSettings,overrides):
    for label,override in overrides.items():
        if isinstance(override, collections.abc.Mapping):
            Update(defaultSettings[label],override)
        else:
            defaultSettings[label] = override
    return defaultSettings

#Defining parameters and Hyperparameters for the run.
for (dirpath, dirnames, filenames) in os.walk("configs/run"):
    for filename in filenames:
        if args.file in filename:
            runConfigFile = os.path.join(dirpath,filename)
            break
with open(runConfigFile) as json_file:
    settings = json.load(json_file)
    settings.update(configOverride)
with open("configs/environment/"+settings["EnvConfig"]) as json_file:
    envSettings = json.load(json_file)
    envSettings = Update(envSettings,envConfigOverride)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment

_,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings,multiprocessing=1)

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device(args.processor):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    GLOBAL_AC = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"])
    GLOBAL_AC.Model.summary()
    saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.getVars+[global_step])
    GLOBAL_AC.InitializeVariablesFromFile(saver,MODEL_PATH)

    progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)

    # Create workers
    workers = []
    for i in range(settings["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = Network(settings["NetworkConfig"],nActions,netConfigOverride,scope=i_name)
        Method = GetFunction(settings["Method"])
        localNetwork = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalAC=GLOBAL_AC,nTrajs=nTrajs)
        localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
        env,_,_,_ = CreateEnvironment(envSettings,multiprocessing=1)
        workers.append(Worker(localNetwork,env,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

COORD = tf.train.Coordinator()
worker_threads = []
for i,worker in enumerate(workers):
    if i==0:
        job = lambda: worker.work(COORD,render=args.render)
    else:
        job = lambda: worker.work(COORD)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)
