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

from networks.network import Network
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import json
from utils.worker import Worker as Worker
from utils.utils import MovingAverage
import threading
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    env,dFeatures,nActions,nTrajs = StartingFunction(settings,envSettings,sess)

GLOBAL_RUNNING_R = MovingAverage(1000)

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
#Creating the Networks and Methods of the Run.
with tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    net = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"])

saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,MODEL_PATH)
print(sess.run(tf.report_uninitialized_variables()))
InitializeVariables(sess)

LOG_PATH = './images/AE/'+EXP_NAME
CreatePath(LOG_PATH)

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

#Add something for randomly sampling figures.
for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
    grid = ConstructSample(env,[i,j])
    if grid is None: continue
    state_new = net.PredictState(state=grid)
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(2,2,1)
    plt.title("State")
    imgplot = plt.imshow(grid[:,:,0],vmin=0, vmax=10)
    fig.add_subplot(2,2,2)
    plt.title("Predicted Next State")
    imgplot = plt.imshow(state_new[0][0,:,:,0],vmin=0, vmax=10)
    fig.add_subplot(2,2,3)
    plt.title("State")
    imgplot = plt.imshow(grid[:,:,1],vmin=0, vmax=10)
    fig.add_subplot(2,2,4)
    plt.title("Predicted Next State")
    imgplot = plt.imshow(state_new[0][0,:,:,1],vmin=0, vmax=10)
    # plt.show()
    plt.savefig(LOG_PATH+"/state"+str(i)+"_"+str(j)+".png")
    plt.close()
    # input()
