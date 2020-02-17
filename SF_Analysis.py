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
    network = Network("configs/network/"+settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    net = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"])

saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,MODEL_PATH)
x = sess.run(tf.report_uninitialized_variables() )
print(x)
InitializeVariables(sess)

psiSamples = []
phiSamples = []
vSamples = []

for i in range(settings["EnvHPs"]["SampleEpisodes"]):

    for functionString in envSettings["BootstrapFunctions"]:
        BootstrapFunctions = GetFunction(functionString)
        s0, loggingDict = BootstrapFunctions(env,settings,envSettings,sess)

    for functionString in envSettings["StateProcessingFunctions"]:
        StateProcessing = GetFunction(functionString)
        s0 = StateProcessing(s0,env,envSettings,sess)

    for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):

        a, networkData = net.GetAction(state=s0,episode=sess.run(global_step),step=j)
        vSamples.append(networkData[0])
        phiSamples.append(networkData[1])
        psiSamples.append(networkData[2])

        for functionString in envSettings["ActionProcessingFunctions"]:
            ActionProcessing = GetFunction(functionString)
            a = ActionProcessing(a,env,envSettings,sess)

        s1,r,done,_ = env.step(a)
        env.render()
        for functionString in envSettings["StateProcessingFunctions"]:
            StateProcessing = GetFunction(functionString)
            s1 = StateProcessing(s1,env,envSettings,sess)

        for functionString in envSettings["RewardProcessingFunctions"]:
            RewardProcessing = GetFunction(functionString)
            r,done = RewardProcessing(s1,r,done,env,envSettings,sess)

        s0 = s1

        if done.all():
            break
psiSamples = np.vstack(psiSamples)
# print(psiSamples[:128,:].shape)

w_g,v_g = np.linalg.eig(psiSamples[:128,:])
# print(np.real(v_g[127,:]))
# def filter()

# print(vSamples[0],np.mean(phiSamples[0] * np.real(v_g[0,:])))

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

import itertools
v=np.zeros((dFeatures[0],dFeatures[1]))
psi0=np.zeros((dFeatures[0],dFeatures[1],128))
for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
    grid = ConstructSample(env,[i,j])
    if grid is None: continue
    a, networkData = net.GetAction(state=grid,episode=sess.run(global_step),step=j)
    v[i,j]=networkData[0]
    psi0[i,j,:]=np.matrix(networkData[1])* np.matrix(np.real(v_g))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Check0(array):
    counter = 0
    for i,j in itertools.product(range(array.shape[0]),range(array.shape[1])):
        if array[i,j] <= 0.0001: counter += 1
    if counter >= 2: return True
    return False

for i in range(128):
    if Check0(psi0[1:-2,1:-2,i]):
        continue
    imgplot = plt.imshow(psi0[1:-2,1:-2,i])
    plt.show()
