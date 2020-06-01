
"""
Framework for setting up an experiment.
"""
import time
import numpy as np
import gym
import gym_minigrid,gym_cap
import tensorflow as tf
import argparse
from urllib.parse import unquote
import os

from networks.network_v3 import buildNetwork
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
from environments.Common import CreateEnvironment

#Input arguments to select details of operation and add overrides for the default config Files
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True,
                    help="File for specific run. Located in ./configs/run")
parser.add_argument("-c", "--config", required=False,
                    help="JSON configuration string to override runtime configs of the script.")
parser.add_argument("-e", "--environment", required=False,
                    help="JSON configuration string to override environment parameters")
parser.add_argument("-n", "--network", required=False,
                    help="JSON configuration string to override network parameters")
parser.add_argument("-s", "--sfnetwork", required=False,
                    help="JSON configuration string to override network parameters")
parser.add_argument("-p", "--processor", required=False, default="/gpu:0",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")
parser.add_argument("-r", "--reward", required=False, default=False, action='store_true',
                    help="Determines if the sub-policies are trained and not Q-tables.")
parser.add_argument("-l", "--load", required=False, default="",
                    help="Whether or not to load different data")

args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
if args.environment is not None: envConfigOverride = json.loads(unquote(args.environment))
else: envConfigOverride = {}
if args.network is not None: netConfigOverride = json.loads(unquote(args.network))
else: netConfigOverride = {}
if args.sfnetwork is not None: SFnetConfigOverride = json.loads(unquote(args.sfnetwork))
else: SFnetConfigOverride = {}

#Reading in Config Files
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
    envSettings.update(envConfigOverride)

EXP_NAME = settings["RunName"]
if "LoadName" in settings:
    LOAD_NAME = settings["LoadName"]
else:
    LOAD_NAME = settings["RunName"]

MODEL_PATH = './models/'+LOAD_NAME+ '/'
IMAGE_PATH = './images/SF/'+EXP_NAME
ts = str(time.time())
MODEL_PATH_ = './models/'+EXP_NAME+"_"+ts+'/'
LOG_PATH = './logs/'+EXP_NAME+"_"+ts
CreatePath(LOG_PATH)
CreatePath(IMAGE_PATH)
CreatePath(MODEL_PATH)
CreatePath(MODEL_PATH_)

#Creating the Environment
env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device(args.processor):
    SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["SFNetworkConfig"],nActions,SFnetConfigOverride,scope="Global")
    # SF5.load_weights(MODEL_PATH+"model.h5")


#Collecting Samples for the Decomposition Analysis.
def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

def PlotOccupancy(states,title=""):
    #Taking average over the list of states.
    state = np.stack(states)
    occupancy = np.amax(state,axis=0)
    #plotting them
    fig=plt.figure(figsize=(5.5, 5.5))
    fig.add_subplot(1,1,1)
    plt.title("State Occupancy")
    imgplot = plt.imshow(occupancy[:,:,0], vmin=0,vmax=10)
    plt.savefig(IMAGE_PATH+"/StateOccupancy_"+title+".png")
    plt.close()

def SmoothOption(option, gamma =0.9):
    # option[option<0.0] = 0.0
    #Create the Adjacency Matric
    states_ = {}
    count = 0
    for i in range(option.shape[0]):
        for j in range(option.shape[1]):
            if option[i,j] != 0:
                states_[count] = [i,j]
                # states_.append([count, [i,j]])
                count+=1
    states=len(states_.keys())
    x = np.zeros((states,states))
    for i in range(len(states_)):
        [locx,locy] = states_[i]
        sum = 0
        for j in range(len(states_)):
            if states_[j] == [locx+1,locy]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx-1,locy]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx,locy+1]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx,locy-1]:
                x[i,j] = 0.25
                sum += 0.25
        x[i,i]= 1.0-sum

    #Create W
    w = np.zeros((states))
    for count,loc in states_.items():
        w[count] = option[loc[0],loc[1]]

    # (I-gamma*Q)^-1
    I = np.identity(states)
    psi = np.linalg.inv(I-gamma*x)

    smoothedOption = np.zeros_like(option,dtype=float)

    value = np.matmul(psi,w)
    for j,loc in states_.items():
        smoothedOption[loc[0],loc[1]] = value[j]

    return smoothedOption

def GetAction(state):
    """
    Contains the code to run the network based on an input.
    """
    p = 1/nActions
    if len(state.shape)==3:
        probs =np.full((1,nActions),p)
    else:
        probs =np.full((state.shape[0],nActions),p)
    actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
    return actions
if args.load == "":
    s = []
    for i in range(settings["SampleEpisodes"]):
        s0 = env.reset()

        for j in range(settings["MAX_EP_STEPS"]+1):

            a = GetAction(state=s0)

            s1,r,done,_ = env.step(a)

            s.append(s0)

            s0 = s1
            if done:
                break
elif args.load == "Uniform" or args.load == "RUniform":
    s = []
    for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        grid = ConstructSample(env,[i,j])
        if grid is None: continue
        s.append(grid)
    s.extend(s)
    s.extend(s)
    s.extend(s)
    s.extend(s)
    s.extend(s)
else:
    loadedData = np.load('./data/'+args.data)
    s = loadedData["s"]


psi = SF2.predict(np.stack(s)) # [X,SF Dim]

dim = psi.shape[1]

if args.load != "" or args.load != "RUniform":
    psiSamples = np.zeros([dim,dim])
    #Randomly collecting samples from the random space.
    s_sampled=[]
    for i in range(dim):
        psiSamples[i,:] = psi[i,:]
        s_sampled.append(s[i])
    PlotOccupancy(s_sampled,title="Policy")
else:
    #Taking Eigenvalues and Eigenvectors of the environment,
    psiSamples = np.zeros([dim,dim])
    #Randomly collecting samples from the random space.
    s_sampled=[]
    for i in range(dim):
        sample = randint(1,psiSamples.shape[0])
        psiSamples[i,:] = psi[sample,:]
        s_sampled.append(s[sample])
    PlotOccupancy(s_sampled,title="Policy")

w_g,v_g = np.linalg.eig(psiSamples)

#Creating Sub-policies
N = settings["NumOptions"]
offset = 0
options = []
for sample in range(N):

    v_option=np.full((dFeatures[0],dFeatures[1]),0,dtype=np.float32)
    for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        grid = ConstructSample(env,[i,j])
        if grid is None: continue
        phi= SF3.predict(np.expand_dims(grid,0))
        if sample+offset >= dim:
            continue
        v_option[i,j]=np.matmul(phi,v_g[:,sample+offset])[0]
    if np.iscomplex(w_g[sample+offset]):
        offset+=1
    v_option_ = SmoothOption(v_option)
    options.append(v_option_)
    imgplot = plt.imshow(v_option_)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+".png")
    plt.close()


def UseSubpolicy(s,subpolicyNum):
    #Extracting location of agent.
    locX,locY = np.unravel_index(np.argmax(s[:,:,0], axis=None), s[:,:,0].shape)
    #Getting Value of all adjacent policies. Ignoring location of the walls.
    actionValue = []
    if [int(locX),int(locY+1),0] == 2:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX),  int(locY+1)  ]) # Go Up
    if [int(locX+1),int(locY),0] == 2:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX+1),int(locY)    ]) # Go Right
    if [int(locX),int(locY-1),0] == 2:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX),  int(locY-1)  ]) # Go Down
    if [int(locX-1),int(locY),0] == 2:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX-1),int(locY)    ]) # Go Left

    #Selecting Action with Highest Value. Will always take a movement.
    return actionValue.index(max(actionValue))

from networks.network import Network

#Creating High level policy
with tf.device(args.processor):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)
    network = Network(settings["NetworkConfig"],N,netConfigOverride)
    Method = GetFunction(settings["Method"])
    net = Method(network,sess,scope="net",stateShape=dFeatures,actionSize=N,HPs=settings["NetworkHPs"],nTrajs=nTrajs)

#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
net.InitializeVariablesFromFile(saver,MODEL_PATH_)
InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])

loggingFunctions=[]
for loggingFunc in settings["LoggingFunctions"]:
    func = GetFunction(loggingFunc)
    loggingFunctions.append(func(env,net,IMAGE_PATH))

for i in range(settings["MAX_EP"]):

    sess.run(global_step_next)
    logging = interval_flag(sess.run(global_step), settings["LogFreq"], 'log')
    saving = interval_flag(sess.run(global_step), settings["SaveFreq"], 'save')

    s0 = env.reset()

    for j in range(settings["MAX_EP_STEPS"]+1):
        updating = interval_flag(j, settings['UPDATE_GLOBAL_ITER'], 'update')

        a_hier, networkData = net.GetAction(state=s0,episode=sess.run(global_step),step=j)
        a = UseSubpolicy(s0,a_hier)
        s1,r,done,_ = env.step(action=a)

        if j == settings["MAX_EP_STEPS"]:
            done = True
        net.AddToTrajectory([s0,a_hier,r,s1,done]+networkData)

        s0 = s1
        if updating:   # update global and assign to local net
            net.Update(settings["NetworkHPs"],sess.run(global_step))
        if done or j == settings["MAX_EP_STEPS"]:
            net.Update(settings["NetworkHPs"],sess.run(global_step))
            break

    loggingDict = env.getLogging()
    if logging:
        dict = net.GetStatistics()
        loggingDict.update(dict)
        Record(loggingDict, writer, sess.run(global_step))
        for func in loggingFunctions:
            func(sess.run(global_step))

    if saving:
        saver.save(sess, MODEL_PATH_+'/ctf_policy.ckpt', global_step=sess.run(global_step))
    progbar.update(i)
