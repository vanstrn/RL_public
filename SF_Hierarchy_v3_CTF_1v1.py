

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
LoadName = settings["LoadName"]
MODEL_PATH = './models/'+LoadName+ '/'
ts = str(time.time())
IMAGE_PATH = './images/SF/'+EXP_NAME+"_"+ts+'/'
MODEL_PATH_ = './models/'+EXP_NAME+"_"+ts+'/'
LOG_PATH = './logs/CTF_1v1/'+EXP_NAME+"_"+ts
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
    SF5.load_weights(MODEL_PATH+"model.h5")

#Collecting Samples for the Decomposition Analysis.

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
s = []
s_next = []
r_store = []

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

for i in range(settings["SampleEpisodes"]):
    s0 = env.reset()

    for j in range(settings["MAX_EP_STEPS"]+1):

        a = GetAction(state=s0)

        s1,r,done,_ = env.step(a)
        if arreq_in_list(s0,s):
            pass
        else:
            s.append(s0)
            s_next.append(s1)
            r_store.append(r)

        s0 = s1
        if done:
            break


def ConstructSample(env,position1,position2):
    if position1 == position2:
        return None
    grid = env.get_obs_blue
    locX,locY = np.unravel_index(np.argmax(grid[:,:,4], axis=None), grid[:,:,0].shape)
    locX2,locY2 = np.unravel_index(np.argmin(grid[:,:,4], axis=None), grid[:,:,0].shape)
    #Removing the agent
    grid[locX,locY,4] = 0
    grid[locX2,locY2,4] = 0
    if grid[position1[0],position1[1],3] == 1 or grid[position2[0],position2[1],3] == 1:
        return None
    grid[position1[0],position1[1],4] = 5
    grid[position2[0],position2[1],4] = -5
    return grid

def SmoothOption(option_, gamma =0.9):
    # option[option<0.0] = 0.0
    #Create the Adjacency Matric
    v_option=np.full((dFeatures[0],dFeatures[1],dFeatures[0],dFeatures[1]),0,dtype=np.float32)
    for i2,j2 in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        option = option_[:,:,i2,j2]
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

        v_option[:,:,i2,j2] = smoothedOption
    return v_option

#Selecting the samples:
psi = SF2.predict(np.vstack(s)) # [X,SF Dim]

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)
#test for approximate equality (for floating point types)
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr,atol=1E-6)), False)

if settings["Selection"]=="First":
    samples = [];points=[]
    i =0
    while len(samples) < settings["TotalSamples"]:
        if not arreqclose_in_list(psi[i,:], samples):
            samples.append(psi[i,:])
            points.append(i)
        i+=1
elif settings["Selection"]=="Random":
    samples = [];points=[]
    while len(samples) < settings["TotalSamples"]:
        idx = randint(1,psi.shape[0])
        if not arreqclose_in_list(psi[idx,:], samples):
            samples.append(psi[idx,:])
            points.append(idx)
elif settings["Selection"]=="Hull-pca":
    #PCA Decomp to dimension:
    import pandas as pd
    from sklearn.decomposition import PCA
    feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
    df = pd.DataFrame(psi,columns=feat_cols)
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)

    from SampleSelection import SampleSelection_v2
    points = SampleSelection_v2(pca_result,settings["TotalSamples"],returnIndicies=True)
elif settings["Selection"]=="Hull_tsne":
    #PCA Decomp to dimension:
    import pandas as pd
    from sklearn.manifold import TSNE
    feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
    df = pd.DataFrame(psi,columns=feat_cols)
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    from SampleSelection import SampleSelection_v2
    points = SampleSelection_v2(tsne_results,settings["TotalSamples"],returnIndicies=True)
elif settings["Selection"]=="Hull_cluster":
    #PCA Decomp to dimension:
    import pandas as pd
    from sklearn.decomposition import PCA
    feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
    df = pd.DataFrame(psi,columns=feat_cols)
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)

    from SampleSelection import SampleSelection_v3
    points = SampleSelection_v3(pca_result,settings["TotalSamples"],returnIndicies=True)

psiSamples=[]
for point in points:
    psiSamples.append(psi[point,:])

while len(psiSamples) < len(psiSamples[0]):
    psiSamples.extend(psiSamples)

samps = np.stack(psiSamples)
samps2 = samps[0:samps.shape[1],:]
w_g,v_g = np.linalg.eig(samps2)

# print("here")
dim = samps2.shape[1]
#Creating Sub-policies
N = settings["NumOptions"]
offset = 0
options = []
for sample in range(int(N/2)):
    print("Creating Option",sample)
    v_option=np.full((dFeatures[0],dFeatures[1],dFeatures[0],dFeatures[1]),0,dtype=np.float32)
    for i1,j1,i2,j2 in itertools.product(range(dFeatures[0]),range(dFeatures[1]),range(dFeatures[0]),range(dFeatures[1])):
        if sample+offset >= dim:
            continue
        grid = ConstructSample(env,[i1,j1],[i2,j2])
        if grid is None: continue
        phi= SF3.predict(np.expand_dims(grid,0))
        v_option[i1,j1,i2,j2]=np.matmul(phi,v_g[:,sample+offset])[0]
        if np.iscomplex(w_g[sample+offset]):
            offset+=1
    print("Smoothing Option")
    v_option_ = SmoothOption(v_option)
    options.append(v_option_)
    options.append(-v_option_)
    #Plotting the first couple samples with random enemy positions:
    v_map = v_option_[:,:,10,10]
    imgplot = plt.imshow(v_map)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(1)+".png")
    plt.close()
    v_map = v_option_[:,:,10,15]
    imgplot = plt.imshow(v_map)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(2)+".png")
    plt.close()
    v_map = v_option_[:,:,15,10]
    imgplot = plt.imshow(v_map)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(3)+".png")
    plt.close()
    v_map = v_option_[:,:,10,5]
    imgplot = plt.imshow(v_map)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(4)+".png")
    plt.close()
    v_map = v_option_[:,:,5,10]
    imgplot = plt.imshow(v_map)
    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
    plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(5)+".png")
    plt.close()


def UseSubpolicy(s,subpolicyNum):
    #Extracting location of agent.
    locX,locY = np.unravel_index(np.argmax(s[0,:,:,4], axis=None), s[0,:,:,0].shape)
    locX2,locY2 = np.unravel_index(np.argmin(s[0,:,:,4], axis=None), s[0,:,:,0].shape)
    #Getting Value of all adjacent policies. Ignoring location of the walls.
    actionValue = [options[int(subpolicyNum)][int(locX),int(locY),int(locX2),int(locY2)]]
    if locY-1 < 0:
        actionValue.append(-999)
    elif [0,int(locX),int(locY-1),3] == 1:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX),  int(locY-1),int(locX2),int(locY2)  ]) # Go Up

    if locX+1 > s0.shape[2]-1:
        actionValue.append(-999)
    elif [0,int(locX+1),int(locY),3] == 1:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX+1),int(locY),int(locX2),int(locY2)    ]) # Go Right

    if locY+1 > s0.shape[1]-1:
        actionValue.append(-999)
    elif [0,int(locX),int(locY+1),3] == 1:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX),  int(locY+1),int(locX2),int(locY2)  ]) # Go Down

    if locX-1<0:
        actionValue.append(-999)
    elif [0,int(locX-1),int(locY),3] == 1:
        actionValue.append(-999)
    else:
        actionValue.append(options[int(subpolicyNum)][int(locX-1),int(locY),int(locX2),int(locY2)    ]) # Go Left

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
    # print(j,r)
