import tensorflow as tf
import numpy as np
import gym, gym_minigrid, gym_cap

from utils.RL_Wrapper import TrainedNetwork
from utils.utils import InitializeVariables

"""
Framework for setting up an experiment.
"""

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
parser.add_argument("-p", "--processor", required=False, default="/gpu:0",
                    help="Processor identifier string. Ex. /cpu:0 /gpu:0")
parser.add_argument("-i", "--phi", required=False, default=True, action='store_false',
                    help="Whether to run the Phi training of a neural network. If tag is present will not run.")
parser.add_argument("-s", "--psi", required=False, default=True, action='store_false',
                    help="Whether to run the Psi training of a neural network. If tag is present will not run.")
parser.add_argument("-a", "--analysis", required=False, default=True, action='store_false',
                    help="Whether to run the Analysis of the network. If tag is present will not run.")
parser.add_argument("-d", "--data", required=False, default="",
                    help="Which data to use for the training.")
parser.add_argument("-l", "--load", required=False, default="",
                    help="Whether or not to load different models")

args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
if args.environment is not None: envConfigOverride = json.loads(unquote(args.environment))
else: envConfigOverride = {}
if args.network is not None: netConfigOverride = json.loads(unquote(args.network))
else: netConfigOverride = {}

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
MODEL_PATH = './models/'+EXP_NAME+ '/'
LOG_PATH = './images/SF/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Saving config files in the model directory
with open(LOG_PATH+'/runSettings.json', 'w') as outfile:
    json.dump(settings, outfile)
with open(MODEL_PATH+'/netConfigOverride.json', 'w') as outfile:
    json.dump(netConfigOverride, outfile)

#Creating the Environment
env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device(args.processor):
    SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")

    if args.load == "all":
        SF5.load_weights(MODEL_PATH+"/model.h5")
    elif args.load == "phi":
        SF1.load_weights(MODEL_PATH+"/model_phi.h5")
    elif args.load == "psi":
        SF2.load_weights(MODEL_PATH+"/model_psi.h5")
    elif args.load == "phi/psi":
        SF1.load_weights(MODEL_PATH+"/model_phi.h5")
        SF2.load_weights(MODEL_PATH+"/model_psi.h5")
    else: print("Did not load weights")

#Definition of misc. functions
def M4E(y_true,y_pred):
    return K.mean(K.pow(y_pred-y_true,4))
def OneHot(a,length=4):
    aOH = [0]*length
    aOH[int(a)] = 1
    return aOH

def GetAction(state):
    """
    Contains the code to run the network based on an input.
    """
    if "probs" in settings:
        probs = np.asarray(settings["probs"])
    else:
        p = 1/nActions
        if len(state.shape)==3:
            probs =np.full((1,nActions),p)
        else:
            probs =np.full((state.shape[0],nActions),p)
    actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
    return actions
if args.data == "":
    #Collecting samples
    s = []
    s_next = []
    r_store = []
    label = []
    action =[]
    for i in range(settings["SampleEpisodes"]):
        s0 = env.reset()

        for j in range(settings["MAX_EP_STEPS"]+1):

            a = GetAction(state=s0)

            s1,r,done,_ = env.step(a)

            s.append(s0)
            s_next.append(s1)
            r_store.append(r)
            action.append(OneHot(a))
            label.append(1)

            s0 = s1
            if done:
                break

else:
    loadedData = np.load('./data/'+args.data)
    s = loadedData["s"]
    s_next = loadedData["s_next"]
    r_store = loadedData["r_store"]
    action = loadedData["action"]
    if "label" in loadedData:
        label = loadedData["label"]

def PlotOccupancy(states,title=""):
    #Taking average over the list of states.
    state = np.stack(states)
    occupancy = np.amax(state,axis=0)
    #plotting them
    fig=plt.figure(figsize=(5.5, 5.5))
    fig.add_subplot(1,1,1)
    plt.title("State Occupancy")
    imgplot = plt.imshow(occupancy[:,:,0], vmin=0,vmax=10)
    plt.savefig(LOG_PATH+"/StateOccupancy_"+title+".png")
    plt.close()

def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]
#Defining Saving Functions for the models
class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs=None):
        self.superEpochs = superEpochs
    def on_epoch_end(self, epoch, logs=None):
        if self.superEpochs is not None:
            return
        else:
            if epoch%settings["SaveFreq"] == 0:
                SF5.save_weights(MODEL_PATH+"model.h5")
    def on_train_end(self, logs=None):
        if self.superEpochs is None:
            SF5.save_weights(MODEL_PATH+"model.h5")
        else:
            if self.superEpochs%settings["SaveFreq"] == 0:
                SF5.save_weights(MODEL_PATH+"model.h5")

class SaveModel_Phi(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%settings["SaveFreq"] == 0:
            SF1.save_weights(MODEL_PATH+"model_phi_"+str(epoch)+".h5")

class SaveModel_Psi(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs):
        self.superEpochs = superEpochs
    def on_train_end(self, logs=None):
        if self.superEpochs%settings["SaveFreq"] == 0:
            SF2.save_weights(MODEL_PATH+"model_psi_"+str(self.superEpochs)+".h5")

class ValueTest(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs):
        self.superEpochs = superEpochs
    def on_epoch_end(self,epoch, logs=None):
        if self.superEpochs%settings["LogFreq"] == 0:
            env.reset()
            rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                [value] = SF4.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(env.grid.encode()[:,:,0], vmin=0, vmax=10)
            fig.add_subplot(2,1,2)
            plt.title("Value Prediction")
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(LOG_PATH+"/ValuePred"+str(self.superEpochs)+".png")
            plt.close()

class ImageGenerator(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%settings["LogFreq"] == 0:
            for i in range(5):
                samp = i*100+randint(0,200)
                state = s[samp]
                act = action[samp]
                [state_new,reward] = SF1.predict([np.expand_dims(act,0),np.expand_dims(state,0)])
                fig=plt.figure(figsize=(5.5, 8))
                fig.add_subplot(2,1,1)
                plt.title("State")
                imgplot = plt.imshow(state[:,:,0], vmin=0, vmax=10)
                fig.add_subplot(2,1,2)
                plt.title("Predicted Next State")
                imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
                if i == 0:
                    plt.savefig(LOG_PATH+"/StatePredEpoch"+str(epoch)+".png")
                else:
                    plt.savefig(LOG_PATH+"/StatePred"+str(i)+".png")
                plt.close()


class RewardTest(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%settings["LogFreq"] == 0:
            env.reset()
            rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                [state_new,reward] = SF1.predict([np.stack([[0,0,0,0]]),np.expand_dims(grid,0)])
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(env.grid.encode()[:,:,0], vmin=0, vmax=10)
            fig.add_subplot(2,1,2)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(LOG_PATH+"/RewardPred"+str(epoch)+".png")
            plt.close()

#Defining which optimizer will be used for the training. Used for both Psi and Phi training.
if settings["Optimizer"] == "Adam":
    opt = tf.keras.optimizers.Adam(settings["LearningRate"])
elif settings["Optimizer"] == "RMS":
    opt = tf.keras.optimizers.RMSprop(settings["LearningRate"])
elif settings["Optimizer"] == "Adagrad":
    opt = tf.keras.optimizers.Adagrad(settings["LearningRate"])
elif settings["Optimizer"] == "Adadelta":
    opt = tf.keras.optimizers.Adadelta(settings["LearningRate"])
elif settings["Optimizer"] == "Adamax":
    opt = tf.keras.optimizers.Adamax(settings["LearningRate"])
elif settings["Optimizer"] == "Nadam":
    opt = tf.keras.optimizers.Nadam(settings["LearningRate"])
elif settings["Optimizer"] == "SGD":
    opt = tf.keras.optimizers.SGD(settings["LearningRate"])
elif settings["Optimizer"] == "SGD-Nesterov":
    opt = tf.keras.optimizers.SGD(settings["LearningRate"],nesterov=True)
elif settings["Optimizer"] == "Amsgrad":
    opt = tf.keras.optimizers.Adam(settings["LearningRate"],amsgrad=True)

if args.phi:
    PlotOccupancy(s,title="TrainingOccupancy")
    SF1.compile(optimizer=opt, loss=[M4E,"mse"], loss_weights = [1.0,1.0])
    SF1.fit(
        [np.stack(action),np.stack(s)],
        [np.stack(s_next),np.stack(r_store)],
        epochs=settings["PhiEpochs"],
        batch_size=settings["BatchSize"],
        shuffle=True,
        callbacks=[ImageGenerator(),SaveModel(),SaveModel_Phi(),RewardTest()])

if "DefaultParams" not in netConfigOverride:
    netConfigOverride["DefaultParams"] = {}
netConfigOverride["DefaultParams"]["Trainable"]=False
with tf.device(args.processor):
    SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    SF5.load_weights(MODEL_PATH+"/model.h5")

if args.psi:
    SF2.compile(optimizer=opt, loss="mse")
    phi = SF3.predict([np.stack(s)])
    gamma=settings["Gamma"]
    for i in range(settings["PsiEpochs"]):

        psi_next = SF2.predict([np.stack(s_next)])

        labels = phi+gamma*psi_next
        SF2.fit(
            [np.stack(s)],
            [np.stack(labels)],
            epochs=settings["FitIterations"],
            batch_size=settings["BatchSize"],
            shuffle=True,
            callbacks=[ValueTest(i),SaveModel(i),SaveModel_Psi(i)])

if args.analysis:
    psi = SF2.predict([np.stack(s)]) # [X,SF Dim]
    phi = SF3.predict([np.stack(s)])

    ##-Repeat M times to evaluate the effect of sampling.
    M = 3
    dim = psi.shape[1]
    for replicate in range(M):
        #Taking Eigenvalues and Eigenvectors of the environment,
        psiSamples = np.zeros([dim,dim])
        #Randomly collecting samples from the random space.
        s_sampled=[]
        for i in range(dim):
            sample = randint(1,psiSamples.shape[0])
            s_sampled.append(s[sample])
            psiSamples[i,:] = psi[sample,:]
        PlotOccupancy(s_sampled,title="Replicate"+str(replicate))

        w_g,v_g = np.linalg.eig(psiSamples)

        #Creating Eigenpurposes of the N highest Eigenvectors and saving images
        N = 5
        offset = 0
        for sample in range(N):

            v_option=np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                phi= SF3.predict([np.expand_dims(grid,0)])
                if sample+offset >= dim:
                    continue
                v_option[i,j]=np.matmul(phi,v_g[:,sample+offset])[0]
                if np.iscomplex(w_g[sample+offset]):
                    offset+=1

            imgplot = plt.imshow(v_option)
            plt.title("Replicate  "+str(replicate)+" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
            plt.savefig(LOG_PATH+"/option"+str(sample)+"replicate"+str(replicate)+".png")
            plt.close()

        #Doing an uniform sampling option plots
        s = [];label=[];a=[]
        for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
            grid = ConstructSample(env,[i,j])
            if grid is None: continue
            s.append(grid)
            label.append(1)
            a.append([0,0,0,0])
        s.extend(s);label.extend(label);a.extend(a)
        s.extend(s);label.extend(label);a.extend(a)
        s.extend(s);label.extend(label);a.extend(a)
        s.extend(s);label.extend(label);a.extend(a)
        s.extend(s);label.extend(label);a.extend(a)
        psi_uniform = SF2.predict([np.stack(s)])
        psiSamples = np.zeros([dim,dim])
        #Randomly collecting samples from the random space.
        s_sampled=[]
        for i in range(dim):
            psiSamples[i,:] = psi_uniform[i,:]
            s_sampled.append(s[i])
        w_g,v_g = np.linalg.eig(psiSamples)

        #Creating Eigenpurposes of the N highest Eigenvectors and saving images
        N = 5
        offset = 0
        for sample in range(N):

            v_option=np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = ConstructSample(env,[i,j])
                if grid is None: continue
                phi= SF3.predict([np.expand_dims(grid,0)])
                if sample+offset >= dim:
                    continue
                v_option[i,j]=np.matmul(phi,v_g[:,sample+offset])[0]
                if np.iscomplex(w_g[sample+offset]):
                    offset+=1

            imgplot = plt.imshow(v_option)
            plt.title("Uniform Sampling Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
            plt.savefig(LOG_PATH+"/option"+str(sample)+"uniform.png")
            plt.close()



    import pandas as pd
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns


    # mnist = fetch_mldata("MNIST original")
    X = psi_uniform
    y = np.stack(label)

    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]


    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 1),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
    plt.savefig(LOG_PATH+"/PCA_2D_1.png")
    plt.close()
    sns.scatterplot(
        x="pca-two", y="pca-three",
        hue="y",
        palette=sns.color_palette("hls", 1),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )

    plt.savefig(LOG_PATH+"/PCA_2D_2.png")
    plt.close()

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[rndperm,:]["pca-one"],
        ys=df.loc[rndperm,:]["pca-two"],
        zs=df.loc[rndperm,:]["pca-three"],
        c=df.loc[rndperm,:]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')


    plt.savefig(LOG_PATH+"/PCA_3D.png")
