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

from networks.networkAE import *
from networks.network_v3 import buildNetwork
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import json
from utils.worker import Worker as Worker
import threading
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.backend as K
from random import randint
from environments.Common import CreateEnvironment
import time

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

#Creating the Environment

env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
with tf.device('/gpu:0'):
    netConfigOverride["DefaultParams"]["Trainable"] = False
    SF1,SF2,SF3,SF4,SF5 = SFNetwork2(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global",SFSize=settings["SFSize"])
    SF5.load_weights(MODEL_PATH+"/model.h5")

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
    return actions   # return a int and extra data that needs to be fed to buffer.
def ConstructSample(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5: #Wall
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

s = []
s_next = []
r_store = []
label = []
for i in range(settings["SampleEpisodes"]):
    s0 = env.reset()

    for j in range(settings["MAX_EP_STEPS"]+1):

        a = GetAction(state=s0)

        s1,r,done,_ = env.step(a)

        s.append(s0)
        loc = np.where(s0 == 10)
        if loc[0]>=10 and loc[1]>=10:
            label.append(1)
        elif loc[0]<9 and loc[1]>=10:
            label.append(2)
        elif loc[0]>=10 and loc[1]<9:
            label.append(3)
        elif loc[0]<9 and loc[1]<9:
            label.append(4)
        else:
            label.append(5)
        s_next.append(s1)
        r_store.append(r)

        s0 = s1
        if done:
            break

####Evaluating using random sampling ########
# Processing state samples into Psi.
psiSamples = SF2.predict(np.stack(s)) # [X,SF Dim]
phi= SF3.predict(np.stack(s))

import time
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# mnist = fetch_mldata("MNIST original")
X = phi
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
    palette=sns.color_palette("hls", 5),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
if settings["SaveFigure"]:
    plt.savefig(LOG_PATH+"/PCA_2D_1.png")
else:
    plt.show()
sns.scatterplot(
    x="pca-two", y="pca-three",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

if settings["SaveFigure"]:
    plt.savefig(LOG_PATH+"/PCA_2D_2.png")
else:
    plt.show()

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

if settings["SaveFigure"]:
    plt.savefig(LOG_PATH+"/PCA_3D.png")
else:
    plt.show()
