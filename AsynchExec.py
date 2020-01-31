"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import tensorflow as tf
import threading
import json

from networks.network import Network
from methods.A3C import A3C
from utils.utils import InitializeVariables,MovingAverage,CreatePath
from utils.record import Record,SaveHyperparams
from utils.worker import Worker as Worker


#Defining parameters and Hyperparameters for the run.
with open("configs/run/test.json") as json_file:
    settings = json.load(json_file)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment
sess = tf.Session()
env = gym.make('CartPole-v0')
env.seed(1)  # Create a consistent seed so results are reproducible.
env = env.unwrapped
N_F = env.observation_space.shape[0]
N_A = env.action_space.n

GLOBAL_RUNNING_R = MovingAverage(400)
global_episodes = 0
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step,1)

#Creating the Networks and Methods of the Run.
network = Network("configs/network/"+settings["NetworkConfig"],N_A,scope="Global")
GLOBAL_AC = A3C(network,sess,stateShape=N_F,actionSize=N_A,scope="Global",HPs=settings["NetworkHPs"])

# Create worker
workers = []
for i in range(settings["EnvHPs"]["NumberENV"]):
    i_name = 'W_%i' % i   # worker name
    network = Network("configs/network/"+settings["NetworkConfig"],N_A,scope=i_name)
    localNetwork = A3C(network,sess,stateShape=N_F,actionSize=N_A,scope=i_name,HPs=settings["NetworkHPs"],globalAC=GLOBAL_AC,)
    workers.append(Worker(i_name,localNetwork, sess, settings["EnvHPs"],global_step,global_step_next))


#Creating Auxilary Functions for logging and saving.
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.getVars+[global_step])
# SaveHyperparams(writer,HPs)
GLOBAL_AC.InitializeVariablesFromFile(saver,MODEL_PATH)
InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

COORD = tf.train.Coordinator()
worker_threads = []
for worker in workers:
    job = lambda: worker.work(COORD,GLOBAL_RUNNING_R,saver,writer,MODEL_PATH)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)
