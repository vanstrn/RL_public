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
with open("configs/run/"+args.file) as json_file:
    settings = json.load(json_file)
    settings = Update(settings,configOverride)
with open("configs/environment/"+settings["EnvConfig"]) as json_file:
    envSettings = json.load(json_file)
    envSettings = Update(envSettings,envConfigOverride)

EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Creating the Environment

class WorkerSlave(object):
    def __init__(self,localNetwork,env,sess,global_step,global_step_next,settings,
                    progbar,writer,MODEL_PATH,saver):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess=sess
        self.env=env
        self.net = localNetwork
        self.global_step = global_step
        self.global_step_next = global_step_next
        self.settings =settings
        self.progbar =progbar
        self.writer=writer
        self.MODEL_PATH=MODEL_PATH
        self.saver=saver

    def work(self,COORD,render=False):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.settings["MaxEpisodes"]:

            self.sess.run(self.global_step_next)

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'log')
            saving = interval_flag(self.sess.run(self.global_step), self.settings["SaveFreq"], 'save')

            s0 = self.env.reset()

            for j in range(self.settings["MaxEpisodeSteps"]+1):

                a,networkData = self.net.GetAction(state = s0,episode=self.sess.run(self.global_step),step=j)

                s1,r,done,_ = self.env.step(a)
                if render:
                    self.env.render()

                self.net.AddToTrajectory([s0,a,r,s1,done]+networkData)

                s0 = s1
                if done or j == self.settings["MaxEpisodeSteps"]:   # update global and assign to local net
                    self.net.PushToBuffer()
                    break

            self.progbar.update(self.sess.run(self.global_step))
            if logging:
                loggingDict = self.env.getLogging()
                Record(loggingDict, self.writer, self.sess.run(self.global_step))
            if saving:
                self.saver.save(self.sess, self.MODEL_PATH+'/ctf_policy.ckpt', global_step=self.sess.run(self.global_step))

class WorkerMaster(object):
    def __init__(self,localNetwork,sess,global_step,global_step_next,settings,
                    progbar,writer,MODEL_PATH,saver):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess=sess
        self.net = localNetwork
        self.global_step = global_step
        self.global_step_next = global_step_next
        self.settings =settings
        self.progbar =progbar
        self.writer=writer
        self.MODEL_PATH=MODEL_PATH
        self.saver=saver

    def work(self,COORD,render=False):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.settings["MaxEpisodes"]:

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'log')

            self.net.Update(self.settings["NetworkHPs"],self.sess.run(self.global_step))

            if logging:
                loggingDict = self.net.GetStatistics()
                Record(loggingDict, self.writer, self.sess.run(self.global_step))


class ApexBuffer():
    def __init__(self,maxSamples=10000):
        self.maxSamples = maxSamples
        self.buffer=[]
        self.priorities=[]
        self.trajLengths=[]
        self.flag = True
        self.slice=0
        self.sampleSize=0

    def AddTrajectory(self,sample,priority):
        self.buffer.append(sample)
        self.priorities.append(priority)
        self.trajLengths.append(len(sample[0]))


    def Sample(self):
        return self.buffer[0:self.slice] , self.sampleSize


    def PrioritizeandPruneSamples(self,sampleSize):
        if len(self.trajLengths) ==0:
            return
        if self.flag:
            self.flag=False
        self.priorities, self.buffer,self.trajLengths = (list(t) for t in zip(*sorted(zip(self.priorities, self.buffer,self.trajLengths), reverse=True)))

        #Pruning the least favorable samples
        while sum(self.trajLengths) >= self.maxSamples:
            self.priorities.pop(-1)
            self.buffer.pop(-1)
            self.trajLengths.pop(-1)
        self.sampleSize = 0;self.slice=0
        for length in self.trajLengths:
            self.sampleSize += length
            self.slice +=1
            if self.sampleSize > sampleSize:
                break


    def UpdatePriorities(self,priorities):
        self.priorities[0:self.slice] = priorities
        self.flag = True
        return self.buffer

sharedBuffer = ApexBuffer()
_,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings,multiprocessing=1)

#Creating the Networks and Methods of the Run.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings["GPUCapacitty"], allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
workers = []
progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
with tf.device(args.processor):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    network = Network(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    Updater = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"],sharedBuffer=sharedBuffer)
    Updater.Model.summary()
    saver = tf.train.Saver(max_to_keep=3, var_list=Updater.getVars+[global_step])
    Updater.InitializeVariablesFromFile(saver,MODEL_PATH)
    workers.append(WorkerMaster(Updater,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

    # Create workers
    for i in range(settings["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = Network(settings["NetworkConfig"],nActions,netConfigOverride,scope=i_name)
        Method = GetFunction(settings["Method"])
        localNetwork = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalAC=Updater,nTrajs=nTrajs,sharedBuffer=sharedBuffer)
        localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
        env,_,_,_ = CreateEnvironment(envSettings,multiprocessing=1)
        workers.append(WorkerSlave(localNetwork,env,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

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
