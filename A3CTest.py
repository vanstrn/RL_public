"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import tensorflow as tf
import threading


from networks.network import Network
from networks.dnn_1out import DNN10ut,DNN10ut_
from networks.dnn_2out import DNN2Out
from methods.A3C import A3C,A3C_s
from utils.utils import InitializeVariables,MovingAverage
from utils.record import Record,SaveHyperparams
from utils.worker import Worker


if __name__ == "__main__":
    #Defining parameters and Hyperparameters for the run.
    HPs = {
        "MAX_EP_STEPS" : 1000,
        "MAX_EP" : 10000,
        "SAVE_FREQ" : 100,
        "LOG_FREQ" : 10,
        "Critic LR": 1E-3,
        "Actor LR": 1E-4,
        "N_WORKERS":1,
        "UPDATE_GLOBAL_ITER":10,
        "GAMMA":.9
        }
    EXP_NAME = 'Test15'
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME

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
    # criticNetwork = DNN10ut("critic",1,networkName="Critic")
    # actorNetwork = DNN10ut("actor",N_A,)
    # net = A2C(actorNetwork,criticNetwork,sess,stateShape=[1,N_F],actionSize=N_A)
    network = DNN2Out("Global",N_A,1,networkName="Test")
    GLOBAL_AC = A3C_s(network,sess,stateShape=N_F,actionSize=N_A,scope="Global",lr_c=HPs["Critic LR"],lr_a=HPs["Actor LR"])
    workers = []
    # Create worker
    for i in range(HPs["N_WORKERS"]):
        i_name = 'W_%i' % i   # worker name
        network = DNN2Out(i_name,N_A,1,networkName="Test")
        localNetwork = A3C_s(network,sess,stateShape=N_F,actionSize=N_A,scope=i_name,globalAC=GLOBAL_AC,lr_c=HPs["Critic LR"],lr_a=HPs["Actor LR"])
        workers.append(Worker(i_name,localNetwork, sess, HPs,global_step,global_step_next))


    #Creating Auxilary Functions for logging and saving.
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.getVars+[global_step])
    SaveHyperparams(writer,HPs)
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
