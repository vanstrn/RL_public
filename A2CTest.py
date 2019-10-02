"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import tensorflow as tf

from networks.network import Network
from networks.dnn_1out import DNN10ut,DNN10ut_
from networks.dnn_2out import DNN2Out
from methods.A2C import A2C,A2C_s
from utils.utils import InitializeVariables
from utils.record import Record,SaveHyperparams

if __name__ == "__main__":
    #Defining parameters and Hyperparameters for the run.
    HPs = {
        "MAX_EP_STEPS" : 1000,
        "MAX_EP" : 10000,
        "SAVE_FREQ" : 100,
        "LOG_FREQ" : 10,
        "Critic LR": 1E-3,
        "Actor LR": 1E-4,
        "Number Environments" : 4,
        }
    EXP_NAME = 'Test11'
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME

    #Creating the Environment
    sess = tf.Session()
    env = gym.make('CartPole-v0')
    env.seed(1)  # Create a consistent seed so results are reproducible.
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    global_episodes = 0
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    #Creating the Networks and Methods of the Run.
    # criticNetwork = DNN10ut("critic",1,networkName="Critic")
    # actorNetwork = DNN10ut("actor",N_A,)
    # net = A2C(actorNetwork,criticNetwork,sess,stateShape=[1,N_F],actionSize=N_A)

    network = DNN2Out("AC",N_A,1,networkName="Test")
    net = A2C_s(network,sess,stateShape=[None,N_F],actionSize=N_A,lr_c=HPs["Critic LR"],lr_a=HPs["Actor LR"])

    #Creating Auxilary Functions for logging and saving.
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
    SaveHyperparams(writer,HPs)
    net.InitializeVariablesFromFile(saver,MODEL_PATH)
    InitializeVariables(sess) #Included to catch if there are any uninitalized variables.
    x = network.getVars
    print(x)
    x = network.getAParams
    print(x)

    #Running the Simulation
    for i in range(HPs["MAX_EP"]):
        sess.run(global_step_next)
        s0 = env.reset()
        track_r = []

        for j in range(HPs["MAX_EP_STEPS"]):

            action = net.GetAction(state=s0)
            s1,r,done,_ = env.step(action)
            if done: r = -20
            track_r.append(r)

            #Update Step
            net.Learn(s0,action,r,s1)


            s0 = s1
            if done or j >= HPs["MAX_EP_STEPS"]:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                print("episode:", sess.run(global_step), "  reward:", int(running_reward))
                break

        #Closing Functions that will be executed after every episode.
        if i % HPs["SAVE_FREQ"] == 0:
            saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))
            pass
        if i % HPs["LOG_FREQ"] == 0:
            tag = 'Training Results/'
            Record({
                tag+'Reward': running_reward,
                }, writer, i)
