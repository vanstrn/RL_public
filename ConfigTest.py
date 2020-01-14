"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import tensorflow as tf

from networks.network import Network
from methods.A2C import A2C
from utils.utils import InitializeVariables, CreatePath
from utils.record import Record,SaveHyperparams
import json

if __name__ == "__main__":
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

    global_episodes = 0
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    network = Network("configs/network/"+settings["NetworkConfig"],N_A)

    net = A2C(network,sess,stateShape=[None,N_F],actionSize=N_A,HPs=settings["NetworkHPs"])

    #Creating Auxilary Functions for logging and saving.
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
    net.InitializeVariablesFromFile(saver,MODEL_PATH)
    InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

    total_step = 1
    #Running the Simulation
    for i in range(settings["EnvHPs"]["MAX_EP"]):
        sess.run(global_step_next)
        s0 = env.reset()
        track_r = []

        for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]):

            a, networkData = net.GetAction(state=s0)
            s1,r,done,_ = env.step(a)
            if done: r = -20
            track_r.append(r)

            #Update Step
            net.AddToBuffer([s0,a,r,s1,done]+networkData)

            if total_step % settings["EnvHPs"]['UPDATE_GLOBAL_ITER'] == 0 or done:   # update global and assign to local net
                net.Update(settings["NetworkHPs"])

            s0 = s1
            total_step += 1
            if done or j >= settings["EnvHPs"]["MAX_EP_STEPS"]:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))
                break

        #Closing Functions that will be executed after every episode.
        if i % settings["EnvHPs"]["SAVE_FREQ"] == 0:
            saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))
            pass
        if i % settings["EnvHPs"]["LOG_FREQ"] == 0:
            tag = 'Training Results/'
            Record({
                tag+'Reward': running_reward,
                }, writer, i)
