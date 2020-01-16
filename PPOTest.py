"""
Modular Framework for setting up singular environment sampling
based on a runtime config file.
"""

import numpy as np
import gym
import tensorflow as tf

from networks.network import Network
from utils.utils import InitializeVariables, CreatePath
from utils.record import Record,SaveHyperparams
import json
from importlib import import_module #Used to import module based on a string.

def GetFunction(string):
    module_name, func_name = string.rsplit('.',1)
    module = import_module(module_name)
    func = getattr(module,func_name)
    return func

if __name__ == "__main__":
    #Defining parameters and Hyperparameters for the run.
    with open("configs/run/ppoRun.json") as json_file:
        settings = json.load(json_file)
    with open("configs/environment/"+settings["EnvConfig"]) as json_file:
        envSettings = json.load(json_file)

    EXP_NAME = settings["RunName"]
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME
    CreatePath(LOG_PATH)
    CreatePath(MODEL_PATH)

    #Creating the Environment
    sess = tf.Session()

    for functionString in envSettings["StartingFunctions"]:
        StartingFunction = GetFunction(functionString)
        env,N_F,N_A = StartingFunction(envSettings,sess)

    global_episodes = 0
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    network = Network("configs/network/"+settings["NetworkConfig"],N_A)

    Method = GetFunction(settings["Method"])
    net = Method(network,sess,stateShape=[N_F],actionSize=N_A,HPs=settings["NetworkHPs"])

    #Creating Auxilary Functions for logging and saving.
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=3, var_list=net.getVars+[global_step])
    net.InitializeVariablesFromFile(saver,MODEL_PATH)
    InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

    total_step = 1
    #Running the Simulation
    for i in range(settings["EnvHPs"]["MAX_EP"]):

        sess.run(global_step_next)
        track_r = []

        for functionString in envSettings["BootstrapFunctions"]:
            BootstrapFunctions = GetFunction(functionString)
            s0, loggingDict = BootstrapFunctions(env,envSettings,sess)

        for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]):

            for functionString in envSettings["StateProcessingFunctions"]:
                StateProcessing = GetFunction(functionString)
                s0 = StateProcessing(s0,envSettings,sess)

            a, networkData = net.GetAction(state=s0)

            for functionString in envSettings["ActionProcessingFunctions"]:
                ActionProcessing = GetFunction(functionString)
                r = ActionProcessing(a,env,sess)

            s1,r,done,_ = env.step(a)

            for functionString in envSettings["RewardProcessingFunctions"]:
                RewardProcessing = GetFunction(functionString)
                r = RewardProcessing(s1,r,done,env,envSettings,sess)

            #Update Step
            net.AddToBuffer([s0,a,r,s1,done]+networkData)

            if total_step % settings["EnvHPs"]['UPDATE_GLOBAL_ITER'] == 0 or done:   # update global and assign to local net
                net.Update(settings["NetworkHPs"])

            for functionString in envSettings["LoggingFunctions"]:
                LoggingFunctions = GetFunction(functionString)
                loggingDict = LoggingFunctions(loggingDict,s1,r,done,env,envSettings,sess)

            s0 = s1
            total_step += 1
            if done or j >= settings["EnvHPs"]["MAX_EP_STEPS"]:
                break

        #Closing Functions that will be executed after every episode.
        for functionString in envSettings["EpisodeClosingFunctions"]:
            EpisodeClosingFunction = GetFunction(functionString)
            finalDict = EpisodeClosingFunction(loggingDict,env,envSettings,sess)

        if i % settings["EnvHPs"]["SAVE_FREQ"] == 0:
            saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))

        if i % settings["EnvHPs"]["LOG_FREQ"] == 0:
            Record(finalDict, writer, i)
