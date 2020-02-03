import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np

def RewardShape(s1,reward_raw,done_raw,env,envSettings):
    return reward_raw, np.array(done_raw)

def Bootstrap(env,settings,envSettings):
    s0 = env.reset()
    loggingDict = {"tracking_r":[[] for _ in range(settings["NumberENV"])]}
    return s0, loggingDict

def StateProcessing(s0,env,envSettings):
    return s0["image"]

def Starting(settings,envSettings):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],)
    envs = [make_env() for i in range(settings["NumberENV"])]
    envs = SubprocVecEnv(envs)
    envs.remotes[0].send(('_get_spaces', None))
    N_F, N_A = envs.remotes[0].recv()
    nTrajs = settings["NumberENV"]

    return envs, N_F[0], N_A,nTrajs

def StartingSingle(settings,envSettings):
    env = gym.make(envSettings["EnvName"])
    numberFeatures = env.observation_space["image"].shape
    numberActions = env.action_space.n
    numberActions=3

    return env, list(numberFeatures), numberActions, 1

def Logging(loggingDict,s1,r,done,env,envSettings):
    try:
        for i,envR in enumerate(r):
            if not done[i]: loggingDict["tracking_r"][i].append(envR)
    except: loggingDict["tracking_r"][0].append(r)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting):
    for i in range(settings["NumberENV"]):
        # print(loggingDict["tracking_r"][i])
        ep_rs_sum = sum(loggingDict["tracking_r"][i])

        if 'running_reward' not in globals():
            global running_reward
            running_reward = ep_rs_sum
        else:
            running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    # global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
    # print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))

    finalDict = {"Training Results/Reward":ep_rs_sum}
    return finalDict