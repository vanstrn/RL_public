import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
"""Common processing functions that can be implemented on a variety of different environments
Add Noise
Dropout
Normalize Reward
Flat addition to Reward to encourage game length or shorten game length if '-'
Logging Dictionary Starting and Ending.
"""
def RewardShape(s1,r,done,env,envSettings,sess):
    for idx in range(len(done)):
        if done[idx]: r[idx] = -20
    return r

def Bootstrap(env,settings,envSettings,sess):
    s0 = env.reset()
    loggingDict = {"tracking_r":[[] for _ in range(settings["NumberENV"])]}
    return s0, loggingDict

def Starting(settings,envSettings,sess):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],)
    envs = [make_env() for i in range(settings["NumberENV"])]
    envs = SubprocVecEnv(envs)
    envs.remotes[0].send(('_get_spaces', None))
    N_F, N_A = envs.remotes[0].recv()
    nTrajs = settings["NumberENV"]

    return envs, N_F[0], N_A,nTrajs

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    for i,envR in enumerate(r):
        if not done[i]: loggingDict["tracking_r"][i].append(envR)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting,sess):
    for i in range(settings["NumberENV"]):
        # print(loggingDict["tracking_r"][i])
        ep_rs_sum = sum(loggingDict["tracking_r"][i])

        if 'running_reward' not in globals():
            global running_reward
            running_reward = ep_rs_sum
        else:
            running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
    print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))

    finalDict = {"Training Results/Reward":ep_rs_sum}
    return finalDict
