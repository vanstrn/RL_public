import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np

def RewardShape(s1,reward_raw,done_raw,env,envSettings,sess):
    # if not done_raw: reward_raw += -0.01
    # else: reward_raw += 0.5
    return reward_raw, np.array(done_raw)

def Bootstrap(env,settings,envSettings,sess):
    s0 = env.reset()
    loggingDict = {"tracking_r":[[] for _ in range(settings["NumberENV"])]}
    return s0, loggingDict

def StateProcessing(s0,env,envSettings,sess):
    return s0["image"]

def Starting(settings,envSettings,sess):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],)
    envs = [make_env() for i in range(settings["NumberENV"])]
    envs = SubprocVecEnv(envs)
    envs.remotes[0].send(('_get_spaces', None))
    N_F, N_A = envs.remotes[0].recv()
    nTrajs = settings["NumberENV"]

    return envs, N_F[0], N_A,nTrajs

def StartingSingle(settings,envSettings,sess):
    env = gym.make(envSettings["EnvName"])
    numberFeatures = env.observation_space["image"].shape
    numberActions = env.action_space.n
    numberActions=3

    return env, list(numberFeatures), numberActions, 1

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    try:
        for i,envR in enumerate(r):
            if not done[i]: loggingDict["tracking_r"][i].append(envR)
    except: loggingDict["tracking_r"][0].append(r)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting,sess,progbar,GLOBAL_RUNNING_R=None):
    if GLOBAL_RUNNING_R is not None:
        for i in range(settings["NumberENV"]):
            GLOBAL_RUNNING_R.append(sum(loggingDict["tracking_r"][i]))
        finalDict = {"Training Results/Reward":GLOBAL_RUNNING_R()}
        return finalDict


    else:
        for i in range(settings["NumberENV"]):
            # print(loggingDict["tracking_r"][i])
            ep_rs_sum = sum(loggingDict["tracking_r"][i])

            if 'running_reward' not in globals():
                global running_reward
                running_reward = ep_rs_sum
                # progbar.add("Reward")
            running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
        progbar.update(sess.run(global_step)[0],values=[("Reward",running_reward)])
        # print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))

        finalDict = {"Training Results/Reward":running_reward}
        return finalDict
