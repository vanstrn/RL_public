import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv

def RewardShape(s1,r,done,env,envSettings,sess):
    if done: r = -20
    return r

def Bootstrap(env,envSettings,sess):
    s0 = env.reset()
    loggingDict = {"tracking_r":[]}
    return s0, loggingDict

def Starting(settings,envSettings,sess):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],)
    envs = [make_env() for i in range(settings["NumberENV"])]
    print(envs[0])
    N_F = envs[0].observation_space.shape[0]
    N_A = envs[0].action_space.n

    envs = SubprocVecEnv(envs)
    # env = gym.make(envSettings["EnvName"])
    env.seed(envSettings["Seed"])  # Create a consistent seed so results are reproducible.
    # env = env.unwrapped
    nTrajs = settings["NumberENV"]

    return env, N_F, N_A,nTrajs

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    loggingDict["tracking_r"].append(r)
    return loggingDict

def Closing(loggingDict,env,envSetting,sess):
    ep_rs_sum = sum(loggingDict["tracking_r"])

    if 'running_reward' not in globals():
        global running_reward
        running_reward = ep_rs_sum
    else:
        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
    print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))

    finalDict = {"Training Results/Reward":ep_rs_sum}
    return finalDict
