import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import policy
import time

def use_this_policy():
    heur_policy_list = [policy.Patrol, policy.Roomba, policy.Defense, policy.Random, policy.AStar]
    heur_weight = [1,1,1,1,1]
    heur_weight = np.array(heur_weight) / sum(heur_weight)
    return np.random.choice(heur_policy_list, p=heur_weight)

def Starting(settings,envSettings,sess):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],
                                map_size=envSettings["MapSize"],
                                config_path=envSettings["ConfigPath"])
    envs = [make_env() for i in range(settings["NumberENV"])]
    envs = SubprocVecEnv(envs)
    N_F = envs.observation_space
    N_A = envs.action_space

    nTrajs = len(envs.get_team_blue().flatten())

    if envSettings["Centering"]:
        N_F = (39,39,6)

    return envs, list(N_F), 5, nTrajs

def Bootstrap(env,settings,envSettings,sess):
    s0 = env.reset( config_path=envSettings["EnvName"],
                    policy_red=use_this_policy())
    loggingDict = {"tracking_r":[[] for _ in range(len(env.get_team_blue().flatten()))],
                    "time_start":time.time()}
    return s0, loggingDict

def StateProcessing(s0,env,envSettings,sess):
    padder=[0,0,0,1,0,0,0]
    #Get list of controlled agents
    agents = env.get_team_blue()

    envs, olx, oly, ch = s0.shape
    H = olx*2-1
    W = oly*2-1
    padder = padder[:ch]
    #padder = [0] * ch; padder[3] = 1
    #if len(padder) >= 10: padder[7] = 1

    cx, cy = (W-1)//2, (H-1)//2
    states = np.zeros([len(agents[0])*envs, H, W, len(padder)])
    states[:,:,:] = np.array(padder)
    for idx, env in enumerate(agents):
        for idx2, agent in enumerate(env):
            x, y = agent.get_loc()
            states[idx*len(env)+idx2,max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0[idx]
    return states

def ActionProcessing(a,env,envSettings,sess):
    agents = env.get_team_blue()
    actions = a.reshape(agents.shape)
    return actions

def RewardShape(s1,r,done,env,envSettings,sess):
    done1 = np.ones([len(r),4])
    for idx,d in enumerate(done):
        done1[idx,:] = np.ones([4])*d
    reward = np.ones([len(r),4])
    done1 = done1.flatten()
    if 'was_done' not in globals():
        global was_done
        for idx,rew in enumerate(r):
            if done[idx]:
                reward[idx,:] = np.ones([4])*rew
        was_done = done1
    else:
        for idx,rew in enumerate(r):
            if not was_done[idx]:
                reward[idx,:] = np.ones([4])*rew
            else:
                reward[idx,:] = np.ones([4])*0
        was_done = done1

    reward = reward.flatten() * (1-np.array(was_done, dtype=int))

    return reward,done1

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    for i,envR in enumerate(r):
        if not done[i]: loggingDict["tracking_r"][i].append(envR)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting,progbar,sess):
    for i in range(settings["NumberENV"]):
        # print(loggingDict["tracking_r"][i])
        ep_rs_sum = sum(loggingDict["tracking_r"][i])
        if 'running_reward' not in globals():
            global running_reward
            running_reward = ep_rs_sum
        else:
            running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    time_elapsed = time.time()-loggingDict["time_start"]
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
    # print('Episode: {0:6d} Running Reward: {1:4.2f} Time Elapsed: {2:4.2f}s'.format(int(sess.run(global_step)[0]), running_reward, time_elapsed))
    progbar.update(sess.run(global_step)[0])
    finalDict = {   "Training Results/Reward":ep_rs_sum,
                    "Training Results/Episode Time":time_elapsed/settings["NumberENV"]}
    return finalDict
