import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import policy
import time

def use_this_policy(policyName=None):
    if policyName is None:
        heur_policy_list = [policy.Patrol(), policy.Roomba(), policy.Defense(), policy.Random(), policy.AStar()]
        heur_weight = [1,1,1,1,1]
        heur_weight = np.array(heur_weight) / sum(heur_weight)
        return np.random.choice(heur_policy_list, p=heur_weight)
    elif policyName == "Roomba":
        return policy.Roomba()
    elif policyName == "Patrol":
        return policy.Patrol()
    elif policyName == "Defense":
        return policy.Defense()
    elif policyName == "AStar":
        return policy.AStar()
    elif policyName == "Random":
        return policy.Random()


def Starting(settings,envSettings,sess):
    env = gym.make(envSettings["EnvName"],
                            map_size=envSettings["MapSize"],
                            config_path=envSettings["ConfigPath"])
    N_F = env.observation_space.shape
    N_A = env.action_space

    nTrajs = len(env.get_team_blue.flatten())

    if envSettings["Centering"]:
        print(N_F)
        N_F = (N_F[0]*2-1,N_F[1]*2-1,N_F[2])

    return env, list(N_F), 5, nTrajs

def Bootstrap(env,settings,envSettings,sess):
    s0 = env.reset( config_path=envSettings["EnvName"],
                    policy_red=use_this_policy(envSettings["Policy"]))
    loggingDict = {"tracking_r":[[] for _ in range(len(env.get_team_blue.flatten()))],
                    "time_start":time.time()}
    return s0, loggingDict

def StateProcessing(s0,env,envSettings,sess):
    if envSettings["Centering"]:
        padder=[0,0,0,1,0,0,0]
        #Get list of controlled agents
        agents = env.get_team_blue

        olx, oly, ch = s0.shape
        H = olx*2-1
        W = oly*2-1
        padder = padder[:ch]
        #padder = [0] * ch; padder[3] = 1
        #if len(padder) >= 10: padder[7] = 1

        cx, cy = (W-1)//2, (H-1)//2
        states = np.zeros([len(agents), H, W, len(padder)])
        states[:,:,:] = np.array(padder)
        for idx, agent in enumerate(agents):
            x, y = agent.get_loc()
            states[idx,max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0
    else:
        olx, oly, ch = s0.shape
        H = olx
        W = oly
        states = np.zeros([len(agents), H, W, len(padder)])
        for idx, agent in enumerate(agents):
                states[idx,:,:,:] = s0[idx]

    return states

def ActionProcessing(a,env,envSettings,sess):
    agents = env.get_team_blue
    actions = a.reshape(agents.shape)
    return actions

def RewardShape(s1,reward_raw,done_raw,env,envSettings,sess):
    #Processing the Dones
    agents = env.get_team_blue
    done = np.ones([len(agents)])*done_raw

    #Processing the reward recording zero if the environment is done.
    if 'was_done' not in globals():
        global was_done
        was_done = done
    else:
        was_done = done
    if done.all():
        was_done = np.ones([len(agents)])*False


    if not done_raw:
        reward=np.ones([len(agents)])*reward_raw
    else:
        reward=np.ones([len(agents)])*0
    #     for idx,rew in enumerate(r):
    #         if done[idx]:
    #             reward[idx,:] = np.ones([4])*rew
    # else:
    #     for idx,rew in enumerate(r):
    #         if not was_done[idx]:
    #             reward[idx,:] = np.ones([4])*rew
    #         else:
    #             reward[idx,:] = np.ones([4])*0
    #     was_done = done1
    # for rew in r:
    #     if rew >= 0:
    #         print("Won the game")

    reward = reward.flatten() #* (1-np.array(was_done, dtype=int))

    return reward,done

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    for i,envR in enumerate(r):
        if not done[i]: loggingDict["tracking_r"][i].append(envR)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting,sess,progbar,GLOBAL_RUNNING_R=None,GLOBAL_EP_LEN=None):
    if GLOBAL_RUNNING_R is not None:
        for i in range(len(loggingDict["tracking_r"])):
            try:
                GLOBAL_RUNNING_R.append(loggingDict["tracking_r"][i][-1])
            except IndexError:
                pass
        finalDict = {"Training Results/Reward":GLOBAL_RUNNING_R()}
        return finalDict
    else:
        for i in range(settings["NumberENV"]):
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
