import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import gym_cap.heuristic as policy
import time
from gym import spaces
from random import randint


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



class CTFCentering(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,centering):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.centering = centering
        self.action_space = spaces.Discrete(len(self.ACTION))
        if centering:

            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(4, 39, 39, 6),  # number of cells
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(4, 20, 20, 6),  # number of cells
                dtype='uint8'
            )

    def observation(self, s0):
        padder=[0,0,0,1,0,0]
        #Get list of controlled agents
        agents = self.env.get_team_blue
        if self.centering:

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
                states[idx,:,:,:] = s0

        return states


class Reset(gym.core.Wrapper):
    def __init__(self,env, config_path=None, policy_red=None, **kwargs):
        super().__init__(env)
        self.config_path = config_path
        self.policy_red = policy_red
    def reset(self, **kwargs):
        action = randint(0,self.action_space.n)
        return self.env.reset(config_path=self.config_path, policy_red=use_this_policy(self.policy_red),**kwargs)


class RewardShape(gym.core.RewardWrapper):
    def __init__(self,env, finalReward=-20, **kwargs):
        super().__init__(env)
        self.was_done=None

    def reset(self, **kwargs):
        self.was_done=None
        return self.env.reset(**kwargs)

    def reward(self, reward_raw):

        reward = np.ones([4])*reward_raw

        reward = reward.flatten() #* (1-np.array(self.was_done, dtype=int))
        return reward
