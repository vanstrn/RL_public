import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import gym_cap.heuristic as policy
import time
from gym import spaces
from random import randint
import random
import os


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

class UseMap(gym.core.Wrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,biased=False,execute=True,mapPath="/home/capturetheflag/RL/environments/fair_map_1v0"):
        super().__init__(env)
        self.first=True
        self.execute=execute
        self.biased=biased
        self.episodes=0
        self.map_list = [os.path.join(mapPath, path) for path in os.listdir(mapPath)]
        max_epsilon = 0.70;

    def use_this_map(self,x):
        if self.first:
            self.first=False
            return random.choice(self.map_list)
        if self.biased:
            return random.choice(self.map_list)

        def smoothstep(x, lowx=0.0, highx=1.0, lowy=0, highy=1):
            x = (x-lowx) / (highx-lowx)
            if x < 0:
                val = 0
            elif x > 1:
                val = 1
            else:
                val = x * x * (3 - 2 * x)
                return val*(highy-lowy)+lowy
        prob = smoothstep(x, highx=50000, highy=0.70)
        if np.random.random() < prob:
            return random.choice(self.map_list)
        else:
            return None
    def reset(self,**kwargs):
        if not self.execute:
            return self.env.reset(**kwargs)

        self.episodes+=1
        tmp=self.use_this_map(self.episodes)
        if tmp is not None:
            self.map = tmp
        return self.env.reset(custom_board=self.map,**kwargs)


class CTFCentering(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,centering):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.centering = centering
        self.map_size
        self.action_space = spaces.Discrete(len(self.ACTION))
        if centering:

            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=( 2*self.map_size[0]-1, 2*self.map_size[1]-1, 6),  # number of cells
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=( self.map_size[0], self.map_size[1], 6),  # number of cells
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
class StateStacking(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,nStates,axis=3):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.nStates = nStates
        self.axis = axis

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=( self.map_size[0], self.map_size[1], 6*nStates),  # number of cells
            dtype='uint8'
        )

    def reset(self, **kwargs):
        s0 = self.env.reset(config_path=self.config_path, policy_red=use_this_policy(self.policy_red),**kwargs)
        self.stack = [s0] * self.nStates
        self.observation()
        return self.observation()
    def observation(self, s0):
        if s0 is None:
            return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(s0)
        self.stack.pop(0)
        return np.concatenate(self.stack, axis=self.axis)



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
        self.nAgents = len(self.get_team_blue)

    def reset(self, **kwargs):
        self.was_done=None
        return self.env.reset(**kwargs)

    def reward(self, reward_raw):
        if reward_raw == -0.001:
            reward_raw =0.0
        reward = np.ones([self.nAgents])*reward_raw

        reward = reward.flatten() #* (1-np.array(self.was_done, dtype=int))
        return reward
