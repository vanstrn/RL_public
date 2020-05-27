import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
from utils.utils import GetFunction
import numpy as np
from environments.MiniGridCustom import *
from utils.utils import MovingAverage
"""Common processing functions that can be implemented on a variety of different environments
Add Noise
Dropout
Normalize Reward
Flat addition to Reward to encourage game length or shorten game length if '-'
Logging Dictionary Starting and Ending.
"""

def CreateEnvironment(envSettings,multiprocessing=1):
    """A function that will create an environment based on a config file and wraps
    the environment in the specified wrappers. """

    #Creating the basic environment
    env = gym.make(envSettings["EnvName"], **envSettings["EnvParams"])
    if "Seed" in envSettings:
        env.seed(envSettings["Seed"])

    #Applying wrappers. See example wrappers below.
    env = TrajectoryWrapper(env,multiprocessing=multiprocessing)
    env = ApplyWrappers(env,envSettings["Wrappers"])
    numberFeatures = env.observation_space.shape
    numberActions = env.action_space.n
    nTrajs = env.nTrajs

    return env, list(numberFeatures), numberActions, nTrajs

def ApplyWrappers(env,wrapperList):
    for wrapperDict in wrapperList:
        wrapper = GetFunction(wrapperDict["WrapperName"])
        env = wrapper(env,**wrapperDict["WrapperParameters"])
    return env

class TrajectoryWrapper(gym.core.Wrapper):
    """
    Used to give environments a variable for the number of trajectories
    coming from the environment. Default in most environments is 1.
    """
    def __init__(self,env, multiprocessing,**kwargs):
        super().__init__(env)
        self.nTrajs = 1
        self.multiprocessing = multiprocessing

class FinalReward(gym.core.Wrapper):
    def __init__(self,env, finalReward=-20, **kwargs):
        super().__init__(env)
        self.finalReward = finalReward

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)
        return observation, self.reward(reward,done), done, info

    def reward(self, reward, done):
        if done: return reward + self.finalReward
        else: return reward

class RandomMovement(gym.core.Wrapper):

    def step(self, action):
        action = randint(0,self.action_space.n)
        return self.env.step(action=action)

class NPConverter(gym.core.Wrapper):
    def step(self, action):
        if isinstance(action,np.ndarray) or isinstance(action,list):
            if len(action) == 1:
                action = int(action)
        return self.env.step(action=action)

class RewardLogging(gym.core.Wrapper):
    def __init__(self,env, **kwargs):
        super().__init__(env)
        if self.multiprocessing == 1:
            self.GLOBAL_RUNNING_R = MovingAverage(400)
        else:
            if 'GLOBAL_RUNNING_R' not in globals():
                global GLOBAL_RUNNING_R
                GLOBAL_RUNNING_R = MovingAverage(400)
            self.GLOBAL_RUNNING_R = GLOBAL_RUNNING_R

    def reset(self, **kwargs):
        self.tracking_r = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)
        self.tracking_r.append(reward)
        return observation, reward, done, info

    def getLogging(self):
        """
        Processes the tracked data of the environment.
        In this case it sums the reward over the entire episode.
        """
        self.GLOBAL_RUNNING_R.append(sum(self.tracking_r))
        finalDict = {"TotalReward":self.GLOBAL_RUNNING_R()}
        return finalDict
