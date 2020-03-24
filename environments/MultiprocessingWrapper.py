#To do:
# - Actual Multiprocessing instead of sequential.
# - A way to deal with if an environment finishes before others.
# - Test the performance of the environment.
# Add a thing to modify the number of trajectories. (ex ctf would have 4*N trajectories)


import numpy as np
import random
from multiprocessing import Process, Pipe
from copy import deepcopy

class SimpleMultiEnvWrapper:
    """
    Asynchronous Environment Vectorized run
    """
    def __init__(self, env, numberEnvs=2):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.envList = []
        for i in range(numberEnvs):
            self.envList.append(deepcopy(env))

        self.numberEnvs = numberEnvs
        self.nTrajs = numberEnvs

    def step(self, actions=None):
        """
        Assumes that the environments have the same number of 'trajectories(agents)'
        """
        if actions is None: actions = [None]*self.numberEnvs
        splitActions = np.array_split(out,self.numberEnvs)
        obsList = []
        rewList = []
        doneList = []
        infoList = []
        for i,action in enumerate(splitActions):
            obs, reward, done, info =  self.envList[i].step(action)
            obsList.append(obs)
            rewList.append(reward)
            doneList.append(done)
            infoList.append(info)


        return np.vstack(obsList), np.vstack(rewList), np.vstack(doneList), np.vstack(infoList)

    def reset(self, **kwargs):
        for i in range(self.numberEnvs):
            s = self.envList[i].reset(**kwargs)
        return

    def __len__(self):
        return self.nenvs
