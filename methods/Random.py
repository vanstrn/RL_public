
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
import numpy as np
import random


class Random(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initializes I/O placeholders and the training process of a Multi-step DQN.
        Main principal is that instead of one-step TD diference, the loss is evaluated on a
        temporally extended basis.
        G = R_t + γR_t+1 + ... γ^n-1 R_t+n + q(S_t+n,a*,θ-)
        loss = MSE(G,q(S_t,A_t,θ))

        """
        #Placeholders
        self.actionSize = actionSize
        self.sess = sess
        pass

    def GetAction(self, state,episode,step):
        """
        Contains the code to run the network based on an input.
        """
        actions = random.randint(0,self.actionSize-1)

        return actions ,[]  # return a int and extra data that needs to be fed to buffer.

    def Update(self,episode=0):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Checking that there is enough data for a batch
        pass


    def GetStatistics(self):
        return {}
    def AddToTrajectory(self,list):
        pass

    @property
    def getVars(self):
        return []
