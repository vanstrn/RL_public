import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ConstructSampleMG4R(env,position):
    grid = env.grid.encode()
    if grid[position[0],position[1],1] == 5:
        return None
    grid[position[0],position[1],0] = 10
    return grid[:,:,:2]

def ConstructSampleMG4RP(env,position):
    cell = env.grid.get(*position)
    if cell.type in ["goal",'lava']:
        return None

    env.agent_pos = position
    return env.render(mode = "nah, No render")


class ValuePredictionEvaluation():
    def __init__(self,env,network,imageDir=None):
        self.env = env
        self.network=network
        self.imageDir=imageDir

    def __call__(self,step):
        self.env.reset()
        rewardMap = np.zeros([self.env.width,self.env.height])
        for i,j in itertools.product(range(self.env.width),range(self.env.height)):
            grid = ConstructSampleMG4R(self.env,[i,j])
            if grid is None: continue
            value = self.network.PredictValue(np.expand_dims(grid,0))
            rewardMap[i,j] = value
        fig=plt.figure(figsize=(5.5, 5.5))
        fig.add_subplot(1,1,1)
        plt.title("Value Prediction Epoch "+str(step))
        imgplot = plt.imshow(rewardMap)
        fig.colorbar(imgplot)
        plt.savefig(self.imageDir+"/ValuePred"+str(step)+".png")
        plt.close()


class StatePredictionEvaluation():
    def __init__(self,env,network,imageDir=None):
        self.env = env
        self.network=network
        self.imageDir = imageDir

    def __call__(self,step):
        state = self.env.reset()
        state_new = self.network.PredictState(np.expand_dims(state,0))
        fig=plt.figure(figsize=(5.5, 5.5))
        fig.add_subplot(1,1,1)
        plt.title("Predicted Next State Epoch "+str(step))
        imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
        plt.savefig(self.imageDir+"/StatePredEpoch"+str(step)+".png")
        plt.close()


class RewardPredictionEvaluation():
    def __init__(self,env,network,imageDir=None):
        self.env = env
        self.network=network
        self.imageDir = imageDir

    def __call__(self,step):
            self.env.reset()
            rewardMap = np.zeros([self.env.width,self.env.height])
            for i,j in itertools.product(range(self.env.width),range(self.env.height)):
                grid = ConstructSampleMG4R(self.env,[i,j])
                if grid is None: continue
                reward = self.network.PredictReward(np.expand_dims(grid,0))
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 5.5))
            fig.add_subplot(1,1,1)
            plt.title("Reward Prediction Epoch "+str(step))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/RewardPred"+str(step)+".png")
            plt.close()
