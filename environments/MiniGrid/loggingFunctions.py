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


class ValuePredictionEvaluation(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs,env,network,imageDir=None,freq=100):
        self.env = env
        self.network=network[3]
        self.imageDir=imageDir
        self.freq = freq
        self.superEpochs = superEpochs

    def on_train_end(self, logs=None):
        if self.superEpochs%self.freq == 0:
            self.env.reset()
            rewardMap = np.zeros([self.env.width,self.env.height])
            for i,j in itertools.product(range(self.env.width),range(self.env.height)):
                grid = ConstructSampleMG4R(self.env,[i,j])
                if grid is None: continue
                value = self.network.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 5.5))
            fig.add_subplot(1,1,1)
            plt.title("Value Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/ValuePred"+str(epoch)+".png")
            plt.close()


class StatePredictionEvaluation(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=100):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq

    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            state = self.env.reset()
            state_new,reward = self.network.predict(state)
            fig=plt.figure(figsize=(5.5, 5.5))
            fig.add_subplot(1,1,1)
            plt.title("Predicted Next State Epoch "+str(epoch))
            imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
            plt.savefig(self.imageDir+"/StatePredEpoch"+str(epoch)+".png")
            plt.close()

class StatePredictionEvaluation_action(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=100):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq

    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq  == 0:
            state = self.env.reset()
            fig=plt.figure(figsize=(17, 5.5))
            fig.add_subplot(1,5,1)
            plt.title("State Epoch "+str(epoch))
            imgplot = plt.imshow(state[:,:,0],vmin=0, vmax=10)
            for i in range(4):
                act = np.zeros([1,4])
                act[0,i] = 1
                state_new,reward = self.network.predict(state,action=act)
                fig.add_subplot(1,5,i+2)
                plt.title("Predicted Next State Epoch "+str(epoch))
                imgplot = plt.imshow(state_new[0,:,:,0],vmin=0, vmax=10)
            plt.savefig(self.imageDir+"/StatePredEpoch"+str(epoch)+".png")
            plt.close()


class RewardPredictionEvaluation(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=100):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq

    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq  == 0:
            self.env.reset()
            rewardMap = np.zeros([self.env.width,self.env.height])
            for i,j in itertools.product(range(self.env.width),range(self.env.height)):
                grid = ConstructSampleMG4R(self.env,[i,j])
                if grid is None: continue
                state_new,reward = self.network.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 5.5))
            fig.add_subplot(1,1,1)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/RewardPred"+str(epoch)+".png")
            plt.close()
