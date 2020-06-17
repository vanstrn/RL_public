import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

class ValueTest_StackedStates(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[3]
        self.imageDir=imageDir
        self.freq = freq
        self.superEpochs = superEpochs
    def on_epoch_end(self,epoch, logs=None):
        if self.superEpochs%self.freq == 0:
            s=self.env.reset()
            rewardMap = np.zeros((s.shape[1],s.shape[2]))
            for i,j in itertools.product(range(s.shape[1]),range(s.shape[2])):
                grid = self.env.ConstructSample([i,j])
                if grid is None: continue
                tmp = np.concatenate([grid] * 4, axis=2)
                [value] = SF4.predict(np.expand_dims(tmp,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(self.env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Value Prediction")
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/ValuePred"+str(self.superEpochs)+".png")
            plt.close()

class ValueTest(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[3]
        self.imageDir=imageDir
        self.freq = freq
        self.superEpochs = superEpochs

    def on_epoch_end(self,epoch, logs=None):
        if self.superEpochs%self.freq == 0:
            s=self.env.reset()
            rewardMap = np.zeros((s.shape[1],s.shape[2]))
            for i,j in itertools.product(range(s.shape[1]),range(s.shape[2])):
                grid = self.env.ConstructSample([i,j])
                if grid is None: continue
                [value] = self.network.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(self.env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Value Prediction")
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/ValuePred"+str(self.superEpochs)+".png")
            plt.close()

class ImageGenerator_StackedStates(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            state = self.env.reset()
            [state_new,reward] = self.network.predict([np.stack([[0,0,0,0,0]]),state])
            fig=plt.figure(figsize=(16, 8))
            fig.add_subplot(2,4,1)
            plt.title("State Territory")
            imgplot = plt.imshow(state[0,:,:,1], vmin=-2, vmax=2)
            fig.add_subplot(2,4,2)
            plt.title("State Flags")
            imgplot = plt.imshow(state[0,:,:,2], vmin=-10, vmax=10)
            fig.add_subplot(2,4,3)
            plt.title("State Obstacles")
            imgplot = plt.imshow(state[0,:,:,3], vmin=-2, vmax=2)
            fig.add_subplot(2,4,4)
            plt.title("State Agents")
            imgplot = plt.imshow(state[0,:,:,4],vmin=-10, vmax=10)
            fig.add_subplot(2,4,5)
            plt.title("Predicted Next State Territory")
            imgplot = plt.imshow(state_new[0,:,:,1],vmin=-2, vmax=2)
            fig.add_subplot(2,4,6)
            plt.title("Predicted Next State Flags")
            imgplot = plt.imshow(state_new[0,:,:,2],vmin=-10, vmax=10)
            fig.add_subplot(2,4,7)
            plt.title("Predicted Next State Obstacles")
            imgplot = plt.imshow(state_new[0,:,:,3],vmin=-2, vmax=2)
            fig.add_subplot(2,4,8)
            plt.title("Predicted Next State Agents")
            imgplot = plt.imshow(state_new[0,:,:,4],vmin=-10, vmax=10)
            plt.savefig(self.imageDir+"/StatePredEpoch"+str(epoch)+".png")

class ImageGenerator_actions(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            state = self.env.reset()
            [state_new,reward] = self.network.predict([np.stack([[0,0,0,0,0]]),state])
            fig=plt.figure(figsize=(16, 8))
            fig.add_subplot(2,4,1)
            plt.title("State Territory")
            imgplot = plt.imshow(state[0,:,:,1], vmin=-2, vmax=2)
            fig.add_subplot(2,4,2)
            plt.title("State Flags")
            imgplot = plt.imshow(state[0,:,:,2], vmin=-10, vmax=10)
            fig.add_subplot(2,4,3)
            plt.title("State Obstacles")
            imgplot = plt.imshow(state[0,:,:,3], vmin=-2, vmax=2)
            fig.add_subplot(2,4,4)
            plt.title("State Agents")
            imgplot = plt.imshow(state[0,:,:,4],vmin=-10, vmax=10)
            fig.add_subplot(2,4,5)
            plt.title("Predicted Next State Territory")
            imgplot = plt.imshow(state_new[0,:,:,1],vmin=-2, vmax=2)
            fig.add_subplot(2,4,6)
            plt.title("Predicted Next State Flags")
            imgplot = plt.imshow(state_new[0,:,:,2],vmin=-10, vmax=10)
            fig.add_subplot(2,4,7)
            plt.title("Predicted Next State Obstacles")
            imgplot = plt.imshow(state_new[0,:,:,3],vmin=-2, vmax=2)
            fig.add_subplot(2,4,8)
            plt.title("Predicted Next State Agents")
            imgplot = plt.imshow(state_new[0,:,:,4],vmin=-10, vmax=10)
            plt.savefig(self.imageDir+"/StatePredEpoch"+str(epoch)+".png")
            plt.close()

class ImageGenerator(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            state = self.env.reset()
            [state_new,reward] = self.network.predict([state])
            fig=plt.figure(figsize=(16, 8))
            fig.add_subplot(2,4,1)
            plt.title("State Territory")
            imgplot = plt.imshow(state[0,:,:,1], vmin=-2, vmax=2)
            fig.add_subplot(2,4,2)
            plt.title("State Flags")
            imgplot = plt.imshow(state[0,:,:,2], vmin=-10, vmax=10)
            fig.add_subplot(2,4,3)
            plt.title("State Obstacles")
            imgplot = plt.imshow(state[0,:,:,3], vmin=-2, vmax=2)
            fig.add_subplot(2,4,4)
            plt.title("State Agents")
            imgplot = plt.imshow(state[0,:,:,4],vmin=-10, vmax=10)
            fig.add_subplot(2,4,5)
            plt.title("Predicted Next State Territory")
            imgplot = plt.imshow(state_new[0,:,:,1],vmin=-2, vmax=2)
            fig.add_subplot(2,4,6)
            plt.title("Predicted Next State Flags")
            imgplot = plt.imshow(state_new[0,:,:,2],vmin=-10, vmax=10)
            fig.add_subplot(2,4,7)
            plt.title("Predicted Next State Obstacles")
            imgplot = plt.imshow(state_new[0,:,:,3],vmin=-2, vmax=2)
            fig.add_subplot(2,4,8)
            plt.title("Predicted Next State Agents")
            imgplot = plt.imshow(state_new[0,:,:,4],vmin=-10, vmax=10)
            plt.savefig(self.imageDir+"/StatePredEpoch"+str(epoch)+".png")
            plt.close()


class RewardTest(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq

    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            s = self.env.reset()
            rewardMap = np.zeros((s.shape[1],s.shape[2]))
            for i,j in itertools.product(range(s.shape[1]),range(s.shape[2])):
                grid = self.env.ConstructSample([i,j])
                if grid is None: continue
                [state_new,reward] = self.network.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(self.env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/RewardPred"+str(epoch)+".png")
            plt.close()

class RewardTest_actions(tf.keras.callbacks.Callback):
    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq

    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            s = self.env.reset()
            rewardMap = np.zeros((s.shape[1],s.shape[2]))
            for i,j in itertools.product(range(s.shape[1]),range(s.shape[2])):
                grid = self.env.ConstructSample([i,j])
                if grid is None: continue
                [state_new,reward] = self.network.predict([np.stack([[0,0,0,0,0]]),np.expand_dims(grid,0)])
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(self.env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/RewardPred"+str(epoch)+".png")
            plt.close()

class RewardTest_StackedStates(tf.keras.callbacks.Callback):

    def __init__(self,env,network,imageDir=None,freq=50):
        self.env = env
        self.network=network[0]
        self.imageDir = imageDir
        self.freq = freq
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.freq == 0:
            s = self.env.reset()
            rewardMap = np.zeros((s.shape[1],s.shape[2]))
            for i,j in itertools.product(range(s.shape[1]),range(s.shape[2])):
                grid = self.env.ConstructSample([i,j])
                if grid is None: continue
                tmp = np.concatenate([grid] * 4, axis=2)
                [state_new,reward] = self.network.predict([np.stack([[1,1,1,1,1]]),np.expand_dims(tmp,0)])
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(self.env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(self.imageDir+"/RewardPred"+str(epoch)+".png")
            plt.close()
