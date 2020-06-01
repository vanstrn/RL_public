
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from random import randint

class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self,settings,superEpochs=None):
        self.superEpochs = superEpochs
        self.settings=settings
    def on_epoch_end(self, epoch, logs=None):
        if self.superEpochs is not None:
            return
        else:
            if epoch%self.settings["SaveFreq"] == 0:
                SF5.save_weights(MODEL_PATH+"model.h5")
    def on_train_end(self, logs=None):
        if self.superEpochs is None:
            SF5.save_weights(MODEL_PATH+"model.h5")
        else:
            if self.superEpochs%self.settings["SaveFreq"] == 0:
                SF5.save_weights(MODEL_PATH+"model.h5")

class SaveModel_Phi(tf.keras.callbacks.Callback):
    def __init__(self,settings):
        self.settings=settings
    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.settings["SaveFreq"] == 0:
            SF1.save_weights(MODEL_PATH+"model_phi_"+str(epoch)+".h5")

class SaveModel_Psi(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs,settings):
        self.superEpochs = superEpochs
        self.settings=settings
    def on_train_end(self, logs=None):
        if self.superEpochs%self.settings["SaveFreq"] == 0:
            SF2.save_weights(MODEL_PATH+"model_psi_"+str(self.superEpochs)+".h5")

class ValueTest_CTF(tf.keras.callbacks.Callback):
    def __init__(self,superEpochs,settings):
        self.superEpochs = superEpochs
        self.settings=settings
    def on_epoch_end(self,epoch, logs=None):
        if self.superEpochs%self.settings["LogFreq"] == 0:
            env.reset()
            rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = env.ConstructSample([i,j])
                if grid is None: continue
                [value] = SF4.predict(np.expand_dims(grid,0))
                rewardMap[i,j] = value
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Value Prediction")
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(LOG_PATH+"/ValuePred"+str(self.superEpochs)+".png")
            plt.close()

class ImageGenerator_CTF(tf.keras.callbacks.Callback):
    def __init__(self,settings):
        self.settings=settings
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.settings["LogFreq"] == 0:
            for i in range(5):
                samp = randint(0,len(s)-1)
                state = s[samp]
                act = action[samp]
                [state_new,reward] = SF1.predict([np.expand_dims(act,0),state])
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
                if i == 0:
                    plt.savefig(LOG_PATH+"/StatePredEpoch"+str(epoch)+".png")
                else:
                    plt.savefig(LOG_PATH+"/StatePred"+str(i)+".png")
                plt.close()


class RewardTest_CTF(tf.keras.callbacks.Callback):
    def __init__(self,settings):
        self.settings=settings
    def on_epoch_end(self,epoch, logs=None):
        if epoch%self.settings["LogFreq"] == 0:
            env.reset()
            rewardMap = np.zeros((dFeatures[0],dFeatures[1]))
            for i,j in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                grid = env.ConstructSample([i,j])
                if grid is None: continue
                [state_new,reward] = SF1.predict([np.stack([[0,0,0,0,0]]),np.expand_dims(grid,0)])
                rewardMap[i,j] = reward
            fig=plt.figure(figsize=(5.5, 8))
            fig.add_subplot(2,1,1)
            plt.title("State")
            imgplot = plt.imshow(env.get_obs_blue[:,:,2], vmin=-1, vmax=1)
            fig.add_subplot(2,1,2)
            plt.title("Reward Prediction Epoch "+str(epoch))
            imgplot = plt.imshow(rewardMap)
            fig.colorbar(imgplot)
            plt.savefig(LOG_PATH+"/RewardPred"+str(epoch)+".png")
            plt.close()
