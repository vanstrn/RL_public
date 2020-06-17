
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
from .AdvantageEstimator import gae
import tensorflow as tf
import numpy as np
import scipy
from utils.record import Record
from utils.utils import MovingAverage,CreatePath, GetFunction
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

from networks.network_v3 import buildNetwork
from networks.common import NetworkBuilder
from environments.Common import CreateEnvironment
from random import randint
import itertools
class SF_QMap_CTF_1v1(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initializes a training method for a neural network.

        Parameters
        ----------
        Model : Keras Model Object
            A Keras model object with fully defined layers and a call function. See examples in networks module.
        sess : Tensorflow Session
            Initialized Tensorflow session
        stateShape : list
            List of integers of the inputs shape size. Ex [39,39,6]
        actionSize : int
            Output size of the network.
        HPs : dict
            Dictionary that contains all hyperparameters to be used in the methods training
        nTrajs : int (Optional)
            Number that specifies the number of trajectories to be created for collecting training data.
        scope : str (Optional)
            Name of the PPO method. Used to group and differentiate variables between other networks.

        Returns
        -------
        N/A
        """
        EXP_NAME = settings["RunName"]
        LoadName = settings["LoadName"]
        MODEL_PATH = './models/'+LoadName+ '/'
        IMAGE_PATH = './images/SF/'+EXP_NAME+'/'
        MODEL_PATH_ = './models/'+EXP_NAME+'/'
        LOG_PATH = './logs/CTF_1v1/'+EXP_NAME
        CreatePath(LOG_PATH)
        CreatePath(IMAGE_PATH)
        CreatePath(MODEL_PATH)
        CreatePath(MODEL_PATH_)
        self.sess=sess

        N = settings["NumOptions"]
        with open("configs/environment/"+settings["EnvConfig"]) as json_file:
            envSettings = json.load(json_file)
        env,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings)
        #Create the Q Maps

        if "LoadQMaps" in settings:
            #Loading the Q-tables for the sub-policies
            loadedData = np.load('./models/'+settings["LoadQMaps"]+ '/options.npz')
            opt = loadedData["options"]
            options=[]
            for i in range(opt.shape[0]):
                options.append(opt[i,:,:,:,:])
        else:
            if "LoadSamples" in settings:
                pass
            else:
                #Creating Instance of environment and running through it to generate samples
                def GetAction(state):
                    """
                    Contains the code to run the network based on an input.
                    """
                    p = 1/nActions
                    if len(state.shape)==3:
                        probs =np.full((1,nActions),p)
                    else:
                        probs =np.full((state.shape[0],nActions),p)
                    actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
                    return actions


                s = []
                for i in range(settings["SampleEpisodes"]):
                    s0 = env.reset()

                    for j in range(settings["MAX_EP_STEPS"]+1):

                        a = GetAction(state=s0)

                        s1,r,done,_ = env.step(a)
                        if arreq_in_list(s0,s):
                            pass
                        else:
                            s.append(s0)

                        s0 = s1
                        if done:
                            break

            #Creating and smoothing Q Maps
            def ConstructSamples(env,position2):
                grid = env.get_obs_blue
                locX,locY = np.unravel_index(np.argmax(grid[:,:,4], axis=None), grid[:,:,0].shape)
                locX2,locY2 = np.unravel_index(np.argmin(grid[:,:,4], axis=None), grid[:,:,0].shape)
                #Removing the agent
                grid[locX,locY,4] = 0
                grid[locX2,locY2,4] = 0

                stacked_grids = np.repeat(np.expand_dims(grid,0), grid.shape[0]*grid.shape[1],0)

                for i in range(stacked_grids.shape[1]):
                    for j in range(stacked_grids.shape[2]):
                        stacked_grids[i*stacked_grids.shape[2]+j,stacked_grids.shape[2]-i-1,j,4] = 5

                stacked_grids[:,position2[0],position2[1],4] = -5
                return stacked_grids

            def SmoothOption(option_, gamma =0.9):
                # option[option<0.0] = 0.0
                #Create the Adjacency Matric
                v_option=np.full((dFeatures[0],dFeatures[1],dFeatures[0],dFeatures[1]),0,dtype=np.float32)
                for i2,j2 in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                    option = option_[:,:,i2,j2]
                    states_ = {}
                    count = 0
                    for i in range(option.shape[0]):
                        for j in range(option.shape[1]):
                            if option[i,j] != 0:
                                states_[count] = [i,j]
                                # states_.append([count, [i,j]])
                                count+=1
                    states=len(states_.keys())
                    x = np.zeros((states,states))
                    for i in range(len(states_)):
                        [locx,locy] = states_[i]
                        sum = 0
                        for j in range(len(states_)):
                            if states_[j] == [locx+1,locy]:
                                x[i,j] = 0.25
                                sum += 0.25
                            if states_[j] == [locx-1,locy]:
                                x[i,j] = 0.25
                                sum += 0.25
                            if states_[j] == [locx,locy+1]:
                                x[i,j] = 0.25
                                sum += 0.25
                            if states_[j] == [locx,locy-1]:
                                x[i,j] = 0.25
                                sum += 0.25
                        x[i,i]= 1.0-sum

                    #Create W
                    w = np.zeros((states))
                    for count,loc in states_.items():
                        w[count] = option[loc[0],loc[1]]

                    # (I-gamma*Q)^-1
                    I = np.identity(states)
                    psi = np.linalg.inv(I-gamma*x)

                    smoothedOption = np.zeros_like(option,dtype=float)

                    value = np.matmul(psi,w)
                    for j,loc in states_.items():
                        smoothedOption[loc[0],loc[1]] = value[j]

                    v_option[:,:,i2,j2] = smoothedOption
                return v_option

            SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["SFNetworkConfig"],nActions,{},scope="Global")
            SF5.load_weights('./models/'+LoadName+ '/'+"model.h5")

            #Selecting the samples:
            psi = SF2.predict(np.vstack(s)) # [X,SF Dim]

            #test for approximate equality (for floating point types)
            def arreqclose_in_list(myarr, list_arrays):
                return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr,atol=1E-6)), False)
            if settings["Selection"]=="First":
                samples = [];points=[]
                i =0
                while len(samples) < settings["TotalSamples"]:
                    if not arreqclose_in_list(psi[i,:], samples):
                        samples.append(psi[i,:])
                        points.append(i)
                    i+=1
            elif settings["Selection"]=="Random":
                samples = [];points=[]
                while len(samples) < settings["TotalSamples"]:
                    idx = randint(1,psi.shape[0])
                    if not arreqclose_in_list(psi[idx,:], samples):
                        samples.append(psi[idx,:])
                        points.append(idx)
            elif settings["Selection"]=="Random_sampling":
                #PCA Decomp to dimension:
                import pandas as pd
                from sklearn.decomposition import PCA
                feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
                df = pd.DataFrame(psi,columns=feat_cols)
                np.random.seed(42)
                rndperm = np.random.permutation(df.shape[0])
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(df[feat_cols].values)

                from SampleSelection import SampleSelection_v1
                points = SampleSelection_v1(pca_result,settings["TotalSamples"],returnIndicies=True)
            elif settings["Selection"]=="Hull_pca":
                #PCA Decomp to dimension:
                import pandas as pd
                from sklearn.decomposition import PCA
                feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
                df = pd.DataFrame(psi,columns=feat_cols)
                np.random.seed(42)
                rndperm = np.random.permutation(df.shape[0])
                pca = PCA(n_components=4)
                pca_result = pca.fit_transform(df[feat_cols].values)

                from SampleSelection import SampleSelection_v2
                points = SampleSelection_v2(pca_result,settings["TotalSamples"],returnIndicies=True)
            elif settings["Selection"]=="Hull_tsne":
                #PCA Decomp to dimension:
                import pandas as pd
                from sklearn.manifold import TSNE
                feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
                df = pd.DataFrame(psi,columns=feat_cols)
                np.random.seed(42)
                rndperm = np.random.permutation(df.shape[0])
                tsne = TSNE(n_components=3, verbose=1, perplexity=10, n_iter=1000)
                tsne_results = tsne.fit_transform(df[feat_cols].values)

                from SampleSelection import SampleSelection_v2
                points = SampleSelection_v2(tsne_results,settings["TotalSamples"],returnIndicies=True)
            elif settings["Selection"]=="Hull_cluster":
                #PCA Decomp to dimension:
                import pandas as pd
                from sklearn.decomposition import PCA
                feat_cols = [ 'pixel'+str(i) for i in range(psi.shape[1]) ]
                df = pd.DataFrame(psi,columns=feat_cols)
                np.random.seed(42)
                rndperm = np.random.permutation(df.shape[0])
                pca = PCA(n_components=4)
                pca_result = pca.fit_transform(df[feat_cols].values)

                from SampleSelection import SampleSelection_v3
                points = SampleSelection_v3(pca_result,settings["TotalSamples"],returnIndicies=True)
            else:
                print("Invalid Method selected")
                exit()

            psiSamples=[]
            for point in points:
                psiSamples.append(psi[point,:])

            while len(psiSamples) < len(psiSamples[0]):
                psiSamples.extend(psiSamples)

            samps = np.stack(psiSamples)
            samps2 = samps[0:samps.shape[1],:]
            w_g,v_g = np.linalg.eig(samps2)

            # print("here")
            dim = samps2.shape[1]
            #Creating Sub-policies
            offset = 0
            options = []
            for sample in range(int(N/2)):
                print("Creating Option",sample)
                v_option=np.full((dFeatures[0],dFeatures[1],dFeatures[0],dFeatures[1]),0,dtype=np.float32)
                for i2,j2 in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
                    if sample+offset >= dim:
                        continue
                    grids = ConstructSamples(env,[i2,j2])
                    phi= SF3.predict(grids)
                    v_option[:,:,i2,j2]=np.real(np.matmul(phi,v_g[:,sample+offset])).reshape([dFeatures[0],dFeatures[1]])
                    if np.iscomplex(w_g[sample+offset]):
                        offset+=1
                print("Smoothing Option")
                v_option_ = SmoothOption(v_option)
                options.append(v_option_)
                options.append(-v_option_)
                #Plotting the first couple samples with random enemy positions:
                v_map = v_option_[:,:,10,10]
                imgplot = plt.imshow(v_map)
                plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(1)+".png")
                plt.close()
                v_map = v_option_[:,:,10,17]
                imgplot = plt.imshow(v_map)
                plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(2)+".png")
                plt.close()
                v_map = v_option_[:,:,17,10]
                imgplot = plt.imshow(v_map)
                plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(3)+".png")
                plt.close()
                v_map = v_option_[:,:,10,2]
                imgplot = plt.imshow(v_map)
                plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(4)+".png")
                plt.close()
                v_map = v_option_[:,:,2,10]
                imgplot = plt.imshow(v_map)
                plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                plt.savefig(IMAGE_PATH+"/option"+str(sample)+"_"+str(5)+".png")
                plt.close()

            #Saving the different options. to log:
            np.savez_compressed(MODEL_PATH_ +"options.npz", options=np.stack(options))

            self.options = options

        # Creating nested Method that will be updated.
        network = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=N)
        Method = GetFunction(settings["SubMethod"])
        self.nestedMethod = Method(sess,settings,netConfigOverride,stateShape=dFeatures,actionSize=N,nTrajs=nTrajs)

    def GetAction(self, state, episode=1,step=0):
        """
        Method to run data through the neural network.

        Parameters
        ----------
        state : np.array
            Data with the shape of [N, self.stateShape] where N is number of smaples

        Returns
        -------
        actions : list[int]
            List of actions based on NN output.
        extraData : list
            List of data that is passed to the execution code to be bundled with state data.
        """
        # Running the nested Method.
        action_hier, networkData = self.nestedMethod.GetAction(state,episode,step)

        #Using the Q-table to select a primitive action.
        action = self.UseSubpolicy(state,action_hier)

        return action ,[action_hier]+networkData


    def Update(self,episode=0):
        """
        Passes arguments to the nested method.
        """
        self.nestedMethod.Update(episode)

    def AddToTrajectory(self,sample):
        """
        Passes arguments to the nested method.
        """
        #Replacing the action with the hierarchical action.
        nestedMethodSample = [sample[0]] + [sample[5]] + sample[2:5] +sample[6:]
        self.nestedMethod.AddToTrajectory(nestedMethodSample)

    def GetStatistics(self):
        #Getting statistics of the nested methods
        return self.nestedMethod.GetStatistics()

    def UseSubpolicy(self,s,subpolicyNum):
        #Extracting location of agent.
        locX,locY = np.unravel_index(np.argmax(s[0,:,:,4], axis=None), s[0,:,:,0].shape)
        locX2,locY2 = np.unravel_index(np.argmin(s[0,:,:,4], axis=None), s[0,:,:,0].shape)
        #Getting Value of all adjacent policies. Ignoring location of the walls.
        actionValue = [self.options[int(subpolicyNum)][int(locX),int(locY),int(locX2),int(locY2)]]
        if locX+1 > s.shape[1]-1:
            actionValue.append(-999)
        elif [0,int(locX+1),int(locY),3] == 1:
            actionValue.append(-999)
        else:
            actionValue.append(self.options[int(subpolicyNum)][int(locX+1),int(locY),int(locX2),int(locY2)  ]) # Go Up

        if locY+1 > s.shape[2]-1:
            actionValue.append(-999)
        elif [0,int(locX),int(locY+1),3] == 1:
            actionValue.append(-999)
        else:
            actionValue.append(self.options[int(subpolicyNum)][int(locX),int(locY+1),int(locX2),int(locY2)    ]) # Go Right

        if locY-1 < 0:
            actionValue.append(-999)
        elif [0,int(locX-1),int(locY),3] == 1:
            actionValue.append(-999)
        else:
            actionValue.append(self.options[int(subpolicyNum)][int(locX-1),int(locY),int(locX2),int(locY2)  ]) # Go Down

        if locY-1<0:
            actionValue.append(-999)
        elif [0,int(locX),int(locY-1),3] == 1:
            actionValue.append(-999)
        else:
            actionValue.append(self.options[int(subpolicyNum)][int(locX),int(locY-1),int(locX2),int(locY2)    ]) # Go Left

        #Selecting Action with Highest Value. Will always take a movement.
        return actionValue.index(max(actionValue))



    @property
    def getVars(self):
        return self.nestedMethod.getVars
