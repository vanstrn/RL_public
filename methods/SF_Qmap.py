
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

class SF_QMap(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,env,nTrajs=1,**kwargs):
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
        MODEL_PATH_ = './models/'+EXP_NAME+'/'
        MODEL_PATH = './models/'+LoadName+'/'
        LOG_PATH = './logs/'+EXP_NAME+'/'
        CreatePath(LOG_PATH)
        CreatePath(MODEL_PATH_)
        self.sess=sess
        self.env=env

        N = settings["NumOptions"]

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
                print("Creating Samples")
                #Creating Instance of environment and running through it to generate samples
                def GetAction(state):
                    """
                    Contains the code to run the network based on an input.
                    """
                    p = 1/actionSize
                    if len(state.shape)==3:
                        probs =np.full((1,actionSize),p)
                    else:
                        probs =np.full((state.shape[0],actionSize),p)
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

            with open(MODEL_PATH+'netConfigOverride.json') as json_file:
                networkOverrides = json.load(json_file)
            # if "DefaultParams" not in networkOverrides:
            #     networkOverrides["DefaultParams"] = {}
            # networkOverrides["DefaultParams"]["Trainable"]=False
            SF1,SF2,SF3,SF4,SF5 = buildNetwork(settings["SFNetworkConfig"],actionSize,networkOverrides,scope="Global")
            SF5.load_weights(MODEL_PATH+"model.h5")

            #Selecting the samples:
            psi = SF2.predict(np.vstack(s)) # [X,SF Dim]

            #test for approximate equality (for floating point types)
            def arreqclose_in_list(myarr, list_arrays):
                return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr,atol=1E-6)), False)

            print("Selecting Samples")
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
                    idx = randint(1,psi.shape[0]-1)
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

            # QMapStructure = self.env.GetQMapStructure()
            print("Getting data for a Q-Map")
            grids = self.env.ConstructAllSamples()
            phis= SF3.predict(grids)

            for sample in range(int(N/2)):
                print("Creating Option",sample)
                if sample+offset >= dim:
                    continue
                v_option,v_option_inv=self.env.ReformatSamples(np.real(np.matmul(phis,v_g[:,sample+offset])))
                options.append(v_option)
                options.append(v_option_inv)
                if np.iscomplex(w_g[sample+offset]):
                    offset+=1
                if settings["PlotOptions"]:
                    imgplot = plt.imshow(v_option)
                    plt.title(" Option "+str(sample)+" Value Estimate | Eigenvalue:" +str(w_g[sample+offset]))
                    plt.savefig(LOG_PATH+"/option"+str(sample)+".png")
                    plt.close()

                #Plotting the first couple samples with random enemy positions:

            #Saving the different options. to log:
            np.savez_compressed(MODEL_PATH_ +"options.npz", options=np.stack(options))

            self.options = options

        # Creating nested Method that will be updated.
        network = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=N)
        Method = GetFunction(settings["SubMethod"])
        self.nestedMethod = Method(sess,settings,netConfigOverride,stateShape=stateShape,actionSize=N,nTrajs=nTrajs)

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
        return self.env.UseSubpolicy(s,self.options[int(subpolicyNum)])




    @property
    def getVars(self):
        return self.nestedMethod.getVars
