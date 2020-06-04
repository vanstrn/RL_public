"""
Outlines the basic structure for a Reinforcement Learning Method
Each new method should have a link either to reference paper or a link to some description of the network in GitHub Repo.
"""

import numpy as np
import tensorflow as tf

class Method(object):
    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initialization for a method.

        Parameters:
        -----
        sess : Tensorflow Session
            Initialized Tensorflow session
        settings : dict
            Dictionary of the possible inputs to the method. Important attributes:
            networkConfig, HPs
        netConfigOverride : dict
            Disctionary containing any variables that will overwrite the
        stateShape : list
            List of integers of the inputs shape size. Ex [39,39,6]
        actionSize : int
            Output size of the network.
        nTrajs : int (Optional)
            Number that specifies the number of trajectories to be created for collecting training data.
        **kwargs :
            Used to collect unused variables

        Returns:
        -----
        N/A

        -----
        Initialization for a method. This should:
        1. Build the network(s) based off on settings["networkConfig"],netConfigOverride
        2. Create I/O for the network
        3. Define loss functions and update functions which are used to update the network
        4. Create data buffers that store the data and are used to update the network
        5. Create statistic buffers which track attributes of the network and can be used for logging

        """
        raise NotImplementedError

    def ChooseAction(self, s):
        """
        Contains the code to run the network based on an input.

        Parameters:
        -----
        s : numpy.array or list
            state space to run the network on

        Returns:
        -----
        actions : list
            The actions the method chooses
        networkData : list
            A list of additional data from the network. This is done to allow
            the data to be associated with the explicit action in the data buffer.

        -----
        Method should return an action.

        """
        raise NotImplementedError

    def Update(self, episode=0):
        """
        Contains the code to update the network.

        Parameters:
        -----
        episode : int
            Episode number. Future input that will allows network to change
            learning parameters based on episode.

        Returns:
        -----
        N/A

        -----
        Method should perform the following generic actions
        1. Load data and process data from buffers in the method.
        2. Update the network by calling update operations created in __init__
        3. Add relavent statistics to the statistics buffers.

        """
        raise NotImplementedError
        raise NotImplementedError

    def AddToTrajectory(self,sample):
        """Adds a sample to the buffer.

        Parameters:
        -----
        sample : list
            Takes the form of [s0,a,r,s1,done,extraData]

        Returns:
        -----
        N/A

        -----
        takes the inputted sample and puts them into separate ordered lists of the same data.
        extraData is outputted from the Network and is appended to the sample. This
        allows pass through for any network parameters that are needed for updating.
        """
        try:

            target_shape = sample[0].shape[0]
            if target_shape == 1:
                self.buffer[0].append(sample)
                return
            if target_shape != len(self.buffer):
                raise ValueError
            for i in range(len(sample)-1):
                if sample[i+1].shape[0] != target_shape:
                    sample[i+1] = sample[i+1].reshape(target_shape)

            for i in range(len(sample[0])):
                tmp = []
                for j in range(len(sample)):
                    tmp.append(sample[j][i])
                self.buffer[i].append(tmp)

        except ValueError: #Singular inputs.
            self.buffer[0].append(sample)

    def ClearTrajectory(self):
        """Empties the buffer of all data.

        Parameters:
        -----
        N/A

        Returns:
        -----
        N/A

        -----
        This assumes the common buffer that I use where there are multiple buffers in a list.

        """
        for traj in self.buffer:
            traj.clear()

    def GetStatistics(self,saver):
        """Gathers and labels different statistics for the network

        Parameters:
        -----
        N/A

        Returns:
        -----
        statisticsDictionary : dict
            A dictionary with operational statistics of the network.

        -----
        """
        raise NotImplementedError

    def InitializeVariablesFromFile(self,saver, model_path):
        """Initializes variables for the method.

        Parameters:
        -----
        saver : tf.train.Saver
            Saver instance that has all the variables of the network.
        model_path : str
            Path to where Tensorflow ckpt exists.

        Returns:
        -----
        N/A

        -----
        """

        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

    @property
    def getVars(self):
        """Gathers all variable of the network.
        This will often call the internal network.

        Parameters:
        -----
        N/A

        Returns:
        -----
        variables : list
            A list of variables used by the method(network).

        -----
        """
        raise NotImplementedError
