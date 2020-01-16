"""
Sets up the basic Method Class which lays out all required functions of a Neural Network Method.
Each new method should have a link either to reference paper or a link to some description of the network in GitHub Repo.
"""

import numpy as np
import tensorflow as tf

class Method(object):
    def __init__(self,namespace):
        """
        Builds the Neural Network.
            -Create inputs and outputs for the Neural Network.
            -Labels all layers so they can be reused properly in a Transfer Scenario.
        Creates different statistics
            -Vanishing gradient which can be used for debuging.
        """
        raise NotImplementedError

    def ChooseAction(self, s):
        """
        Contains the code to run the network based on an input.
        """
        raise NotImplementedError

    def Update(self, buffer):
        """
        Contains the code to run updates on the network.
        """
        raise NotImplementedError

    def AddToTrajectory(self,sample):
        """Add a sample to the buffer.
        Takes the form of [s0,a,r,s1,done,extraData]
        extraData is outputted from the Network and is appended to the sample.
        Also handles any data that needs to be processed in the network.
        """
        self.buffer[0].append(sample)

    def ClearTrajectory(self):
        """Add a sample to the buffer.
        Takes the form of [s0,a,r,s1,done,extraData]
        extraData is outputted from the Network and is appended to the sample.
        Also handles any data that needs to be processed in the network.
        """
        for traj in self.buffer:
            traj.clear()

    def SaveStatistics(self,saver):
        """
        Contains the code to save internal information of the Neural Network.
        """
        raise NotImplementedError

    def InitializeVariablesFromFile(self,saver, model_path):
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")
