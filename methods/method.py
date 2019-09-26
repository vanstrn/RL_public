"""
Sets up the basic Method Class which lays out all required functions of a Neural Network Method.
Each new method should have a link either to reference paper or a link to some description of the network in GitHub Repo.
"""

import numpy as np


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

    def SaveStatistics(self,saver):
        """
        Contains the code to save internal information of the Neural Network.
        """
        raise NotImplementedError
