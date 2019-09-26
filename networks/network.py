"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""

import numpy as np


class Network(object):
    def __init__(self,namespace):
        """
        """
        raise NotImplementedError

    def ChooseAction(self, s,):
        """
        """
        raise NotImplementedError

    def Learn(self, buffer):
        """
        """
        raise NotImplementedError

    def SaveStatistics(self,saver):
        """
        """
        raise NotImplementedError
