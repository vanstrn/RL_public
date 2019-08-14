""" Reinforce Learning Policy Generator

This module calls any pre-trained tensorflow model and generates the policy.
It includes method to switch between the network and weights.

For use: generate policy to simulate along with CTF environment.

for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

Last Modified:
    Seung Hyun Kim
    created :Wed Oct 24 12:21:34 CDT 2018
"""

import numpy as np

class Clone:
    """Policy generator class for CtF env.

    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action  : Required method to generate a list of actions.
        load_model  : Load pre-defined model (*.meta file). Only TensorFlow model supported
        load_weight : Load/reload weight to the model.
    """

    def __init__(self, clone=None):
        """Constuctor for policy class.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """
        self.gen_action = clone

    def PolicyGen(self, free_map, agent_list):
        return self
