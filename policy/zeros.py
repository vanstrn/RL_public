
"""Stationary

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

"""

#import numpy as np

from policy.policy import Policy

class Zeros(Policy):
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
        patrol: Private method to control a single unit.
    """
    def gen_action(self, agent_list, observation, free_map=None):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        action_out = []
        
        if free_map is not None: self.free_map = free_map
        
        for idx,agent in enumerate(agent_list):
            action_out.append(0)
        
        return action_out

