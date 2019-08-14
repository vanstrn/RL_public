"""Defense  agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np
import gym_cap.envs.const as const

from policy.policy import Policy

class Defense(Policy):
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
        patrol: Private method to control a single unit.
    """

    def __init__(self):
        super().__init__()

    def initiate(self, free_map, agent_list):
        self.free_map = free_map
        self.free_map_old = free_map
        self.team = agent_list[0].team

        self.flag_location = None
        self.random = np.random
        self.exploration = 0.5

        self.flag_code = const.TEAM1_FLAG

    def gen_action(self, agent_list, observation):
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

        # if map changes then reset the flag location
        # search for a flag until finds it
        if self.flag_location == None:

            loc = self.scan_obs(self.free_map, self.flag_code)
            if len(loc) is not 0:
                self.flag_location = loc[0]

            for idx,agent in enumerate(agent_list):
                a = self.random_search(agent, idx, self.free_map)
                action_out.append(a)

            return action_out

        # go to the flag to defend it
        for idx,agent in enumerate(agent_list):
            a = self.flag_approach(agent, idx, self.free_map)
            action_out.append(a)

        return action_out

    def random_search(self, agent, index, obs):
        """Generate 1 action for given agent object."""
        action = self.random.randint(0, 5)

        return action

    def flag_approach(self, agent, index, obs):
        """Generate 1 action for given agent object."""
        x,y = agent.get_loc()
        action = 0

        if self.flag_location[0] > x+1:
            action = 2
        elif self.flag_location[0] < x-1:
            action = 4
        elif self.flag_location[1] > y+1:
            action = 3
        elif self.flag_location[1] < y-1:
            action = 1

        if self.random.random() < self.exploration:
            action = self.random.randint(0, 5)

        return action

    def scan_obs(self, obs, value):
        location = []

        for y in range(len(obs)):
            for x in range(len(obs[0])):
                if obs[x][y] == const.TEAM1_FLAG:
                    location.append([x,y])

        return location
