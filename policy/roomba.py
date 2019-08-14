"""Simple agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np
from collections import defaultdict

from policy.policy import Policy

class Roomba(Policy):
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
        policy: Method to determine the action based on observation for a single unit
        scan : Method that returns the dictionary of object around the agent

    Variables:
        exploration : exploration rate
        previous_move : variable to save previous action
    """

    def initiate(self, free_map, agent_list):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.free_map = free_map
        self.agent_list = agent_list

        self.random = np.random
        self.exploration = 0.05
        self.previous_move = self.random.randint(0, 5, len(agent_list)).tolist()

        self.enemy_range = 4 # Range to see around and avoid enemy
        self.flag_range = 5  # Range to see the flag

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

        for idx, agent in enumerate(agent_list):
            # Initialize Variables
            action = self.policy(agent, observation, idx)
            action_out.append(action)

        return action_out

    def policy(self, agent, obs, agent_id):
        """ Policy

        This method generate an action for given agent.
        Agent is given with limited vision of a field.
        This method provides simple protocol of movement based on the agent's location and it's vision.

        Protocol :
            1. Scan the area with flag_range:
                - Flag within radius : Set the movement towards flag
                - No Flag : random movement
            2. Scan the area with enemy_range:
                - Enemy in the direction of movement
                    - If I'm in enemy's territory: reverse direction
                    - Continue
                - Else: contonue moving in the direction
            3. Random exploration
                - 0.1 chance switching direction of movement
                - Follow same direction
                - Change direction if it heats the wall
        """
        dir_x = [0, 0, 1, 0, -1] # dx for [stay, down, right, up, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, down, right, up,left]
        def blocking(x,y,d):
            nx = x + dir_x[d]
            ny = y + dir_y[d]
            if nx < 0 or nx >= mapx: return True
            elif ny < 0 or ny >= mapy: return True

            return self.free_map[nx][ny]==8 or obs[nx,ny,4] != 0

        lx, ly = agent.get_loc()
        mapx, mapy = self.free_map.shape

        # Continue the previous action
        action = self.previous_move[agent_id]

        # 1. Set direction to flag
        is_flag, coord = self.obj_in_range(lx, ly, self.flag_range, obs, 2)
        if is_flag:
            fx, fy = coord[0]
            action_pool = []
            if fy > 0: # move down
                action_pool.append(3)
            if fy < 0: # move up
                action_pool.append(1)
            if fx > 0: # move left
                action_pool.append(2)
            if fx < 0: # move right
                action_pool.append(4)
            if action_pool == []:
                action_pool = [0]
            action = np.random.choice(action_pool)
        
        # 2. Scan with enemy range
        in_home = self.free_map[agent.get_loc()] == agent.team
        is_enemy, enemy_locs = self.obj_in_range(lx, ly, self.enemy_range, obs, 4)
        opposite_move = [0, 3, 4, 1, 2]
        for ex, ey in enemy_locs:
            if not in_home:
                if (ey > 0 and abs(ex) < 2 and action == 3) or \
                   (ey < 0 and abs(ex) < 2 and action == 1) or \
                   (ex > 0 and abs(ey) < 2 and action == 2) or \
                   (ex < 0 and abs(ey) < 2 and action == 4):
                    action = opposite_move[action]
            else:
                if ey > 0: # move down
                    action = 3
                elif ey < 0: # move up
                    action = 1
                elif ex > 0: # move left
                    action = 2
                elif ex < 0: # move right
                    aciton = 4

        if action == 0 or np.random.random() <= self.exploration: # Exploration
            action = np.random.randint(1,5)

        # Checking obstacle
        if blocking(lx, ly, action): # Wall or other obstacle
            action_pool = [move for move in range(1,5) if not blocking(lx, ly, move)]
            if action_pool == []:
                action_pool = [0]
            action = np.random.choice(action_pool)

        # Save move
        self.previous_move[agent_id] = action

        return action

    def obj_in_range(self, x, y, r, obs, chn, elem=-1):
        loc_list = np.where(obs[:,:,chn]==elem)
        #rel_coord = np.column_stack(loc_list) - (x,y)
        dif_coord = []
        rsqr = r*r
        for dx, dy in zip(*loc_list):
            dx -= x
            dy -= y
            if dx*dx+dy*dy <= rsqr:
                dif_coord.append((dx,dy))
        return len(dif_coord)>0, dif_coord

    def center_pad(self, m, width, padder=8):
        lx, ly = m.shape
        pm = np.empty((lx+(2*width),ly+(2*width)), dtype=np.int)
        pm[:] = padder
        pm[width:lx+width, width:ly+width] = m
        return pm

