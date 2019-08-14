import numpy as np
from policy.policy import Policy
from gym_cap.envs.const import *


class Fighter(Policy):
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self):

        super().__init__()

    def initiate(self, free_map, agent_list):
        super().initiate(free_map, agent_list)
        self.agent_type = {agent_list[0]:'aggr',agent_list[1]:'aggr',agent_list[2]:'aggr',agent_list[3]:'def'}
    
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
            if not agent.isAlive:
                action_out.append(0)
                continue

            if self.agent_type[agent] == 'aggr':
                goal = self.search_nearest(agent, observation, TEAM2_UGV)
                
                if goal == agent.get_loc():
                    action_out.append(0)
                else:
                    action_out.append(self.aggr_policy(agent,observation,goal,idx))

            elif self.agent_type[agent] == 'def':
                action = self.def_policy(agent,observation,idx)
                action_out.append(action)

        return action_out
    
    def search_nearest(self, agent, obs, code):
        """
        function for finding the nearest code
        """
        dist = []        
        end = np.argwhere(obs[:,:,CHANNEL[code]]==REPRESENT[code])
        if len(end) != 0:
            for fx, fy in end:
                x, y = agent.get_loc()
                dist.append((fx-x)**2 + (fy-y)**2)
            min_dist = np.argmin(dist)
            return tuple(end[min_dist])

        else:
            return agent.get_loc()

    def aggr_policy(self, agent, obs, goal, idx):
        """
        policy for aggresive agent
        """
        cur_loc = agent.get_loc()

        route = self.route_astar(cur_loc, goal) 
        if len(route) > 1:
            new_loc = route[1]
            return self.move_toward(cur_loc, new_loc)
        return 0
    
    def def_policy(self, agent, obs, idx):
        """
        policy for defensive agent
        """
        guard_radius = 4
        down_radius = 6
        flag_x, flag_y = np.argwhere(obs[:,:,CHANNEL[TEAM1_FLAG]]==REPRESENT[TEAM1_FLAG])[0]
        enemy_x, enemy_y = self.search_nearest(agent,obs,TEAM2_UGV)
        x, y = agent.get_loc()

        if (flag_x-x)**2 + (flag_y-y)**2 <= guard_radius**2 and (enemy_x-x)**2 + (enemy_y-y)**2 <= down_radius**2:
            action = self.aggr_policy(agent,obs,(enemy_x,enemy_y),idx)
        elif (flag_x-x)**2 + (flag_y-y)**2 >= guard_radius**2 and (enemy_x-x)**2 + (enemy_y-y)**2 <= down_radius**2:
            action = self.aggr_policy(agent,obs,(enemy_x,enemy_y),idx)
        elif (flag_x-x)**2 + (flag_y-y)**2 <= guard_radius**2 and (enemy_x-x)**2 + (enemy_y-y)**2 >= down_radius**2:
            action = 0
        else:
            action = self.aggr_policy(agent,obs,(flag_x, flag_y),idx) # Return to flag

        return action
