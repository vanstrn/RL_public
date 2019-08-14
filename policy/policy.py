"""Policy Template

This module includes all necessary features for PolicyGen class.
Module can be assigned as blue/red policy within CtF environment
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np

import gym_cap.envs.const as const

class Policy:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    Provides basic methods for building policy.
    
    Must-have Methods:
        initiate: Required method that runs everytime episode is initialized.
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.

        Define:
            agent_list (list): list of all friendly units.
            free_map (np.array): 2d map of static environment (optional).
        
        """
        self.free_map = None
        self.agent_list = None
        
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
        raise NotImplementedError

    def initiate(self, free_map, agent_list):
        """Initiation method
        
        This method is called when the environment reset
        Any initialization or initiation for each game should be included here
        The new static-map and agent list is given as parameter
        
        Args:
            agent_list (list): list of all friendly units.
            free_map (np.array): 2d map of static environment (optional).
        """
        self.free_map = free_map
        self.agent_list = agent_list

    """
    All the methods below can be used to build policy.
    Methods can be used in gen_action() or initiate() methods.

    Methods:
        move_toward : Output corresponding action given two coordinates
        next_loc    : Output coordinate after action
        can_move    : Check if the move is possible from the position
        distance    : Calculate distance between two point
        route_astar : Outputs route(coordinate) from start to end 
    """
    def move_toward(self, start, target):
        """
        Output action to move from start to neighbor.
        It is crude method to move in certain direction.
        Due to the grid-environment, it does not move diagonally.

        Args:
            start (tuple): coordinate of staring location 
            target (tuple): coordinate of targeting location

        Return:
            int : corresponding action to move towards the target
        """
        if target[1] - start[1] > 0:
            return 3
        elif target[1] - start[1] < 0:
            return 1
        elif target[0] - start[0] > 0:
            return 2
        elif target[0] - start[0] < 0:
            return 4
        else:
            return 0  # Only when start==neighbor

    def next_loc(self, position, move):
        """
        Return next coordinate

        Args:
            position (tuple)
            move (int)

        Return:
            tuple
        """
        dir_x = [0, 0, 1, 0, -1]
        dir_y = [0,-1, 0, 1,  0]
        return (position[0]+dir_x[move], position[1]+dir_y[move])

    def can_move(self, position, move):
        """
        Check if the movement is possible

        Args:
            position (tuple)
            move (int)

        Return:
            bool
        """
        nx, ny = self.next_loc(position, move)
        if nx < 0 or nx >= 20:
            return False
        elif ny < 0 or ny >= 20:
            return False
        return self.free_map[nx][ny] != const.OBSTACLE

    def distance(self, start, goal, euc=False):
        """
        Distance between two point
        Use L1 norm distance for grid world

        Args:
            start (tuple)
            end (tuple)
            euc (boolean): Set true to make it Euclidean distance

        return:
            int
        """
        if euc:
            return ((start[0]-goal[0])**2 + (start[1]-goal[1])**2) ** 0.5
        return abs(start[0]-goal[0]) + abs(start[1]-goal[1])

    def route_astar(self, start, goal):
        """
        Finds route from start to goal.
        Implemented A* algorithm

        *The 1-norm distance was used

        Args:
            start (tuple): coordinate of start position
            end (tuple): coordinate of end position

        Return:
            total_path (list):
                List of coordinate in tuple.
                Return None if path does not exist.

        """

        openSet = set([start])
        closedSet = set()
        cameFrom = {}
        fScore = {}
        gScore = {}
        if len(goal) == 0:
            return None
        fScore[start] = self.distance(start, goal)
        gScore[start] = 0

        while openSet:
            min_score = min([fScore[c] for c in openSet])
            for position in openSet:
                if fScore.get(position,np.inf) == min_score:
                    current = position
                    break

            if current == goal:
                total_path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    total_path.append(current)
                total_path.reverse()

                return total_path

            openSet.remove(current)
            closedSet.add(current)

            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            neighbours = []
            for dx, dy in directions:
                x2, y2 = current
                x = x2 + dx
                y = y2 + dy
                if (x >= 0 and x < self.free_map.shape[0]) and \
                   (y >= 0 and y < self.free_map.shape[1]) and \
                   self.free_map[x,y] != const.OBSTACLE:
                    neighbours.append((x, y))

            for neighbour in neighbours:
                if neighbour in closedSet:
                    continue
                tentative_gScore = gScore[current]  # + transition cost
                if neighbour not in openSet:
                    openSet.add(neighbour)
                elif tentative_gScore >= gScore[neighbour]:
                    continue
                cameFrom[neighbour] = current
                gScore[neighbour] = tentative_gScore
                fScore[neighbour] = gScore[neighbour] + self.distance(neighbour, goal)

        return None
