import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import gym_cap.heuristic as policy
import time
from gym import spaces
from random import randint
import random
import os
from utils.utils import MovingAverage



def use_this_policy(policyName=None):
    if policyName is None:
        heur_policy_list = [policy.Patrol(), policy.Roomba(), policy.Defense(), policy.Random(), policy.AStar()]
        heur_weight = [1,1,1,1,1]
        heur_weight = np.array(heur_weight) / sum(heur_weight)
        return np.random.choice(heur_policy_list, p=heur_weight)
    elif policyName == "Roomba":
        return policy.Roomba()
    elif policyName == "Patrol":
        return policy.Patrol()
    elif policyName == "Defense":
        return policy.Defense()
    elif policyName == "AStar":
        return policy.AStar()
    elif policyName == "Random":
        return policy.Random()

def SmoothOption(option, obstacles,gamma =0.9):
    # option[option<0.0] = 0.0
    #Create the Adjacency Matric
    states_ = {}
    count = 0
    for i in range(option.shape[0]):
        for j in range(option.shape[1]):
            if option[i,j] != 0:
                states_[count] = [i,j]
                # states_.append([count, [i,j]])
                count+=1
    states=len(states_.keys())
    x = np.zeros((states,states))
    for i in range(len(states_)):
        [locx,locy] = states_[i]
        sum = 0
        for j in range(len(states_)):
            if states_[j] == [locx+1,locy]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx-1,locy]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx,locy+1]:
                x[i,j] = 0.25
                sum += 0.25
            if states_[j] == [locx,locy-1]:
                x[i,j] = 0.25
                sum += 0.25
        x[i,i]= 1.0-sum

    #Create W
    w = np.zeros((states))
    for count,loc in states_.items():
        w[count] = option[loc[0],loc[1]]

    # (I-gamma*Q)^-1
    I = np.identity(states)
    psi = np.linalg.inv(I-gamma*x)

    smoothedOption = np.zeros_like(option,dtype=float)

    value = np.matmul(psi,w)
    for j,loc in states_.items():
        smoothedOption[loc[0],loc[1]] = value[j]

    return smoothedOption

def SmoothOption_2(option_,obstacles, gamma =0.9):
    # option[option<0.0] = 0.0
    #Create the Adjacency Matric
    v_option=np.full((dFeatures[0],dFeatures[1],dFeatures[0],dFeatures[1]),0,dtype=np.float32)
    for i2,j2 in itertools.product(range(dFeatures[0]),range(dFeatures[1])):
        option = option_[:,:,i2,j2]
        states_ = {}
        count = 0
        for i in range(option.shape[0]):
            for j in range(option.shape[1]):
                if option[i,j] != 0:
                    states_[count] = [i,j]
                    # states_.append([count, [i,j]])
                    count+=1
        states=len(states_.keys())
        x = np.zeros((states,states))
        for i in range(len(states_)):
            [locx,locy] = states_[i]
            sum = 0
            for j in range(len(states_)):
                if states_[j] == [locx+1,locy] and not obstacles[locx+1,locy]:
                    x[i,j] = 0.25
                    sum += 0.25
                if states_[j] == [locx-1,locy] and not obstacles[locx-1,locy]:
                    x[i,j] = 0.25
                    sum += 0.25
                if states_[j] == [locx,locy+1] and not obstacles[locx,locy+1]:
                    x[i,j] = 0.25
                    sum += 0.25
                if states_[j] == [locx,locy-1] and not obstacles[locx,locy-1]:
                    x[i,j] = 0.25
                    sum += 0.25
            x[i,i]= 1.0-sum

        #Create W
        w = np.zeros((states))
        for count,loc in states_.items():
            w[count] = option[loc[0],loc[1]]

        # (I-gamma*Q)^-1
        I = np.identity(states)
        psi = np.linalg.inv(I-gamma*x)

        smoothedOption = np.zeros_like(option,dtype=float)

        value = np.matmul(psi,w)
        for j,loc in states_.items():
            smoothedOption[loc[0],loc[1]] = value[j]

        v_option[:,:,i2,j2] = smoothedOption
    return v_option

class SampleConstructor(gym.core.Wrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.agents_repr = 5
        self.flags_repr = 2

    def ConstructSample(self,position):
        """Constructing All Samples into a Q table. """
        grid = self.get_obs_blue
        if grid[position[0],position[1],3] == 1:
            return None
        #Moving the agent
        loc = self.get_team_blue[0].get_loc()
        grid[loc[0],loc[1],4]=0
        grid[:,:,4] = np.where(grid[:,:,4]==5,self.agents_repr,grid[:,:,4])
        grid[:,:,4] = np.where(grid[:,:,4]==-1,-self.agents_repr,grid[:,:,4])
        grid[:,:,2] = np.where(grid[:,:,2]==1,self.flags_repr,grid[:,:,2])
        grid[:,:,2] = np.where(grid[:,:,2]==-1,-self.flags_repr,grid[:,:,2])
        grid[position[0],position[1],4] = self.agents_repr
        return grid
    def ConstructSample_e(self,position,position2):
        """Constructing All Samples into a Q table. """
        grid = self.get_obs_blue
        if grid[position[0],position[1],3] == 1:
            return None
        #Moving the agent
        loc = self.get_team_blue[0].get_loc()
        grid[loc[0],loc[1],4]=0
        loc = self.get_team_red[0].get_loc()
        grid[loc[0],loc[1],4]=0

        grid[:,:,4] = np.where(grid[:,:,4]==1,self.agents_repr,grid[:,:,4])
        grid[:,:,4] = np.where(grid[:,:,4]==-1,-self.agents_repr,grid[:,:,4])
        grid[:,:,2] = np.where(grid[:,:,2]==1,self.flags_repr,grid[:,:,2])
        grid[:,:,2] = np.where(grid[:,:,2]==-1,-self.flags_repr,grid[:,:,2])
        grid[position[0],position[1],4] = self.agents_repr
        grid[position2[0],position2[1],4] = -self.agents_repr
        return grid

    def ConstructAllSamples(self):
        """Constructing All Samples into a Q table. """
        #### Getting Background Grid
        grid = self.get_obs_blue
        locX,locY = np.unravel_index(np.argmax(grid[:,:,4], axis=None), grid[:,:,0].shape)
        locX2,locY2 = np.unravel_index(np.argmin(grid[:,:,4], axis=None), grid[:,:,0].shape)
        #Removing the agent
        grid[locX,locY,4] = 0
        grid[locX2,locY2,4] = 0

        #### Creating Grids for no enemies
        stacked_grids = np.repeat(np.expand_dims(grid,0), grid.shape[0]*grid.shape[1],0)
        for i in range(stacked_grids.shape[1]):
            for j in range(stacked_grids.shape[2]):
                stacked_grids[i*stacked_grids.shape[2]+j,stacked_grids.shape[2]-i-1,j,4] = self.agents_repr

        if self.mode == "sandbox":
            return stacked_grids

        #### Creating Grids for 1v1 Case
        all_grids = [stacked_grids]
        if self.mode != "sandbox":

            for i_enemy in range(stacked_grids.shape[1]):
                for j_enemy in range(stacked_grids.shape[2]):

                    stacked_grids_i = np.repeat(np.expand_dims(grid,0), grid.shape[0]*grid.shape[1],0)
                    for i in range(stacked_grids_i.shape[1]):
                        for j in range(stacked_grids_i.shape[2]):
                            stacked_grids_i[i*stacked_grids_i.shape[2]+j,stacked_grids_i.shape[2]-i-1,j,4] = self.agents_repr
                            stacked_grids_i[i*stacked_grids_i.shape[2]+j,stacked_grids_i.shape[2]-i_enemy-1,j_enemy,4] = -self.agents_repr

                    print(i_enemy,j_enemy)
                    all_grids.append(stacked_grids_i)

        return np.vstack(all_grids)

    def ReformatSamples(self,values):
        """Formating Data back into a Q Table. """
        grid = self.get_obs_blue
        if self.mode == "sandbox":
            value_map = np.reshape(values,grid.shape[:2])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    # print(grid[i,j,3],i,j)
                    if grid[i,j,3] == 1:
                        value_map[i,j] = 0.0
                        # print("Here")
            smoothed_value_map = SmoothOption(value_map,grid[i,j,3])
            smoothed_value_map_inv = -smoothed_value_map
            for i in range(grid.shape[1]):
                for j in range(grid.shape[2]):
                    if grid[i,j,3] == 1:
                        # smoothed_value_map[i,j] = -999
                        # smoothed_value_map_inv[i,j] = -999
                        pass
            return smoothed_value_map, smoothed_value_map_inv

        else:
            print("Here?")
            #Getting the first value_map for no
            value_map = np.reshape(values[:grid.shape[0]*grid.shape[1],:],grid.shape[:2])
            for i in range(grid.shape[1]):
                for j in range(grid.shape[2]):
                    if grid[i,j,3] == 1:
                        value_map[i,j] = 0.0
            smoothed_value_map_1v0 = SmoothOption(value_map,grid[i,j,3])
            smoothed_value_map_inv_1v0 = -smoothed_value_map
            for i in range(grid.shape[1]):
                for j in range(grid.shape[2]):
                    if grid[i,j,3] == 1:
                        smoothed_value_map_1v0[i,j] = -999
                        smoothed_value_map_inv_1v0[i,j] = -999

            #Other value maps
            value_map = np.reshape(values[grid.shape[1]*grid.shape[1]:,:],grid.shape+grid.shape[:2])
            for i in range(grid.shape[1]):
                for j in range(grid.shape[2]):
                    if grid[i,j,3] == 1:
                        value_map[i,j,:,:] = 0.0

            smoothed_value_map = SmoothOption_2(value_map,grid[i,j,3])
            smoothed_value_map_inv = -smoothed_value_map
            for i in range(grid.shape[1]):
                for j in range(grid.shape[2]):
                    if grid[i,j,3] == 1:
                        smoothed_value_map[i,j,:,:] = -999
                        smoothed_value_map_inv[i,j,:,:] = -999
            value_map_final = np.zeros([grid.shape[0],grid.shape[1],grid.shape[0]+1,grid.shape[1]+1])
            value_map_final[:,:,:grid.shape[0],:grid.shape[1]] = smoothed_value_map
            value_map_final[:,:,grid.shape[0],:grid.shape[1]] = smoothed_value_map_1v0
            value_map_final_inv = np.zeros([grid.shape[0],grid.shape[1],grid.shape[0]+1,grid.shape[1]+1])
            value_map_final_inv[:,:,:grid.shape[0],:grid.shape[1]] = smoothed_value_map_inv
            value_map_final_inv[:,:,grid.shape[0],:grid.shape[1]] = smoothed_value_map_inv_1v0

    def UseSubpolicy(self,s,subpolicy):
        #Extracting location of agent.

        if self.mode == "sandbox":

            locX,locY = np.unravel_index(np.argmax(s[0,:,:,4], axis=None), s[0,:,:,0].shape)
            #Getting Value of all adjacent policies. Ignoring location of the walls.
            actionValue = [subpolicy[int(locX),int(locY)]]
            if locX+1 > s.shape[1]-1:
                actionValue.append(-999)
            elif [0,int(locX+1),int(locY),3] == 1:
                actionValue.append(-999)
            else:
                actionValue.append(subpolicy[int(locX+1),int(locY)]) # Go Up

            if locY+1 > s.shape[2]-1:
                actionValue.append(-999)
            elif [0,int(locX),int(locY+1),3] == 1:
                actionValue.append(-999)
            else:
                actionValue.append(subpolicy[int(locX),int(locY+1)]) # Go Right

            if locY-1 < 0:
                actionValue.append(-999)
            elif [0,int(locX-1),int(locY),3] == 1:
                actionValue.append(-999)
            else:
                actionValue.append(subpolicy[int(locX-1),int(locY)]) # Go Down

            if locY-1<0:
                actionValue.append(-999)
            elif [0,int(locX),int(locY-1),3] == 1:
                actionValue.append(-999)
            else:
                actionValue.append(subpolicy[int(locX),int(locY-1)]) # Go Left

            #Selecting Action with Highest Value. Will always take a movement.
            return actionValue.index(max(actionValue))
        else:
            pass

class UseMap(gym.core.Wrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,unbiased=False,execute=True,mapPath="fair_map_1v0"):
        super().__init__(env)
        self.first=True
        self.execute=execute
        self.unbiased=unbiased
        self.episodes=0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.map_list = [os.path.join(os.path.join(dir_path,mapPath), path) for path in os.listdir(os.path.join(dir_path,mapPath))]
        max_epsilon = 0.70;
    def GetLabel(self):
        if self.execute:
            return(self.map_list.index(self.map))
        else:
            return 1
    def use_this_map(self,x):
        if self.first:
            self.first=False
            return random.choice(self.map_list)
        if self.unbiased:
            return random.choice(self.map_list)

        def smoothstep(x, lowx=0.0, highx=1.0, lowy=0, highy=1):
            x = (x-lowx) / (highx-lowx)
            if x < 0:
                val = 0
            elif x > 1:
                val = 1
            else:
                val = x * x * (3 - 2 * x)
                return val*(highy-lowy)+lowy
        prob = smoothstep(x, highx=50000, highy=0.70)
        if np.random.random() < prob:
            return random.choice(self.map_list)
        else:
            return None
    def reset(self,**kwargs):
        if not self.execute:
            return self.env.reset(**kwargs)

        self.episodes+=1
        tmp=self.use_this_map(self.episodes)
        if tmp is not None:
            self.map = tmp
        return self.env.reset(custom_board=self.map,**kwargs)


class CTFCentering(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,centering):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.centering = centering
        self.map_size
        self.action_space = spaces.Discrete(len(self.ACTION))
        if centering:

            self.observation_space = spaces.Box(
                low=-255,
                high=255,
                shape=( 2*self.map_size[0]-1, 2*self.map_size[1]-1, 6),  # number of cells
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=-255,
                high=255,
                shape=( self.map_size[0], self.map_size[1], 6),  # number of cells
                dtype='uint8'
            )

    def observation(self, s0):
        padder=[0,0,0,1,0,0]
        #Get list of controlled agents
        agents = self.env.get_team_blue
        if self.centering:

            olx, oly, ch = s0.shape
            H = olx*2-1
            W = oly*2-1
            padder = padder[:ch]
            #padder = [0] * ch; padder[3] = 1
            #if len(padder) >= 10: padder[7] = 1

            cx, cy = (W-1)//2, (H-1)//2
            states = np.zeros([len(agents), H, W, len(padder)])
            states[:,:,:] = np.array(padder)
            for idx, agent in enumerate(agents):
                x, y = agent.get_loc()
                states[idx,max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0
        else:
            olx, oly, ch = s0.shape
            H = olx
            W = oly
            states = np.zeros([len(agents), H, W, len(padder)])
            for idx, agent in enumerate(agents):
                states[idx,:,:,:] = s0

        return states

class CTFObsModifier(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,agents=5,flags=4):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.agents_repr = agents
        self.flags_repr = flags

    def observation(self, s0):
        s0[:,:,:,4] = np.where(s0[:,:,:,4]==1,self.agents_repr,s0[:,:,:,4])
        s0[:,:,:,4] = np.where(s0[:,:,:,4]==-1,-self.agents_repr,s0[:,:,:,4])
        s0[:,:,:,2] = np.where(s0[:,:,:,2]==1,self.flags_repr,s0[:,:,:,2])
        s0[:,:,:,2] = np.where(s0[:,:,:,2]==-1,-self.flags_repr,s0[:,:,:,2])
        return s0


class StateStacking(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,nStates,lstm=False,axis=3):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.nStates = nStates
        self.axis = axis

        self.lstm = lstm
        if self.lstm: #If lstm then the states need to be stacked in a different way.
            self.observation_space = spaces.Box(
                low=-255,
                high=255,
                shape=(nStates, self.map_size[0], self.map_size[1], 6),  # number of cells
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=-255,
                high=255,
                shape=( self.map_size[0], self.map_size[1], 6*nStates),  # number of cells
                dtype='uint8'
            )

    def reset(self, **kwargs):
        s0 = self.env.reset(**kwargs)

        if self.lstm:
            self.stack = [s0] * self.nStates
            return np.stack(self.stack, axis=self.axis)
        else:
            self.stack = [s0] * self.nStates
            return np.concatenate(self.stack, axis=self.axis)

    def observation(self, s0):
        if s0 is None:
            return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(s0)
        self.stack.pop(0)

        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)


class Reset(gym.core.Wrapper):
    """Wrapper used to allow for changing """
    def __init__(self,env, config_path=None, policy_red=None, **kwargs):
        super().__init__(env)
        self.config_path = config_path
        self.policy_red = policy_red
        if isinstance(self.config_path,list):
            self.cur_config = random.choice(self.config_path)
    def reset(self, next_config=False, **kwargs):
        if self.config_path is None:
            return self.env.reset(policy_red=use_this_policy(self.policy_red),**kwargs)
        elif isinstance(self.config_path,list):
            if next_config:
                self.cur_config = random.choice(self.config_path)
            return self.env.reset(config_path=self.cur_config, policy_red=use_this_policy(self.policy_red),**kwargs)
        else:
            return self.env.reset(config_path=self.config_path, policy_red=use_this_policy(self.policy_red),**kwargs)
    def GetEnemyPolicy(self):
        pass
    def GetCurrentConfig(self):
        return self.cur_config


class RewardShape(gym.core.RewardWrapper):
    def __init__(self,env, finalReward=-20, **kwargs):
        super().__init__(env)
        self.was_done=None
        self.nAgents = len(self.get_team_blue)

    def reset(self, **kwargs):
        self.was_done=None
        return self.env.reset(**kwargs)

    def reward(self, reward_raw):
        # reward for episode terminating without winner.
        if (self.run_step > 150) and (self.red_win == self.blue_win):
            reward_raw = -0.5
        elif self.blue_win:
            reward_raw = 1.0
        elif self.red_win:
            reward_raw = -1.0
        #Somewhat removing time dependence in environment.
        if reward_raw == -0.001:
            reward_raw =0.0

        reward = np.ones([self.nAgents])*reward_raw

        return reward.flatten() #* (1-np.array(self.was_done, dtype=int))



class RewardLogging(gym.core.Wrapper):
    def __init__(self,env, **kwargs):
        super().__init__(env)
        if self.multiprocessing == 1:
            self.GLOBAL_RUNNING_R = MovingAverage(400)
            self.win_rate = MovingAverage(400)
            self.red_killed = MovingAverage(400)
        else:
            if 'GLOBAL_RUNNING_R' not in globals():
                global GLOBAL_RUNNING_R
                GLOBAL_RUNNING_R = MovingAverage(400)
            self.GLOBAL_RUNNING_R = GLOBAL_RUNNING_R
            self.win_rate = MovingAverage(400)
            self.red_killed = MovingAverage(400)

    def reset(self, **kwargs):
        self.tracking_r = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)
        self.tracking_r.append(reward)
        return observation, reward, done, info

    def getLogging(self):
        """
        Processes the tracked data of the environment.
        In this case it sums the reward over the entire episode.
        """
        self.win_rate.append(int(self.blue_win))
        self.GLOBAL_RUNNING_R.append(sum(self.tracking_r))
        self.red_killed.append(int(self.red_eliminated))
        finalDict = {"Env Results/TotalReward":self.GLOBAL_RUNNING_R(),
                     "Env Results/WinRate":self.win_rate(),
                     "Env Results/RedKilled":self.red_killed()}
        return finalDict
