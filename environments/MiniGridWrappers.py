import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
from utils.utils import MovingAverage
import numpy as np
from gym_minigrid.wrappers import *

class DiscreteAction(gym.core.Wrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Discrete(4)

    def step(self,action):
        env = self.unwrapped
        if action == 0: # Go Up
            env.agent_dir = 3
        if action == 1: # Go Right
            env.agent_dir = 0
        if action == 2: # Go Down
            env.agent_dir = 1
        if action == 3: # Go Left
            env.agent_dir = 2
        obs, reward, done, info = env.step(2)
        return obs, reward, done, info

class FullyObsWrapper_v2(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 2),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return np.expand_dims(full_grid[:,:,:2],0)

class FullyObsWrapper_v5(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 2),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        flagX,flagY = np.unravel_index(np.argmax(full_grid[:,:,0], axis=None), full_grid[:,:,0].shape)

        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            OBJECT_TO_IDX['agent']
        ])
        # changing flag
        full_grid[flagX,flagY] = np.array([
            OBJECT_TO_IDX['goal'],
            OBJECT_TO_IDX['goal'],
            OBJECT_TO_IDX['goal']
        ])

        ret = full_grid[:,:,np.r_[0,2]]

        # print(ret[:,:,0])
        # print(ret[:,:,1])

        return np.expand_dims(full_grid[:,:,np.r_[0,2]],0)

class RewardWrapper(gym.core.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward,done), done, info

    def reward(self,reward,done):
        if done and reward !=0:
            return 1
        else: return 0

class FullyObsWrapper_v3(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 1),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid[:,:,:1]
        }
class FullyObsWrapper_v4(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 1),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid /= np.maximum(full_grid)
        return {
            'mission': obs['mission'],
            'image': full_grid[:,:,:1]
        }

class Grayscale(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 1),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        grayscale =  (0.3 * rgb_img[0]) + (0.59 * rgb_img[1]) + (0.11 * rgb_img[2])

        return {
            'mission': obs['mission'],
            'image': grayscale
        }

class StackedFrames_v1(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,stackedFrames=4):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, stackedFrames),  # number of cells
            dtype='uint8'
        )
        self.stackedFrames = stackedFrames

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid /= np.maximum(full_grid)
        if env.step_count == 0:
            self.stack = [full_grid[:,:,:1]] * self.stackedFrames
        else:
            self.stack.append(full_grid[:,:,:1])
            self.stack.pop(0)

        np.squeeze(np.stack(self.stack,2))
        return {
            'mission': obs['mission'],
            'image': np.squeeze(np.stack(self.stack,2))
        }
class GrayscaleStackedFrames(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,stackedFrames=4):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, stackedFrames),  # number of cells
            dtype='uint8'
        )
        self.stackedFrames = stackedFrames

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        grayscale =  (0.3 * rgb_img[0]) + (0.59 * rgb_img[1]) + (0.11 * rgb_img[2])

        grayscale /= np.maximum(grayscale)
        if env.step_count == 0:
            self.stack = [grayscale[:,:,:1]] * self.stackedFrames
        else:
            self.stack.append(grayscale[:,:,:1])
            self.stack.pop(0)
        return {
            'mission': obs['mission'],
            'image': np.squeeze(np.stack(self.stack,2))
        }


def SmoothOption(option, gamma =0.9):
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

class SampleConstructor(gym.core.Wrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

    def ConstructAllSamples(self):
        """Constructing All Samples into a Q table. """
        #### Getting Background Grid
        grid = self.grid.encode()
        flagX,flagY = np.unravel_index(np.argmax(grid[:,:,0], axis=None), grid[:,:,0].shape)
        grid[flagX,flagY] = np.array([
                    OBJECT_TO_IDX['goal'],
                    OBJECT_TO_IDX['goal'],
                    OBJECT_TO_IDX['goal']
                ])

        stacked_grids = np.repeat(np.expand_dims(grid,0), grid.shape[0]*grid.shape[1],0)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j,1] == 5:
                    pass
                stacked_grids[i*grid.shape[1]+j,i,j,0] = 10
                stacked_grids[i*grid.shape[1]+j,i,j,1] = 10
                stacked_grids[i*grid.shape[1]+j,i,j,2] = 10
        return stacked_grids[:,:,:,np.r_[0,2]]

    def ReformatSamples(self,values):
        """Formating Data back into a Q Table. """
        grid = self.grid.encode()
        value_map = np.reshape(values,grid.shape[:2])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j,0] == 2:
                    value_map[i,j] = 0.0
        smoothed_value_map = SmoothOption(value_map)
        smoothed_value_map_inv = -smoothed_value_map
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j,0] == 2:
                    # smoothed_value_map[i,j] = -999
                    # smoothed_value_map_inv[i,j] = -999
                    pass
        return smoothed_value_map, smoothed_value_map_inv

    def UseSubpolicy(self,s,subpolicy):
        #Extracting location of agent.

        locX,locY = np.unravel_index(np.argmax(s[0,:,:,0], axis=None), s[0,:,:,0].shape)
        #Getting Value of all adjacent policies. Ignoring location of the walls.
        actionValue = []
        if [int(locX),int(locY+1),0] == 2:
            actionValue.append(-999)
        else:
            actionValue.append(subpolicy[int(locX),  int(locY+1)  ]) # Go Up
        if [int(locX+1),int(locY),0] == 2:
            actionValue.append(-999)
        else:
            actionValue.append(subpolicy[int(locX+1),int(locY)    ]) # Go Right
        if [int(locX),int(locY-1),0] == 2:
            actionValue.append(-999)
        else:
            actionValue.append(subpolicy[int(locX),  int(locY-1)  ]) # Go Down
        if [int(locX-1),int(locY),0] == 2:
            actionValue.append(-999)
        else:
            actionValue.append(subpolicy[int(locX-1),int(locY)    ]) # Go Left

        #Selecting Action with Highest Value. Will always take a movement.
        return actionValue.index(max(actionValue))
