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

        return full_grid[:,:,:2]

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
