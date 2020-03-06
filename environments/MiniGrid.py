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

        self.observation_space.spaces["image"] = spaces.Box(
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

        return {
            'mission': obs['mission'],
            'image': full_grid[:,:,:2]
        }

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
class StackedFrames_v2(gym.core.ObservationWrapper):
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
        return {
            'mission': obs['mission'],
            'image': np.squeeze(np.stack(self.stack,2))
        }


def RewardShape(s1,reward_raw,done_raw,env,envSettings,sess):
    # if not done_raw: reward_raw += -0.01
    # else: reward_raw += 0.5
    if reward_raw != 0:
        reward=1
    else:
        reward=0
    return reward, np.array(done_raw)

def Bootstrap(env,settings,envSettings,sess):
    s0 = env.reset()
    loggingDict = {"tracking_r":[[] for _ in range(settings["NumberENV"])]}
    return s0, loggingDict

def Bootstrap_Asynch(env,settings,envSettings,sess):
    s0 = env.reset()
    loggingDict = {"tracking_r":[[] for _ in range(1)]}
    return s0, loggingDict

def StateProcessing(s0,env,envSettings,sess):
    return s0["image"]

def Starting(settings,envSettings,sess):
    def make_env():
        return lambda: gym.make(envSettings["EnvName"],)
    envs = [make_env() for i in range(settings["NumberENV"])]
    envs = SubprocVecEnv(envs)
    envs.remotes[0].send(('_get_spaces', None))
    N_F, N_A = envs.remotes[0].recv()
    nTrajs = settings["NumberENV"]

    return envs, N_F[0], N_A,nTrajs

def StartingSingle(settings,envSettings,sess):
    if envSettings["EnvName"] == "MiniGrid-Empty-8x8-v0":
        env = gym.make(envSettings["EnvName"],)
    elif envSettings["EnvName"]=="MiniGrid-FourRooms-v0":
        env = gym.make(envSettings["EnvName"],goal_pos=envSettings["GoalLocation"],agent_pos=envSettings["AgentLocation"])
    else:
        env = gym.make(envSettings["EnvName"],)
    env.max_steps=settings["EnvHPs"]["MAX_EP_STEPS"]
    env = DiscreteAction(env)
    env = FullyObsWrapper_v2(env)
    numberFeatures = env.observation_space["image"].shape
    numberActions = env.action_space.n
    # numberActions=3

    return env, list(numberFeatures), numberActions, 1

def StartingSinglePixel(settings,envSettings,sess):
    if envSettings["EnvName"] == "MiniGrid-Empty-8x8-v0":
        env = gym.make(envSettings["EnvName"],)
    elif envSettings["EnvName"]=="MiniGrid-FourRooms-v0":
        env = gym.make(envSettings["EnvName"],goal_pos=envSettings["GoalLocation"],agent_pos=envSettings["AgentLocation"])
    else:
        env = gym.make(envSettings["EnvName"],)
    env.max_steps=settings["EnvHPs"]["MAX_EP_STEPS"]
    env = DiscreteAction(env)
    env = RGBImgObsWrapper(env,tile_size=6)
    numberFeatures = env.observation_space["image"].shape
    numberActions = env.action_space.n
    # numberActions=3

    return env, list(numberFeatures), numberActions, 1

def StartingSingleStacked(settings,envSettings,sess):
    if envSettings["EnvName"] == "MiniGrid-Empty-8x8-v0":
        env = gym.make(envSettings["EnvName"],)
    elif envSettings["EnvName"]=="MiniGrid-FourRooms-v0":
        env = gym.make(envSettings["EnvName"],goal_pos=envSettings["GoalLocation"],agent_pos=envSettings["AgentLocation"])
    else:
        env = gym.make(envSettings["EnvName"],)
    env.max_steps=settings["EnvHPs"]["MAX_EP_STEPS"]
    env = DiscreteAction(env)
    env = StackedFrames_v1(env)
    numberFeatures = env.observation_space["image"].shape
    numberActions = env.action_space.n
    # numberActions=3

    return env, list(numberFeatures), numberActions, 1

def Logging(loggingDict,s1,r,done,env,envSettings,sess):
    try:
        for i,envR in enumerate(r):
            if not done[i]: loggingDict["tracking_r"][i].append(envR)
    except: loggingDict["tracking_r"][0].append(r)
    return loggingDict

def Closing(loggingDict,env,settings,envSetting,sess,progbar,GLOBAL_RUNNING_R=None,GLOBAL_EP_LEN=None):
    if GLOBAL_RUNNING_R is not None:
        for i in range(len(loggingDict["tracking_r"])):
            GLOBAL_RUNNING_R.append(sum(loggingDict["tracking_r"][i]))
            GLOBAL_EP_LEN.append(len(loggingDict["tracking_r"][i]))
        finalDict = {"Training Results/Reward":GLOBAL_RUNNING_R(),
                    "Training Results/Episode Length": GLOBAL_EP_LEN()}
        return finalDict


    else:
        for i in range(settings["NumberENV"]):
            ep_rs_sum = sum(loggingDict["tracking_r"][i])
            if 'running_reward' not in globals():
                global running_reward
                running_reward = MovingAverage(100)
            if 'episode_length' not in globals():
                global episode_length
                episode_length =  MovingAverage(100)
            running_reward.append(ep_rs_sum)
            episode_length.append(len(loggingDict["tracking_r"][i]))

        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_step")
        progbar.update(sess.run(global_step)[0],values=[("Reward",running_reward)])
        # print("episode:", sess.run(global_step), "  running reward:", int(running_reward),"  reward:",int(ep_rs_sum))

        finalDict = {"Training Results/Reward":running_reward(),
                     "Training Results/Episode Length": episode_length()}
        return finalDict
