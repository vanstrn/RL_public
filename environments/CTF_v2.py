import gym
import tensorflow as tf
from utils.multiprocessing import SubprocVecEnv
import numpy as np
import policy
import time

def use_this_policy(policyName=None):
    if policyName is None:
        heur_policy_list = [policy.Patrol, policy.Roomba, policy.Defense, policy.Random, policy.AStar]
        heur_weight = [1,1,1,1,1]
        heur_weight = np.array(heur_weight) / sum(heur_weight)
        return np.random.choice(heur_policy_list, p=heur_weight)
    elif policyName == "Roomba":
        return policy.Roomba
    elif policyName == "Patrol":
        return policy.Patrol
    elif policyName == "Defense":
        return policy.Defense
    elif policyName == "AStar":
        return policy.AStar
    elif policyName == "Random":
        return policy.Random



class CTFCentering(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env,centering):
        super().__init__(env)
        nAgents = len(self.env.get_team_blue)
        self.centering = centering
        if centering:

            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(4, 39, 39, 6),  # number of cells
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(4, 20, 20, 6),  # number of cells
                dtype='uint8'
            )

    def observation(self, s0):
        padder=[0,0,0,1,0,0]
        #Get list of controlled agents
        agents = self.env.get_team_blue()
        if self.centering:

            envs, olx, oly, ch = s0.shape
            H = olx*2-1
            W = oly*2-1
            padder = padder[:ch]
            #padder = [0] * ch; padder[3] = 1
            #if len(padder) >= 10: padder[7] = 1

            cx, cy = (W-1)//2, (H-1)//2
            states = np.zeros([len(agents[0])*envs, H, W, len(padder)])
            states[:,:,:] = np.array(padder)
            for idx, self.env in enumerate(agents):
                for idx2, agent in enumerate(self.env):
                    x, y = agent.get_loc()
                    states[idx*len(self.env)+idx2,max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0[idx]
        else:
            envs, olx, oly, ch = s0.shape
            H = olx
            W = oly
            states = np.zeros([len(agents[0])*envs, H, W, len(padder)])
            for idx, self.env in enumerate(agents):
                for idx2, agent in enumerate(self.env):
                    states[idx*len(self.env)+idx2,::,:] = s0[idx]

        return states


class Reset(gym.core.Wrapper):
    def __init__(self,env, config_path, policy_red=None, **kwargs):
        super().__init__(env)
        self.config_path = config_path
        self.policy_red = policy_red
    def reset(self, **kwargs):
        action = randint(0,self.action_space.n)
        return self.env.reset(config_path=slef.config_path, policy_red=use_this_policy(self.policy_red)**kwargs)


class RewardShape(Wrapper):
    def __init__(self,env, finalReward=-20, **kwargs):
        super().__init__(env)
        self.was_done=None

    def reset(self, **kwargs):
        self.was_done=None
        return self.env.reset(**kwargs)

    def reward(self, reward_raw):

        done = np.zeros([len(done_raw),4])
        for idx,d in enumerate(done_raw):
            done[idx,:] = np.ones([4])*d
        done = done.flatten()

        #Processing the reward recording zero if the environment is done.
        self.was_done = done

        if done.all():
            self.was_done = np.ones([len(done_raw)*4])*False


        reward = np.ones([len(reward_raw),4])
        for idx,env_reward in enumerate(reward_raw):
            if not self.was_done[idx*4]:
                reward[idx,:]=reward[idx,:]*env_reward
            else:
                reward[idx,:]=reward[idx,:]*0
        #     for idx,rew in enumerate(r):
        #         if done[idx]:
        #             reward[idx,:] = np.ones([4])*rew
        # else:
        #     for idx,rew in enumerate(r):
        #         if not self.was_done[idx]:
        #             reward[idx,:] = np.ones([4])*rew
        #         else:
        #             reward[idx,:] = np.ones([4])*0
        #     self.was_done = done1
        # for rew in r:
        #     if rew >= 0:
        #         print("Won the game")

        reward = reward.flatten() #* (1-np.array(self.was_done, dtype=int))

        return reward
