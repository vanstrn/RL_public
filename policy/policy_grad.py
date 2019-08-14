"""Patrolling agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

"""

import numpy as np
import pickle
import threading
import multiprocessing as mp
import sys
import time


class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
        patrol: Private method to control a single unit.
    """
    
    def __init__(self, free_map, agent_list):
        """Constuctor for policy class.
        
        Patrolling policy provides the actions for the team of units that
        command units to approach the boarder between friendly and enemy
        zones and patrol along it.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        # Hyperparameters
        self.H = 400  # number of hidden layer neurons
        self.BATCH_SIZE = 10  # every how many episodes to do a param update?
        self.LEARNING_Rate = 1e-4
        self.GAMMA = 0.99  # discount factor for reward
        self.DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.RESUME = True  # RESUME from previous checkpoint
        self.RENDER = True
        self.MULTI_PROC = True  # run in parallel (if using multi core machine)
        
        # model initialization
        self.D = 20 * 20  # input dimensionality
        self.K = 5  # output dimensionality

        try:
            self.model = pickle.load(open('save.p', 'rb'))
        except:
            self.model = {}
            self.model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
            self.model['W2'] = np.random.randn(K, H) / np.sqrt(H)


        self.free_map = free_map 
        
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
        
        
        
        return action_out

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def prepro(I):
        """ prepro frame into (20x20) 1D float vector """
        return I.astype(np.float).ravel()

    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self,x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state


    def policy_backward(self, eph, epdlogp, epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).T  # .ravel()
        dh = np.dot(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}


    def train_model(self, l, episode_number, running_reward, pos=0, first_time=True):
        """
        Runs policy gradient (can be run in parallel)
    
        :param l: lock to acquire when training/saving the model
        :return: void
        """
    
        grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}  # update buffers that add up gradients over a batch
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory
    
        env = gym.make("cap-v0")
        observation = env.reset(mode="sandbox")
        prev_x = None  # used in computing the difference frame
        xs, hs, dlogps, drs = [], [], [], []
        reward_sum = 0
        episode_number_loc = 0
        while True:
            if RENDER and episode_number_loc % 100 == 0 and pos == 0:
                env.render(mode="env")  # env.render()
    
            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(env._env)
            x = cur_x  # - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x
    
            # forward the policy network and sample an action from the returned probability
            aprob, h = policy_forward(x)
            aprob = aprob / np.sum(aprob)
            action = np.random.choice(5, p=aprob)
    
            # record various intermediates (needed later for backprop)
            xs.append(x)  # observation
            hs.append(h)  # hidden state
            y = np.zeros(5)
            y[action] = 1  # a "fake label"
            dlogps.append(
                y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    
            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            if reward > 0 or done:
                reward = reward
            else:
                reward = 0.0
            reward_sum += reward
    
            drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
    
            if done:  # an episode finished
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []  # reset array memory
    
                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                # print('rew',epr)
                # print('epr',discounted_epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                # if np.std(discounted_epr) is not 0:
                discounted_epr /= (np.std(discounted_epr) + 1e-5)
                # print('norm_epr',discounted_epr)
                episode_number_loc += 1
    
                with l:
                    episode_number.value += 1
                    epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
                    grad = policy_backward(eph, epdlogp, epx)
                    for k in model:
                        grad_buffer[k] += grad[k]  # accumulate grad over batch
    
                    # perform rmsprop parameter update every BATCH_SIZE episodes
                    if episode_number.value % BATCH_SIZE == 0:
                        for k, v in model.items():  # iteritems():
                            g = grad_buffer[k]  # gradient
                            rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g ** 2
                            if np.isnan(rmsprop_cache[k]).any():
                                print('NaN in gradient')
                                continue
                            model[k] += LEARNING_Rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
    
                    # boring book-keeping
                    if first_time:
                        running_reward.value = reward_sum
                        first_time = False
                    else:
                        running_reward.value = running_reward.value * 0.99 + reward_sum * 0.01
                    print(
                        'proc: %i, episode: %i, episode reward total was %f, running mean: %f' % (
                        pos, episode_number.value, reward_sum, running_reward.value))
                    if episode_number.value % 100 == 0:
                        pickle.dump(model, open('save.p', 'wb'))
                    reward_sum = 0
                    observation = env.reset(mode="sandbox")
                    prev_x = None


