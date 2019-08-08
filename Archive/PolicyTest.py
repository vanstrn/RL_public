
import gym
import tensorflow as tf
from keras.optimizers import Adam
import random
import numpy as np
from DNN import CreateDNN, Create2InDNN
from collections import deque

class ActorCritic:
    def __init__(self,inputSpace,actionSpace):
        #Initializing variables
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125

        #Initialize Actor
        self.actor = CreateDNN(inputSpace,actionSpace,Adam(lr=0.001))

        #initialize Critic
        self.critic = Create2InDNN(inputSpace,actionSpace,Adam(lr=0.001))

        #Initialize Memory for training
        self.memory = deque(maxlen=2000)
    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
    def train(self,):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.train_critic(samples)
        self.train_actor(samples)

    def train_critic(self,samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.actor.predict(tf.convert_to_tensor(new_state[None, :],dtype=tf.float32),batch_size=None,steps=1)
                future_reward = self.critic.predict(
                    [tf.convert_to_tensor(new_state[None, :],dtype=tf.float32), tf.convert_to_tensor(target_action[None, :],dtype=tf.float32)],batch_size=None,steps=1)[0][0]
                reward += self.gamma * future_reward
            self.critic.fit([cur_state, action], reward, verbose=0)

    def train_actor(self,samples):
        pass

    def act(self):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            action = np.random.choice(self.action_size, p=probs.numpy()[0])
            self.actor.predict(cur_state)


if __name__ == "__main__":
    render = True
    sess = tf.Session()
    env = gym.make('CartPole-v0')

    ac = ActorCritic(env.observation_space.shape,(env.action_space.n,))

    episodes = 5
    for i in range(episodes):
        current_state = env.reset()
        done = False
        ep_reward = 0.
        ep_steps = 0
        ep_loss = 0


        while not done:

            #Get Action and feed into environment
            # print(current_state)
            logits = ac.actor.predict(tf.convert_to_tensor(current_state[None, :],dtype=tf.float32),batch_size=None,steps=1)
            probs = np.squeeze(tf.nn.softmax(logits).eval(session = sess))
            action = np.random.choice(env.action_space.n, p=probs)

            #Inputting action into to the environment
            new_state, reward, done, _ = env.step(action)
            if render:
                env.render()

            ac.remember(current_state, action, reward, new_state, done)
            ac.train()
            #Handle Training
            #Closing stuff
            current_state = new_state
