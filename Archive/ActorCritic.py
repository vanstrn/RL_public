#Actor Critic Implementation Cart-Pole

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
from datetime import datetime
from tensorflow import keras


from DNN import CreateDNN,Create2InDNN
import tensorflow as tf

class AC:
    def __init__(self,env,layerSizes=[24,48]):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.actorModel = CreateDNN(inputShape =self.env.observation_space.shape,
                                outputShape =self.env.action_space.n)
        self.targetActorModel = CreateDNN(inputShape =self.env.observation_space.shape,
                                outputShape =self.env.action_space.n)
        self.criticModel = Create2InDNN(inputShape1 =self.env.observation_space.shape,
                                inputShape2 =(2,))
        self.targetCriticModel = Create2InDNN(inputShape1 =self.env.observation_space.shape,
                                inputShape2 =(2,))

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = .125
        self.gamma = 0.85



    def Remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def Act(self,state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.actorModel.predict(state)[0])

    def Train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target_action = self.targetActorModel.predict(new_state)
            future_reward = self.targetCriticModel.predict([new_state, target_action])
            current_action = self.targetActorModel.predict(state)
            current_reward = self.targetCriticModel.predict([state, current_action])

            advantage = current_action
            advantage[0][action] = reward +self.gamma*future_reward-current_reward
            if not done:
                reward += self.gamma * future_reward

            self.criticModel.fit([state, current_action], [reward], epochs=1, verbose=0)

            self.actorModel.fit(state, advantage, epochs=1, verbose=0)



        #Updating Target Networks
        weights = self.actorModel.get_weights()
        target_weights = self.targetActorModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.targetActorModel.set_weights(target_weights)

        weights = self.criticModel.get_weights()
        target_weights = self.targetCriticModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.targetCriticModel.set_weights(target_weights)


    def Save(self,fileName):
        self.model.save(fileName)

    def Log(self):
        pass

if __name__ == "__main__":
    eps = 1000
    epLen = 500
    render = False
    logging = True

    # env = gym.make("MountainCar-v0")
    if logging:
        logDir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.FileWriter(logDir + "/metrics")

    env = gym.make("CartPole-v0")
    agent = AC(env,layerSizes=[24,48])

    print(env.observation_space.shape)

    for ep in range(eps):
        curState = env.reset()
        curState = curState.reshape(1,4)

        for step in range(epLen):

            action = agent.Act(curState)
            newState, reward, done, _ = env.step(action)
            newState = newState.reshape(1,4)
            agent.Remember(curState, action, reward, newState, done)
            hist = agent.Train()
            agent.Log()
            curState = newState
            if done:
                break

        summary = tf.Summary()
        summary.value.add(tag='Reward',simple_value=step)
        summary.value.add(tag='Epsilon',simple_value=agent.epsilon)
        file_writer.add_summary(summary,ep)
        file_writer.flush()



        print("Episode {} Length: {}".format(ep,step))
