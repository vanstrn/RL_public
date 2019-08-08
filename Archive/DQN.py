import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
from datetime import datetime
from tensorflow import keras


from DNN import CreateDNN
import tensorflow as tf

class DQN:
    def __init__(self,env,layerSizes=[24,48]):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.model = CreateDNN( inputShape  =self.env.observation_space.shape,
                                outputShape =self.env.action_space.n,
                                layerSizes=layerSizes)
        self.targetModel = CreateDNN( inputShape  =self.env.observation_space.shape,
                                outputShape =self.env.action_space.n,
                                layerSizes=layerSizes)

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.10
        self.tau = .10
        self.gamma = 0.85



    def Remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def Act(self,state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def Train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.targetModel.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.targetModel.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0,)

        weights = self.model.get_weights()
        target_weights = self.targetModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.targetModel.set_weights(target_weights)


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
    agent = DQN(env,layerSizes=[100,])

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
