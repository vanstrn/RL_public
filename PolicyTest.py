
import gym
import tensorflow as tf
from keras.optimizers import Adam


from DNN import CreateDNN, Create2InDNN

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

    def train(self,):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.train_critic(samples)
        self.train_actor(samples)

    def train_critic(self,samples):
        pass

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
            action =
            new_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            #Handle Training
            #Closing stuff
            current_state = new_state
