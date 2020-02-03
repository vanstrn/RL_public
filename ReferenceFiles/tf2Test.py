import tensorflow as tf
import gym
import numpy as np

# gamma: discount rate
def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make("CartPole-v0")
n_episodes = 10000
scores=[]
update_every=5

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim = 4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

gradBuffer = model.trainable_variables
for ix,grad in enumerate(gradBuffer):
  gradBuffer[ix] = grad * 0

for e in range(n_episodes):
  # reset the enviroment
  s = env.reset()
  ep_memory = []
  ep_score = 0
  done = False
  while not done:
    s = s.reshape([1,4])
    with tf.GradientTape() as tape:
      #forward pass
      logits = model(s)
      a_dist = logits.numpy()
      # Choose random action with p = action dist
      a = np.random.choice(a_dist[0],p=a_dist[0])
      a = np.argmax(a_dist == a)
      loss = compute_loss([a], logits)
    grads = tape.gradient(loss, model.trainable_variables)
    # make the choosen action
    s, r, done, _ = env.step(a)
    ep_score +=r
    if done: r-=10 # small trick to make training faster

    ep_memory.append([grads,r])
  scores.append(ep_score)
  # Discound the rewards
  ep_memory = np.array(ep_memory)
  ep_memory[:,1] = discount_rewards(ep_memory[:,1])

  for grads, r in ep_memory:
    for ix,grad in enumerate(grads):
      gradBuffer[ix] += grad * r

  if e % update_every == 0:
    optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

  if e % 100 == 0:
    print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))
