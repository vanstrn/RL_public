'''Implementation of Advantage Actor Critic
Author: Neale Van Stralen'''

import tensorflow as tf
import numpy as np
import gym

class ActorCriticNet():
    def __init__(self,model):
        self.model = model

    def GetAction(self,s):
        s = s[np.newaxis, :]
        logits = model.sess.run(model.acts_prob, {model.s: s})
        action = np.random.choice(np.arange(logits.shape[1]), p=logits.ravel())
        return action, logits

    def Learn(self,s0,a,r,s1):
        td = model.UpdateModel(s0,a,r,s1)


class SeparatedModel():
    def __init__(self,sess,input,output,lr_c,lr_a ):
        #Creating IO Placeholders
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, input], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        GAMMA = 0.9
        #Creating Critic Network
        with tf.variable_scope('Actor'):
            l1_a = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1_a'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1_a,
                units=output,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        #Creating Actor Network
        with tf.variable_scope('Critic'):
            l1_c = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1_c'
            )

            self.v = tf.layers.dense(
                inputs=l1_c,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )


        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss


        with tf.variable_scope('train_actor'):
            self.a_train_op = tf.train.AdamOptimizer(lr_a).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
        with tf.variable_scope('train_critic'):
            self.c_train_op = tf.train.AdamOptimizer(lr_c).minimize(self.loss)  # minimize(-exp_v) = maximize(exp_v)




    def UpdateModel(self,s0,a,r,s1):
        s, s_ = s0[np.newaxis, :], s1[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})

        feed_dict = {self.s: s, self.a: a, self.v_: v_, self.r: r}
        _, _ = self.sess.run([self.c_train_op, self.a_train_op], feed_dict)

class SharedModel():
    def __init__(self,sess,input,output,lr_c,lr_a ):
        #Creating IO Placeholders
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, input], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        GAMMA = 0.9
        #Creating Critic Network
        with tf.variable_scope('Shared'):
            l1 = tf.layers.dense(
            inputs=self.s,
            units=20,    # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
            )
        with tf.variable_scope('Actor'):

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=output,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        #Creating Actor Network
        with tf.variable_scope('Critic'):

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )


        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss


        with tf.variable_scope('train_actor'):
            self.a_train_op = tf.train.AdamOptimizer(lr_a).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
        with tf.variable_scope('train_critic'):
            self.c_train_op = tf.train.AdamOptimizer(lr_c).minimize(self.loss)  # minimize(-exp_v) = maximize(exp_v)




    def UpdateModel(self,s0,a,r,s1):
        s, s_ = s0[np.newaxis, :], s1[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})

        feed_dict = {self.s: s, self.a: a, self.v_: v_, self.r: r}
        _, _ = self.sess.run([self.c_train_op, self.a_train_op], feed_dict)



if __name__ == "__main__":

    #Setting up the environment
    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    #Setting up the
    sess = tf.Session()
    # model = SeparatedModel(sess, N_F,N_A,0.001,0.0001)
    model = SharedModel(sess, N_F,N_A,0.001,0.0001)
    AC = ActorCriticNet(model)
    sess.run(tf.global_variables_initializer())

    MAX_EP_STEPS = 1000
    DISPLAY_REWARD_THRESHOLD=200
    #Training Loop
    for i in range(10000):

        s0 = env.reset()
        track_r = []

        for j in range(MAX_EP_STEPS):
            action, logits = AC.GetAction(s0)

            s1,r,done,_ = env.step(action)
            if done: r = -20
            track_r.append(r)

            #Update Step
            AC.Learn(s0,action,r,s1)


            s0 = s1
            if done or j >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i, "  reward:", int(running_reward))
                break
