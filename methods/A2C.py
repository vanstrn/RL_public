
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .buffer import Trajectory
from .method import Method
import tensorflow as tf
import numpy as np

class A2C(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,HPs):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        self.buffer = Trajectory(depth=5)

        #Placeholders
        self.sess=sess

        self.s = tf.placeholder(dtype=tf.float32, shape=stateShape, name="state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None,1], 'r')


        #These need to be returned in the call function of a tf.keras.Model class.
        self.Model = sharedModel

        out = self.Model(self.s)
        self.acts_prob = out["actor"]
        self.critic = out["critic"]

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('Update_Operation'):
            with tf.name_scope('squared_TD_error'):
                self.td_error = self.r + HPs["Gamma"] * self.v_ - self.critic
                self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval

            with tf.name_scope('train_critic'):
                self.train_op_c = tf.train.AdamOptimizer(HPs["Critic LR"]).minimize(self.loss)

            with tf.name_scope('exp_v'):
                log_prob = tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, actionSize, dtype=tf.float32)
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.name_scope('train_actor'):
                self.train_op_a = tf.train.AdamOptimizer(HPs["Actor LR"]).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)



    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), []   # return a int

    def Update(self, HPs):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        self.ProcessBuffer(HPs)

        #Critic Learning Steps
        # s, s_ = np.vstack(self.buffer[0]),np.vstack(self.buffer[3])

        v_ = self.sess.run(self.critic, {self.s: np.vstack(self.buffer[3])})
        feedDict = {self.s: np.vstack(self.buffer[0]),
                    self.v_: v_,
                    self.r: np.vstack(self.buffer[2]),
                    self.a:np.vstack(self.buffer[1])
                    }

        td_error, _, _, exp_v = self.sess.run([self.td_error, self.train_op_c,self.train_op_a, self.exp_v],feedDict)

        #Clear of reset the buffer.
        self.buffer.clear()

    def ProcessBuffer(self,HPs):
        """Take the buffer and calculate future rewards.
        """
        return

    @property
    def getVars(self):
        return self.Model.getVars
    @property
    def getAParams(self):
        return self.sharedModel.getAParams
    @property
    def getCParams(self):
        return self.sharedModel.getCParams
