
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
import tensorflow as tf
import numpy as np
class A2C(Method):

    def __init__(self,actorNetwork,criticNetwork,sess,stateShape,actionSize,lr_c=1E-4,lr_a=1E-3,GAMMA = 0.9):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.sess=sess
        self.s = tf.placeholder(dtype=tf.float32, shape=stateShape, name="state")
        self.a = tf.placeholder(tf.int32, None, "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        #These need to be returned in the call function of a tf.keras.Model class.
        self.CriticModel = criticNetwork
        self.ActorModel = actorNetwork

        self.critic = self.CriticModel(self.s)
        self.acts_prob = self.ActorModel(self.s)

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.critic
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('train_critic'):
            self.train_op_c = tf.train.AdamOptimizer(lr_c).minimize(self.loss)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train_actor'):
            self.train_op_a = tf.train.AdamOptimizer(lr_a).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)



    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def Learn(self, s0,a,r,s1):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        #Critic Learning Steps
        s, s_ = s0[np.newaxis, :], s1[np.newaxis, :]
        # print("here1-----------")
        #
        # v_ = self.sess.run(self.critic, {self.s: s_})
        # feedDict = {self.s: s, self.v_: v_, self.r: r}
        # td_error, _ = self.sess.run([self.td_error, self.train_op_c],feedDict)
        # print(td_error)
        # print(td_error.dtype)
        # #Actor Learning Steps
        # feedDict = {self.s: s, self.a: a, self.td_error: td_error,}
        # _, exp_v = self.sess.run([self.train_op_a, self.exp_v], feedDict)


        v_ = self.sess.run(self.critic, {self.s: s_})
        feedDict = {self.s: s, self.v_: v_, self.r: r,self.a:a}
        td_error, _, _, exp_v = self.sess.run([self.td_error, self.train_op_c,self.train_op_a, self.exp_v],feedDict)

        return exp_v

    def SaveStatistics(self,saver):
        """
        Contains the code to save internal information of the Neural Network.
        """
        raise NotImplementedError
    def InitializeVariablesFromFile(self,saver, model_path):
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")
    @property
    def getVars(self):
        return self.CriticModel.getVars + self.ActorModel.getVars

class A2C_s(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,lr_c=1E-4,lr_a=1E-3,GAMMA = 0.9):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.sess=sess

        self.s = tf.placeholder(dtype=tf.float32, shape=stateShape, name="state")
        self.a = tf.placeholder(tf.int32, None, "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        #These need to be returned in the call function of a tf.keras.Model class.
        self.Model = sharedModel

        self.acts_prob,self.critic = self.Model(self.s)

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.critic
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('train_critic'):
            self.train_op_c = tf.train.AdamOptimizer(lr_c).minimize(self.loss)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train_actor'):
            self.train_op_a = tf.train.AdamOptimizer(lr_a).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)



    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def Learn(self, s0,a,r,s1):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        #Critic Learning Steps
        s, s_ = s0[np.newaxis, :], s1[np.newaxis, :]
        # print("here1-----------")
        #
        # v_ = self.sess.run(self.critic, {self.s: s_})
        # feedDict = {self.s: s, self.v_: v_, self.r: r}
        # td_error, _ = self.sess.run([self.td_error, self.train_op_c],feedDict)
        # print(td_error)
        # print(td_error.dtype)
        # #Actor Learning Steps
        # feedDict = {self.s: s, self.a: a, self.td_error: td_error,}
        # _, exp_v = self.sess.run([self.train_op_a, self.exp_v], feedDict)


        v_ = self.sess.run(self.critic, {self.s: s_})
        feedDict = {self.s: s, self.v_: v_, self.r: r,self.a:a}
        td_error, _, _, exp_v = self.sess.run([self.td_error, self.train_op_c,self.train_op_a, self.exp_v],feedDict)

        return exp_v

    def SaveStatistics(self,saver):
        """
        Contains the code to save internal information of the Neural Network.
        """
        raise NotImplementedError

    def InitializeVariablesFromFile(self,saver, model_path):
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

    @property
    def getVars(self):
        return self.Model.getVars
    @property
    def getAParams(self):
        return self.sharedModel.getAParams
    @property
    def getCParams(self):
        return self.sharedModel.getCParams
