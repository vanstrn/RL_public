
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
from utils.utils import MovingAverage

class A2C(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,HPs,nTrajs=1,scope=None):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        self.sess=sess
        self.HPs = HPs
        self.scope=scope

        #Creating appropriate buffer for the method. This stores the data from the episode
        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]

        #Placeholders
        self.s = tf.placeholder(dtype=tf.float32, shape=[None]+stateShape, name="state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None,1], 'r')

        #These need to be returned in the call function of a tf.keras.Model class.
        self.Model = sharedModel
        self.acts_prob, self.critic = self.Model(self.s)

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('Update_Operation'):
            with tf.name_scope('squared_TD_error'):
                self.td_error = self.r + HPs["Gamma"] * self.v_ - self.critic
                self.c_loss = tf.reduce_mean(tf.square(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval

            with tf.name_scope('train_critic'):
                self.train_op_c = tf.train.AdamOptimizer(HPs["Critic LR"]).minimize(self.c_loss)

            with tf.name_scope('exp_v'):
                log_prob = tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, actionSize, dtype=tf.float32)
                self.a_loss = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.name_scope('train_actor'):
                self.train_op_a = tf.train.AdamOptimizer(HPs["Actor LR"]).minimize(-self.a_loss)  # minimize(-exp_v) = maximize(exp_v)

        # Creating moving averages to measure loss.
        self.actorLossMA = MovingAverage(400)
        self.criticLossMA = MovingAverage(400)

    def GetAction(self, state,episode,step):
        """
        Contains the code to run the network based on an input.
        """
        # s = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: state})   # get probabilities for all actions
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions, []   # return a int

    def Update(self, HPs):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """

        #Waiting for there to be some number of samples to use.
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["MinibatchSize"]:
            return

        for traj in range(len(self.buffer)):

            #Estimating the value of at the next state.
            v_ = self.sess.run(self.critic, {self.s: np.vstack(self.buffer[traj][3])})

            #Creating the
            feedDict = {self.s: np.vstack(self.buffer[traj][0]),
                        self.v_: v_,
                        self.r: np.vstack(self.buffer[traj][2]),
                        self.a:np.vstack(self.buffer[traj][1])
                        }

            c_loss, _, _, a_loss = self.sess.run([self.c_loss, self.train_op_c,self.train_op_a, self.a_loss],feedDict)

            self.actorLossMA.append(a_loss)
            self.criticLossMA.append(c_loss)

        #Clear of reset the buffer.
        self.ClearTrajectory()

    def GetStatistics(self):
        dict = {"Training Results/Loss Critic":self.criticLossMA(),
            "Training Results/Loss Actor":self.actorLossMA()
            }
        return dict

    @property
    def getVars(self):
        return self.Model.getVars
    @property
    def getAParams(self):
        return self.sharedModel.getAParams
    @property
    def getCParams(self):
        return self.sharedModel.getCParams
