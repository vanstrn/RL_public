
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

from networks.common import NetworkBuilder
from utils.utils import MovingAverage, GetFunction,CreatePath,interval_flag


def _log(val):
    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

class AC(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]

        #Placeholders
        self.sess=sess
        self.HPs = settings["NetworkHPs"]

        self.s = tf.placeholder(dtype=tf.float32, shape=[None]+stateShape, name="state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None,1], 'r')


        #These need to be returned in the call function of a tf.keras.Model class.
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)

        inputs = {"state":self.s}
        out = self.Model(inputs)
        self.acts_prob = out["actor"]
        self.critic = out["critic"]

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('Update_Operation'):
            with tf.name_scope('squared_TD_error'):
                self.td_error = self.r + self.HPs["Gamma"] * self.v_ - self.critic
                self.c_loss = tf.reduce_mean(tf.square(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval

            with tf.name_scope('train_critic'):
                self.c_params = self.Model.GetVariables("Critic")
                self.c_grads = tf.gradients(self.c_loss, self.c_params)
                self.update_c_op = tf.train.AdamOptimizer(self.HPs["Critic LR"]).apply_gradients(zip(self.c_grads, self.c_params))

            with tf.name_scope('exp_v'):
                log_prob = tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, actionSize, dtype=tf.float32)
                self.a_loss = -tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.name_scope('train_actor'):
                self.a_params = self.Model.GetVariables("Actor")
                print(self.a_params)
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.update_a_op = tf.train.AdamOptimizer(self.HPs["Actor LR"]).apply_gradients(zip(self.a_grads, self.a_params))

            self.update_ops=[self.update_c_op,self.update_a_op]

            self.entropy = -tf.reduce_mean(self.acts_prob * _log(self.acts_prob), name='entropy')

            self.logging_ops = [self.a_loss,self.c_loss,self.entropy]
            self.labels = ["Loss Actor","Loss Critic","Entropy"]
            self.logging_MA = [MovingAverage(400) for i in range(len(self.logging_ops))]



    def GetAction(self, state, episode=0, step=0):
        """
        Contains the code to run the network based on an input.
        """
        try:
            s = state[np.newaxis, :]
            probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        except ValueError:
            probs = self.sess.run(self.acts_prob, {self.s: state})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), []   # return a int

    def Update(self, episode=0):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < 1:
            return

        for traj in range(len(self.buffer)):
            v_ = self.sess.run(self.critic, {self.s: np.vstack(self.buffer[traj][3])})
            feedDict = {self.s: np.vstack(self.buffer[traj][0]),
                        self.v_: v_,
                        self.r: np.vstack(self.buffer[traj][2]),
                        self.a:np.vstack(self.buffer[traj][1])
                        }
            out = self.sess.run(self.update_ops+self.logging_ops, feedDict)   # local grads applied to global net.
            logging = out[len(self.update_ops):]

            for i,log in enumerate(logging):
                self.logging_MA[i].append(log)

            #Clear of reset the buffer.
        self.ClearTrajectory()
    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/" + label] = self.logging_MA[i]()
        return dict

    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")



class A2C(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]

        #Placeholders
        self.sess=sess
        self.HPs = settings["NetworkHPs"]

        self.s = tf.placeholder(dtype=tf.float32, shape=[None]+stateShape, name="state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None,1], 'r')


        #These need to be returned in the call function of a tf.keras.Model class.
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)

        inputs = {"state":self.s}
        out = self.Model(inputs)
        self.acts_prob = out["actor"]
        self.critic = out["critic"]

        #Defining Training Operations which will be called in the Update Function.
        with tf.variable_scope('Update_Operation'):
            with tf.name_scope('squared_TD_error'):
                self.td_error = self.r + self.HPs["Gamma"] * self.v_ - self.critic
                self.loss = tf.reduce_mean(tf.square(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval

            with tf.name_scope('train_critic'):
                self.train_op_c = tf.train.AdamOptimizer(self.HPs["Critic LR"]).minimize(self.loss)

            with tf.name_scope('exp_v'):
                log_prob = tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, actionSize, dtype=tf.float32)
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.name_scope('train_actor'):
                self.train_op_a = tf.train.AdamOptimizer(self.HPs["Actor LR"]).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

            self.update_ops=[self.train_op_c,self.train_op_a]

            self.entropy = -tf.reduce_mean(self.acts_prob * _log(self.acts_prob), name='entropy')

            self.logging_ops = [self.exp_v,self.loss,self.entropy]
            self.labels = ["Loss Actor","Loss Critic","Entropy"]
            self.logging_MA = [MovingAverage(400) for i in range(len(self.logging_ops))]



    def GetAction(self, state, episode=0, step=0):
        """
        Contains the code to run the network based on an input.
        """
        try:
            s = state[np.newaxis, :]
            probs,critic = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        except ValueError:
            probs,critic = self.sess.run(self.acts_prob, {self.s: state})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), [critic]   # return a int

    def Update(self, episode=0):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < 1:
            return

        for traj in range(len(self.buffer)):
            td_target,advantage=self.ProcessBuffer(traj)
            v_ = self.sess.run(self.critic, {self.s: np.vstack(self.buffer[traj][3])})
            feedDict = {self.s: np.vstack(self.buffer[traj][0]),
                        self.v_: v_,
                        self.r: np.vstack(self.buffer[traj][2]),
                        self.a:np.vstack(self.buffer[traj][1])
                        }
            out = self.sess.run(self.update_ops+self.logging_ops, feedDict)   # local grads applied to global net.
            logging = out[len(self.update_ops):]

            for i,log in enumerate(logging):
                self.logging_MA[i].append(log)

    def ProcessBuffer(self,traj):
        """
        Process the buffer to calculate td_target.

        Parameters
        ----------
        Model : HPs
            Hyperparameters for training.
        traj : Trajectory
            Data stored by the neural network.

        Returns
        -------
        td_target : list
            List Temporal Difference Target for particular states.
        advantage : list
            List of advantages for particular actions.
        """
        split_loc = [i+1 for i, x in enumerate(self.buffer[traj][4]) if x]

        reward_lists = np.split(self.buffer[traj][2],split_loc)
        value_lists = np.split(self.buffer[traj][5],split_loc)

        td_target=[]; advantage=[]
        for rew,value in zip(reward_lists,value_lists):
            td_target_i, advantage_i = gae(rew.reshape(-1),value.reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
            td_target.extend(td_target_i); advantage.extend( advantage_i)
        return td_target, advantage

            #Clear of reset the buffer.
        self.ClearTrajectory()
    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/" + label] = self.logging_MA[i]()
        return dict

    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")
