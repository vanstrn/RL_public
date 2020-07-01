
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
from .AdvantageEstimator import gae
import tensorflow as tf
import numpy as np
from utils.utils import MovingAverage
from utils.record import Record
import random

from networks.common import NetworkBuilder

class DQN(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders

        self.sess=sess
        self.scope="DQN"
        self.HPs = settings["NetworkHPs"]
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)

        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]
        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(self.scope):
                if len(stateShape) == 4:
                    self.states_ = tf.placeholder(shape=[None]+stateShape[1:4], dtype=tf.float32, name='states')
                    self.next_states_ = tf.placeholder(shape=[None]+stateShape[1:4], dtype=tf.float32, name='next_states')
                else:
                    self.states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='states')
                    self.next_states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='next_states')
                self.actions_ = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_hold')
                self.rewards_ = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards_hold')
                self.done_ = tf.placeholder(shape=[None], dtype=tf.float32, name='done_hold')

                input = {"state":self.states_}
                out = self.Model(input)
                self.q = out["Q"]

                out2 = self.Model({"state":self.next_states_})
                q_next = out2["Q"]

                with tf.name_scope('current_Q'):
                    oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                    curr_q = tf.reduce_sum(tf.multiply(self.q, oh_action), axis=-1) # [?, num_agent]

                with tf.name_scope('target_Q'):
                    max_next_q = tf.reduce_max(q_next, axis=-1)
                    td_target = self.rewards_  + self.HPs["Gamma"] * max_next_q
                    # td_target = self.rewards_  + self.HPs["Gamma"] * max_next_q * (1. - self.done_)

                with tf.name_scope('td_error'):
                    loss = tf.keras.losses.MSE(td_target, curr_q)
                    softmax_q = tf.nn.softmax(curr_q)
                    self.entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q+ 1e-5))
                    self.loss=total_loss = loss + self.HPs["EntropyBeta"] * self.entropy

                if self.HPs["Optimizer"] == "Adam":
                    self.optimizer = tf.keras.optimizers.Adam(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "RMS":
                    self.optimizer = tf.keras.optimizers.RMSProp(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adagrad":
                    self.optimizer = tf.keras.optimizers.Adagrad(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adadelta":
                    self.optimizer = tf.keras.optimizers.Adadelta(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adamax":
                    self.optimizer = tf.keras.optimizers.Adamax(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Nadam":
                    self.optimizer = tf.keras.optimizers.Nadam(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "SGD":
                    self.optimizer = tf.keras.optimizers.SGD(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Amsgrad":
                    self.optimizer = tf.keras.optimizers.Nadam(self.HPs["LR"],amsgrad=True)
                else:
                    print("Not selected a proper Optimizer")
                    exit()
                self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

                self.gradients = self.optimizer.get_gradients(total_loss, self.params)
                self.update_op = self.optimizer.apply_gradients(zip(self.gradients, self.params))

                self.grads=[self.gradients]
                self.losses=[self.loss]
                self.update_ops=[self.update_op]

        self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
        self.loss_MA = [MovingAverage(400) for i in range(len(self.losses))]
        self.entropy_MA = MovingAverage(400)
        self.labels = ["Critic"]

    def GetAction(self, state,episode,step):
        """
        Contains the code to run the network based on an input.
        """
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        q = self.sess.run(self.q, {self.states_: state})
        if "Exploration" in self.HPs:
            if self.HPs["Exploration"]=="EGreedy":
                prob = 0.1 + 0.9*(np.exp(-episode/self.HPs["ExplorationDecay"]))
                if random.uniform(0, 1) < prob:
                    actions = random.randint(0,4)
                else:
                    actions = np.argmax(q, axis=-1)
        else:
            actions = np.argmax(q, axis=-1)
        return actions ,[]  # return a int and extra data that needs to be fed to buffer.

    def Update(self,episode=0):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Process the data from the buffer
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        for traj in range(len(self.buffer)):
            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1
            s = np.array_split( self.buffer[traj][0], batches)
            a_his = np.array_split( np.asarray(self.buffer[traj][1]).reshape(-1), batches)
            r = np.array_split( np.asarray(self.buffer[traj][2]).reshape(-1), batches)
            s_next = np.array_split( self.buffer[traj][3], batches)
            done = np.array_split( self.buffer[traj][4], batches)

            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):
                #Create a feedDict from the buffer
                    feedDict = {
                        self.states_ : np.squeeze(np.asarray(s[i])),
                        self.next_states_ : np.squeeze(np.asarray(s_next[i])),
                        self.actions_ : np.squeeze(np.asarray(a_his[i])),
                        self.rewards_ : np.squeeze(np.asarray(r[i])),
                        self.done_ : np.squeeze(np.asarray(done[i],dtype=float))

                    }
                    out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)   # local grads applied to global net.
                    out = np.array_split(out,3)
                    losses = out[1]
                    grads = out[2]

                    for i,loss in enumerate(losses):
                        self.loss_MA[i].append(loss)

                    for i,grads_i in enumerate(grads):
                        total_counter = 1
                        vanish_counter = 0
                        for grad in grads_i:
                            total_counter += np.prod(grad.shape)
                            vanish_counter += (np.absolute(grad)<1e-8).sum()
                        self.grad_MA[i].append(vanish_counter/total_counter)

                    ent = self.sess.run(self.entropy, feedDict)   # local grads applied to global net.
                    entropy = np.average(np.asarray(ent))
                    self.entropy_MA.append(entropy)

        self.ClearTrajectory()


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
            dict["Training Results/Entropy"] = self.entropy_MA()
        return dict


    @property
    def getVars(self):
        return self.Model.getVars(self.scope)
