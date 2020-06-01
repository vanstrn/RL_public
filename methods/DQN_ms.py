
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
from .AdvantageEstimator import MultiStepDiscountProcessing
import tensorflow as tf
import numpy as np
from utils.utils import MovingAverage
from utils.record import Record
import random


class DQN_ms(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,globalAC=None,nTrajs=1):
        """
        Initializes I/O placeholders and the training process of a Multi-step DQN.
        Main principal is that instead of one-step TD diference, the loss is evaluated on a
        temporally extended basis.
        G = R_t + γR_t+1 + ... γ^n-1 R_t+n + q(S_t+n,a*,θ-)
        loss = MSE(G,q(S_t,A_t,θ))

        """
        #Placeholders
        self.actionSize = actionSize
        self.sess=sess
        self.scope=scope
        self.Model = sharedModel

        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]
        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):
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
                    td_target = self.rewards_  + HPs["Gamma"] * max_next_q * (1. - self.done_)

                with tf.name_scope('td_error'):
                    loss = tf.keras.losses.MSE(td_target, curr_q)
                    softmax_q = tf.nn.softmax(curr_q)
                    self.entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q+ 1e-5))
                    self.loss=total_loss = loss + HPs["EntropyBeta"] * self.entropy

                optimizer = tf.keras.optimizers.Adam(HPs["LearningRate"])
                self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

                self.gradients = optimizer.get_gradients(total_loss, self.params)
                self.update_op = optimizer.apply_gradients(zip(self.gradients, self.params))

                self.grads=[self.gradients]
                self.losses=[self.loss]
                self.update_ops=[self.update_op]

        self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
        self.loss_MA = [MovingAverage(400) for i in range(len(self.losses))]
        self.labels = ["Critic"]
        self.HPs = HPs

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
                prob = self.HPs["ExploreSS"] + (1-self.HPs["ExploreSS"])*(np.exp(-episode/self.HPs["ExplorationDecay"]))
                if random.uniform(0, 1) < prob:
                    actions = random.randint(0,self.actionSize-1)
                else:
                    actions = np.argmax(q, axis=-1)
            else:
                actions = np.argmax(q, axis=-1)
        else:
            actions = np.argmax(q, axis=-1)
        return actions ,[]  # return a int and extra data that needs to be fed to buffer.

    def Update(self,HPs,episode=0,statistics=True):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Checking that there is enough data for a batch
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        #Combining all trajs into 1:
        s_list = []
        a_list = []
        done_list = []
        g_list = []
        s_n_list = []
        for traj in range(len(self.buffer)):
            g,s_n=MultiStepDiscountProcessing(self.buffer[traj][2],self.buffer[traj][3],self.HPs["Gamma"],self.HPs["MultiStep"])
            s_list.extend(self.buffer[traj][0])
            a_list.extend(self.buffer[traj][1])
            g_list.extend(g)
            s_n_list.extend(s_n)
            done_list.extend(self.buffer[traj][4])

        #Separating into different batches
        batches = len(s_list)//self.HPs["MinibatchSize"]+1
        s = np.array_split( s_list, batches)
        a_his = np.array_split( np.asarray(a_list).reshape(-1), batches)
        r = np.array_split( np.asarray(g_list).reshape(-1), batches)
        s_next = np.array_split( s_n_list, batches)
        done = np.array_split( done_list, batches)

        #Running all batches through multiple epochs
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
                out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)
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

        self.ClearTrajectory()


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
        return dict

    @property
    def getVars(self):
        return self.Model.getVars(self.scope)
