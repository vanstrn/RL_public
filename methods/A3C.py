
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

class A3C(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,globalAC=None,nTrajs=1):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders

        self.sess=sess
        self.scope=scope
        self.Model = sharedModel
        if len(stateShape) == 4:
            self.s = tf.placeholder(tf.float32, [None]+stateShape[1:4], 'S')
        else:
            self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
        self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
        self.v_target = tf.placeholder(tf.float32, [None], 'Vtarget')

        input = {"state":self.s}
        out = self.Model(input)
        self.a_prob = out["actor"]
        self.v = out["critic"]

        if globalAC is None:   # get global network
            with tf.variable_scope(scope):
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
        else:   # local net, calculate losses
            self.buffer = [Trajectory(depth=6) for _ in range(nTrajs)]
            with tf.variable_scope(scope+"_update"):

                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
                print(self.c_params)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, actionSize, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.entropy =entropy
                    self.exp_v = HPs["EntropyBeta"] * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = tf.train.AdamOptimizer(HPs["Actor LR"]).apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = tf.train.AdamOptimizer(HPs["Critic LR"]).apply_gradients(zip(self.c_grads, globalAC.c_params))

            self.update_ops = [self.update_a_op,self.update_c_op,]
            self.pull_ops = [self.pull_a_params_op,self.pull_c_params_op,]
            self.grads = [self.a_grads,self.c_grads,]
            self.losses = [self.a_loss,self.c_loss,]

            self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
            self.loss_MA = [MovingAverage(400) for i in range(len(self.grads))]
            self.entropy_MA = MovingAverage(400)
            self.labels = ["Actor","Critic",]
            self.HPs = HPs

    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        probs,v = self.sess.run([self.a_prob,self.v], {self.s: state})   # get probabilities for all actions

        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions ,[v]  # return a int and extra data that needs to be fed to buffer.

    def Update(self,HPs,episode=0,statistics=True):
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

            td_target,_ = self.ProcessBuffer(HPs,traj)
            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1

            s=  np.array_split(np.squeeze(self.buffer[traj][0]), batches)
            a_his=  np.array_split(np.asarray(self.buffer[traj][1]).reshape(-1), batches)
            v_target=  np.array_split(td_target, batches)

            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):
                #Create a feedDict from the buffer
                    feedDict = {
                        self.s: s[i],
                        self.a_his: a_his[i],
                        self.v_target: v_target[i],
                    }
                    if not statistics:
                        self.sess.run(self.update_ops, feedDict)
                    else:
                    #Perform update operations
                        out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)   # local grads applied to global net.
                        out = np.array_split(out,3)
                        losses = out[1]
                        grads = out[2]

                        for i,loss in enumerate(losses):
                            self.loss_MA[i].append(loss)

                        for i,grads_i in enumerate(grads):
                            total_counter = 0
                            vanish_counter = 0
                            for grad in grads_i:
                                total_counter += np.prod(grad.shape)
                                vanish_counter += (np.absolute(grad)<1e-8).sum()
                            self.grad_MA[i].append(vanish_counter/total_counter)

                        ent = self.sess.run(self.entropy, feedDict)   # local grads applied to global net.
                        entropy = np.average(np.asarray(ent))
                        self.entropy_MA.append(entropy)

        self.ClearTrajectory()
        self.sess.run(self.pull_ops)   # global variables synched to the local net.


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
            dict["Training Results/Entropy"] = self.entropy_MA()
        return dict


    def ProcessBuffer(self,HPs,traj):
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


    @property
    def getVars(self):
        return self.Model.getVars(self.scope)

    @property
    def getAParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ 'Actor')

    @property
    def getCParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
