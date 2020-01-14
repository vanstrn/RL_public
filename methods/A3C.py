
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
import tensorflow as tf
import numpy as np

class A3C(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,globalAC=None,):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.buffer = Trajectory(depth=5)
        self.sess=sess
        self.Model = sharedModel
        self.s = tf.placeholder(tf.float32, [None, stateShape], 'S')
        self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
        out = self.Model(self.s)
        self.a_prob = out["actor"]
        self.v = out["critic"]

        if globalAC is None:   # get global network
            with tf.variable_scope(scope):
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope+"_update"):

                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, actionSize, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
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



    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.a_prob, {self.s: s})   # get probabilities for all actions

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()) ,[]  # return a int and extra data that needs to be fed to buffer.

    def Update(self,HPs):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Process the data from the buffer
        self.ProcessBuffer(HPs)

        #Create a feedDict from the buffer
        feedDict = {
            self.s: np.vstack(self.buffer[0]),
            self.a_his: np.array(self.buffer[1]),
            self.v_target: np.vstack(self.buffer[2]),
        }

        #Perform update operations
        self.sess.run([self.update_a_op, self.update_c_op], feedDict)   # local grads applied to global net.
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])   # global variables synched to the local net.

        #Clear of reset the buffer.
        self.buffer.clear()

    def AddToBuffer(self,sample):
        """Add a sample to the buffer.
        Takes the form of [s0,a,r,s1,done,extraData]
        extraData is outputted from the Network and is appended to the sample.
        Also handles any data that needs to be processed in the network.
        """
        self.buffer.append(sample)

    def ProcessBuffer(self,HPs):
        """Take the buffer and calculate future rewards.
        """
        buffer_v_s_ = []
        for r in self.buffer[2][::-1]:
            if self.buffer[4][-1]:
                v_s_ = 0   # terminal
            else:
                v_s_ = self.sess.run(self.v, {self.s: self.buffer[3][-1][np.newaxis, :]})[0, 0]

            v_s_ = r + HPs["Gamma"] * v_s_
            buffer_v_s_.append(v_s_)

        buffer_v_s_.reverse()
        self.buffer[2] = buffer_v_s_

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
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ 'Actor')

    @property
    def getCParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
