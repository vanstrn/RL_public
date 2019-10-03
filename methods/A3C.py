
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
            self.c_loss = tf.reduce_mean(tf.square(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval

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
        s, s_ = s0[np.newaxis, :], s1[np.newaxis, :]
        v_ = self.sess.run(self.critic, {self.s: s_})
        feedDict = {self.s: s, self.v_: v_, self.r: r,self.a:a}
        td_error, _, _, exp_v = self.sess.run([self.td_error, self.train_op_c,self.train_op_a, self.exp_v],feedDict)

        return exp_v

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

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

class A3C_s(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,globalAC=None,lr_c=1E-4,lr_a=1E-3,ENTROPY_BETA = 0.001):
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
        self.a_prob,self.v = self.Model(self.s)

        if globalAC is None:   # get global network
            with tf.variable_scope(scope):
                self.a_params = self.Model.getAParams
                self.c_params = self.Model.getCParams
        else:   # local net, calculate losses
            with tf.variable_scope(scope):

                self.a_params = self.Model.getAParams
                self.c_params = self.Model.getCParams

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, actionSize, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = tf.train.AdamOptimizer(lr_a).apply_gradients(zip(self.a_grads, globalAC.getAParams))
                    self.update_c_op = tf.train.AdamOptimizer(lr_c).apply_gradients(zip(self.c_grads, globalAC.getCParams))



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

            v_s_ = r + HPs["GAMMA"] * v_s_
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
        return self.Model.getAParams

    @property
    def getCParams(self):
        return self.Model.getCParams
