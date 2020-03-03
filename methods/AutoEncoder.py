
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

class AE(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,globalAC=None,nTrajs=1):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.actionSize =actionSize
        self.sess=sess
        self.scope=scope
        self.Model = sharedModel
        self.s = tf.placeholder(tf.float32, [None] + stateShape, 'S')
        self.a = tf.placeholder(tf.float32, [None], 'A')
        self.s_next = tf.placeholder(tf.float32, [None] + stateShape, 'S_next')

        input = {"state":self.s,"action":self.a}
        out = self.Model(input)
        self.state_pred = out["prediction"]
        self.phi = out["phi"]

        if globalAC is None:   # get global network
            with tf.variable_scope(scope):
                self.s_params = self.Model.GetVariables("Reconstruction")
        else:   # local net, calculate losses
            self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]
            with tf.variable_scope(scope+"_update"):

                self.s_params = self.Model.GetVariables("Reconstruction")

                with tf.name_scope('s_loss'):
                    if HPs["loss"] == "MSE":
                        self.s_loss = tf.losses.mean_squared_error(self.state_pred,self.s_next)
                    elif HPs["loss"] == "KL":
                        self.s_loss = tf.losses.KLD(self.state_pred,self.s_next)
                    elif HPs["loss"] == "M4E":
                        self.s_loss = tf.reduce_mean((self.state_pred-self.s_next)**4)

                with tf.name_scope('local_grad'):
                    self.s_grads = tf.gradients(self.s_loss, self.s_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_s_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.s_params, globalAC.s_params)]

                with tf.name_scope('push'):
                    self.update_s_op = tf.train.AdamOptimizer(HPs["State LR"]).apply_gradients(zip(self.s_grads, globalAC.s_params))

            self.update_ops = [self.update_s_op]
            self.pull_ops = [self.pull_s_params_op]
            self.grads = [self.s_grads]
            self.losses = [self.s_loss]

            self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
            self.loss_MA = [MovingAverage(400) for i in range(len(self.grads))]
            self.labels = ["State"]

            self.sess.run(self.pull_ops) #Pulling the variables from the global network to initialize.


    def GetAction(self, state,episode=0,step=0,deterministic=False,debug=False):
        """
        Contains the code to run the network based on an input.
        """
        p = 1/self.actionSize
        if len(state.shape)==3:
            probs =np.full((1,self.actionSize),p)
        else:
            probs =np.full((state.shape[0],self.actionSize),p)
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        if debug: print(probs)
        return actions ,[]  # return a int and extra data that needs to be fed to buffer.

    def PredictState(self,state):
        s = state[np.newaxis, :]
        state_pred = self.sess.run([self.state_pred], {self.s: s})
        return state_pred

    def Update(self,HPs,episode=0,statistics=True):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Process the data from the buffer
        for traj in range(len(self.buffer)):
            clip = -1
            try:
                for j in range(2):
                    clip = self.buffer[traj][4].index(True, clip + 1)
            except:
                clip=len(self.buffer[traj][4])


            #Create a feedDict from the buffer
            feedDict = {
                self.s: self.buffer[traj][0][:clip],
                self.s_next: self.buffer[traj][3][:clip],
                self.a: np.asarray(self.buffer[traj][1][:clip]).reshape(-1),
                # self.a_his: np.asarray(self.buffer[traj][1][:clip]).reshape(-1),
            }

            if not statistics:
                self.sess.run(self.update_ops, feedDict)   # local grads applied to global net.
            else:
                #Perform update operations
                try:
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
                            vanish_counter += (np.absolute(grad)<1e-6).sum()
                        self.grad_MA[i].append(vanish_counter/total_counter)
                except:
                    out = self.sess.run(self.update_ops+self.losses, feedDict)   # local grads applied to global net.
                    out = np.array_split(out,2)
                    losses = out[1]

                    for i,loss in enumerate(losses):
                        self.loss_MA[i].append(loss)

        self.sess.run(self.pull_ops)   # global variables synched to the local net.


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
        return dict

    @property
    def getVars(self):
        return self.Model.getVars(self.scope)

    @property
    def getAParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ 'Actor')

    @property
    def getCParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
