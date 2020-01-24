
"""
To Do:
Implement the ACKTR algorithm.
"""
from .method import Method
from .buffer import Trajectory
from utils.dataProcessing import gae
import tensorflow as tf
import numpy as np
from methods import kfac

class ACKTR(Method):

    def __init__(self,Model,sess,stateShape,actionSize,HPs,nTrajs=1):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        self.buffer = [Trajectory(depth=6) for _ in range(nTrajs)]
        self.actionSize = actionSize
        self.sess=sess
        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope("ACKTR_Model"):
                self.Model = Model
                #Placeholders
                self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.td_target_ = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')

                out = self.Model(self.s)
                self.a_prob = out["actor"]
                self.v = out["critic"]
                self.log_logits = out["log_logits"]

                action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
                neglogpac = -tf.reduce_sum(self.log_logits * action_OH, 1)
                pg_loss = tf.reduce_mean(self.advantage_*neglogpac)
                entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')
                pg_loss = pg_loss - HPs["EntropyCoeff"] * entropy
                vf_loss = tf.losses.mean_squared_error(tf.squeeze(self.v), self.td_target_)
                train_loss = pg_loss + HPs["ValueFunctionCoeff"] * vf_loss

                ##Fisher loss construction
                self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
                sample_net = self.v + tf.random_normal(tf.shape(self.v))
                self.vf_fisher = vf_fisher_loss = - HPs["FischerCoeff"]*tf.reduce_mean(tf.pow(self.v - tf.stop_gradient(sample_net), 2))
                self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss


                self.params= params = self.getVars

                self.grads_check = grads = tf.gradients(train_loss,params)
                self.optim = optim = kfac.KfacOptimizer(learning_rate=HPs["LR"], clip_kl=0.001,\
                    momentum=0.9, kfac_update=1, epsilon=0.01,\
                    stats_decay=0.99, is_async=False, cold_iter=10, max_grad_norm=0.5)
                optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
                self.train_op, self.q_runner = optim.apply_gradients(list(zip(grads,params)))


    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        probs,log_logits,v = self.sess.run([self.a_prob,self.log_logits,self.v], {self.s: state})   # get probabilities for all actions
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions, [v]   # return a int

    def Update(self,HPs):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        for traj in range(len(self.buffer)):

            #Ignoring Samples after the environment is done.
            clip = -1
            try:
                for j in range(1):
                    clip = self.buffer[traj][4].index(True, clip + 1)
            except:
                clip=len(self.buffer[traj][4])
            td_target, advantage = self.ProcessBuffer(HPs,traj,clip)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?

            #Staging Buffer inputs into the entries to run through the network.
            if len(self.buffer[traj][0][:clip]) == 0:
                continue
            feed_dict = {self.s: self.buffer[traj][0][:clip],
                         self.a_his: self.buffer[traj][1][:clip],
                         self.td_target_: td_target,
                         self.advantage_: np.reshape(advantage, [-1])}
            #Running the data through th
            self.sess.run(self.update_ops, feed_dict)

    def ProcessBuffer(self,HPs,traj,clip):
        """Take the buffer and calculate future rewards.
        """
        td_target, advantage = gae(self.buffer[traj][2][:clip],self.buffer[traj][5][:clip],0,HPs["Gamma"],HPs["lambda"])
        return td_target, advantage

    @property
    def getVars(self):
        return self.Model.getVars("ACKTR_Model")
