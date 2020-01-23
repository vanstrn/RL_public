
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
from utils.dataProcessing import gae
import tensorflow as tf
import numpy as np


class PPO(Method):

    def __init__(self,Model,sess,stateShape,actionSize,HPs,nTrajs=1):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]
        self.actionSize = actionSize
        with tf.name_scope("PPO_Training"):
            self.sess=sess
            self.Model = Model
            #Placeholders
            self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
            self.td_target_ = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
            self.old_log_logits_ = tf.placeholder(shape=[None, actionSize], dtype=tf.float32, name='old_logit_hold')

            out = self.Model(self.s)
            self.a_prob = out["actor"]
            self.v = out["critic"]
            self.log_logits = out["log_logits"]

            # Entropy
            def _log(val):
                return tf.log(tf.clip_by_value(val, 1e-10, 10.0))
            entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')

            # Critic Loss
            td_error = self.td_target_ - self.v
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * self.advantage_
            clipped_surrogate = tf.clip_by_value(ratio, 1-HPs["eps"], 1+HPs["eps"]) * self.advantage_
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            if HPs["EntropyBeta"] != 0:
                actor_loss = actor_loss - entropy * HPs["EntropyBeta"]
            if HPs["CriticBeta"] != 0:
                actor_loss = actor_loss + critic_loss * HPs["CriticBeta"]
            loss = actor_loss
            # Build Trainer
            self.optimizer = tf.keras.optimizers.Adam(HPs["LR"])
            self.gradients = self.optimizer.get_gradients(loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "PPO_Training/"+self.Model.scope))
            self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "PPO_Training/"+self.Model.scope)))

    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        probs,log_logits,v = self.sess.run([self.a_prob,self.log_logits,self.v], {self.s: state})   # get probabilities for all actions
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions, [v,log_logits]   # return a int

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
            feed_dict = {self.s: self.buffer[traj][0][:clip],
                         self.a_his: self.buffer[traj][1][:clip],
                         self.td_target_: td_target,
                         self.advantage_: np.reshape(advantage, [-1]),
                         self.old_log_logits_: np.reshape(self.buffer[traj][6][:clip], [-1,self.actionSize])}
            #Running the data through th
            self.sess.run(self.update_ops, feed_dict)

    def ProcessBuffer(self,HPs,traj,clip):
        """Take the buffer and calculate future rewards.
        """
        td_target, advantage = gae(self.buffer[traj][2][:clip],self.buffer[traj][5][:clip],0,HPs["Gamma"],HPs["lambda"])
        return td_target, advantage

    @property
    def getVars(self):
        return self.Model.getVars
