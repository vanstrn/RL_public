
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
import tensorflow as tf
import numpy as np


class PPO_s(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,globalAC=None,lr_c=1E-4,lr_a=1E-3,ENTROPY_BETA = 0.001):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        # @staticmethod
        # def Loss(policy, log_prob, old_log_prob,
        #         action, advantage,
        #         td_target, critic,
        #         entropy_beta=0.001, critic_beta=0.5,
        #         eps=0.2,
        #         name_scope='loss'):
        with tf.name_scope(name_scope):
            self.sess=sess
            self.Model = sharedModel
            #Placeholders
            self.s = tf.placeholder(tf.float32, [None, stateShape], 'S')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
            self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
            self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')

            self.a_prob,self.v = self.Model(self.s)
            # Entropy
            entropy = -tf.reduce_mean(policy * Loss._log(policy), name='entropy')

            # Critic Loss
            td_error = td_target - critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_size = tf.shape(policy)[1]
            action_OH = tf.one_hot(action, action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(log_prob * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_prob * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            if entropy_beta != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if critic_beta != 0:
                actor_loss = actor_loss + critic_loss * critic_beta
            loss = actor_loss
            # Build Trainer
            self.optimizer = tf.keras.optimizers.Adam(lr)
            self.gradients = optimizer.get_gradients(loss, model.trainable_variables)
            self.update_ops = optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))

    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.a_prob, {self.s: s})   # get probabilities for all actions

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def Update(self, s0,a,r,s1):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}

        self.sess.run(self.update_ops, feed_dict)


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
