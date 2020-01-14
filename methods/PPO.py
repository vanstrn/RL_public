
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
import tensorflow as tf
import numpy as np


class PPO(Method):

    def __init__(self,Model,sess,stateShape,actionSize,scope,globalAC=None,HPs):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Creating appropriate buffer for the method.
        self.buffer = Trajectory(depth=5)

        with tf.name_scope(name_scope):
            self.sess=sess
            self.Model = Model
            #Placeholders
            self.s = tf.placeholder(tf.float32, [None, stateShape], 'S')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
            self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
            self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')

            out = self.Model(self.s)
            self.a_prob = out["actor"]
            self.v = out["critic"]

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

            if HPs["Entropy Beta"] != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if HPs["Critic Beta"] != 0:
                actor_loss = actor_loss + critic_loss * critic_beta
            loss = actor_loss
            # Build Trainer
            self.optimizer = tf.keras.optimizers.Adam(HPs["LR"])
            self.gradients = self.optimizer.get_gradients(loss, model.getVars)
            self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, model.getVars))

    def GetAction(self, state):
        """
        Contains the code to run the network based on an input.
        """
        s = state[np.newaxis, :]
        probs = self.sess.run(self.a_prob, {self.s: s})   # get probabilities for all actions

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), []   # return a int

    def Update(self,HPs):
        """
        Takes an input buffer and applies the updates to the networks through gradient
        backpropagation
        """
        self.ProcessBuffer(HPs)

        #Staging Buffer inputs into the entries to run through the network.
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}
        #Running the data through th
        self.sess.run(self.update_ops, feed_dict)

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
