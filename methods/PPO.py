
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
import scipy
from utils.record import Record


class PPO(Method):

    def __init__(self,Model,sess,stateShape,actionSize,HPs,nTrajs=1,scope="PPO_Training"):
        """
        Initializes a training method for a neural network.

        Parameters
        ----------
        Model : Keras Model Object
            A Keras model object with fully defined layers and a call function. See examples in networks module.
        sess : Tensorflow Session
            Initialized Tensorflow session
        stateShape : list
            List of integers of the inputs shape size. Ex [39,39,6]
        actionSize : int
            Output size of the network.
        HPs : dict
            Dictionary that contains all hyperparameters to be used in the methods training
        nTrajs : int (Optional)
            Number that specifies the number of trajectories to be created for collecting training data.
        scope : str (Optional)
            Name of the PPO method. Used to group and differentiate variables between other networks.

        Returns
        -------
        N/A
        """
        #Processing inputs
        self.actionSize = actionSize
        self.sess=sess
        self.Model = Model

        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):
                #Placeholders
                print(stateShape)
                self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.td_target_ = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_log_logits_ = tf.placeholder(shape=[None, actionSize], dtype=tf.float32, name='old_logit_hold')

                #Initializing Netowrk I/O
                inputs = {"state":self.s}
                out = self.Model(inputs)
                self.a_prob = out["actor"]
                self.v = out["critic"]
                self.log_logits = out["log_logits"]

                # Entropy
                def _log(val):
                    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))
                entropy = self.entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')

                # Critic Loss
                td_error = self.td_target_ - self.v
                critic_loss = self.critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

                # Actor Loss
                action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
                log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
                old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

                # Clipped surrogate function
                ratio = tf.exp(log_prob - old_log_prob)
                surrogate = ratio * self.advantage_
                clipped_surrogate = tf.clip_by_value(ratio, 1-HPs["eps"], 1+HPs["eps"]) * self.advantage_
                surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
                actor_loss = self.actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

                actor_loss = actor_loss - entropy * HPs["EntropyBeta"]
                loss = actor_loss + critic_loss * HPs["CriticBeta"]

                # Build Trainer
                self.optimizer = tf.keras.optimizers.Adam(HPs["LR"])
                self.gradients = self.optimizer.get_gradients(loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
                self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)))

        #Creating variable for logging.
        self.EntropyMA = 0
        self.CriticLossMA = 0
        self.ActorLossMA = 0
        self.GradMA = 0

    def GetAction(self, state, episode=1,step=0):
        """
        Method to run data through the neural network.

        Parameters
        ----------
        state : np.array
            Data with the shape of [N, self.stateShape] where N is number of smaples

        Returns
        -------
        actions : list[int]
            List of actions based on NN output.
        extraData : list
            List of data that is passed to the execution code to be bundled with state data.
        """
        try:
            probs,log_logits,v = self.sess.run([self.a_prob,self.log_logits,self.v], {self.s: state})
        except ValueError:
            probs,log_logits,v = self.sess.run([self.a_prob,self.log_logits,self.v], {self.s: np.expand_dims(state,axis=0)})
        # probs[:,2] *= 2.0
        # probs[:,1] *= 0.75
        # probs[:,0] *= 0.75
        # probs=scipy.special.softmax(probs)
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        # print(probs,actions)
        return actions, [v,log_logits]

    def Update(self,HPs,episode=0):
        """
        Process the buffer and backpropagates the loses through the NN.

        Parameters
        ----------
        HPs : dict
            Hyperparameters for training.

        Returns
        -------
        N/A
        """
        for traj in range(len(self.buffer)):

            #Finding if there are more than 1 done in the sequence. Clipping values if required.
            clip = -1
            try:
                for j in range(2):
                    clip = self.buffer[traj][4].index(True, clip + 1)
            except:
                clip=len(self.buffer[traj][4])

            td_target, advantage = self.ProcessBuffer(HPs,traj,clip)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?

            #Staging Buffer inputs into the entries to run through the network.
            # print(td_target)
            if len(self.buffer[traj][0][:clip]) == 0:
                continue
            feed_dict = {self.s: self.buffer[traj][0][:clip],
                         self.a_his: np.asarray(self.buffer[traj][1][:clip]).reshape(-1),
                         self.td_target_: td_target,
                         self.advantage_: np.reshape(advantage, [-1]),
                         self.old_log_logits_: np.reshape(self.buffer[traj][6][:clip], [-1,self.actionSize])}
            aLoss, cLoss, entropy,grads, _ = self.sess.run([self.actor_loss,self.critic_loss,self.entropy,self.gradients,self.update_ops], feed_dict)

            self.EntropyMA = self.EntropyMA*.99 + entropy*0.1
            self.CriticLossMA = self.CriticLossMA*.99 + cLoss*0.1
            self.ActorLossMA = self.ActorLossMA*.99 + aLoss*0.1
            total_counter = 0
            vanish_counter = 0
            for grad in grads:
                total_counter += np.prod(grad.shape)
                vanish_counter += (np.absolute(grad)<1e-8).sum()
            self.GradMA = self.GradMA*.99 + vanish_counter/total_counter*0.1


    def GetStatistics(self.):
        dict = {"Training Results/Entropy":self.EntropyMA,
        "Training Results/Critic Loss":self.CriticLossMA,
        "Training Results/Actor Loss":self.ActorLossMA,
        "Training Results/Vanishing Gradient":self.GradMA,}
        return dict


    def ProcessBuffer(self,HPs,traj,clip):
        """
        Process the buffer and backpropagates the loses through the NN.

        Parameters
        ----------
        Model : HPs
            Hyperparameters for training.
        traj : Trajectory
            Data stored by the neural network.
        clip : list[bool]
            List where the trajectory has finished.

        Returns
        -------
        td_target : list
            List Temporal Difference Target for particular states.
        advantage : list
            List of advantages for particular actions.
        """
        # print("Starting Processing Buffer\n")
        # tracker.print_diff()

        td_target, advantage = gae(self.buffer[traj][2][:clip],self.buffer[traj][5][:clip],0,HPs["Gamma"],HPs["lambda"])
        # tracker.print_diff()
        return td_target, advantage

    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")
