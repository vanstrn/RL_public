
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
from utils.utils import MovingAverage

from networks.common import NetworkBuilder

class PPO(Method):

    def __init__(self,sess,settings,netConfigOverride,stateShape,actionSize,nTrajs=1,**kwargs):
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
        self.HPs = settings["NetworkHPs"]

        #Building the network.
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)

        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope("PPO"):
                #Placeholders
                if len(stateShape) == 4:
                    self.s = tf.placeholder(tf.float32, [None]+stateShape[0:4], 'S')
                else:
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
                self.entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')

                # Critic Loss
                td_error = self.td_target_ - self.v
                self.critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

                # Actor Loss
                action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
                log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
                old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

                # Clipped surrogate function
                ratio = tf.exp(log_prob - old_log_prob)
                surrogate = ratio * self.advantage_
                clipped_surrogate = tf.clip_by_value(ratio, 1-self.HPs["eps"], 1+self.HPs["eps"]) * self.advantage_
                surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
                self.actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

                loss = self.actor_loss - self.entropy * self.HPs["EntropyBeta"] + self.critic_loss * self.HPs["CriticBeta"]

                # Build Trainer
                if self.HPs["Optimizer"] == "Adam":
                    self.optimizer = tf.keras.optimizers.Adam(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "RMS":
                    self.optimizer = tf.keras.optimizers.RMSProp(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adagrad":
                    self.optimizer = tf.keras.optimizers.Adagrad(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adadelta":
                    self.optimizer = tf.keras.optimizers.Adadelta(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Adamax":
                    self.optimizer = tf.keras.optimizers.Adamax(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Nadam":
                    self.optimizer = tf.keras.optimizers.Nadam(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "SGD":
                    self.optimizer = tf.keras.optimizers.SGD(self.HPs["LR"])
                elif self.HPs["Optimizer"] == "Amsgrad":
                    self.optimizer = tf.keras.optimizers.Nadam(self.HPs["LR"],amsgrad=True)
                else:
                    print("Not selected a proper Optimizer")
                    exit()
                self.gradients = self.optimizer.get_gradients(loss, self.Model.trainable_variables)
                self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, self.Model.trainable_variables))
                # self.gradients = self.optimizer.get_gradients(loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
                # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
                # self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)))

        #Creating variables for logging.
        self.EntropyMA = MovingAverage(400)
        self.CriticLossMA = MovingAverage(400)
        self.ActorLossMA = MovingAverage(400)
        self.GradMA = MovingAverage(400)

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
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions, [v,log_logits]

    def Update(self,episode=0):
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
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        for traj in range(len(self.buffer)):

            #Finding if there are more than 1 done in the sequence. Clipping values if required.

            td_target, advantage = self.ProcessBuffer(traj)

            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1
            s = np.array_split( self.buffer[traj][0], batches)
            a_his = np.array_split( np.asarray(self.buffer[traj][1]).reshape(-1), batches)
            td_target_ = np.array_split( td_target, batches)
            advantage_ = np.array_split( np.reshape(advantage, [-1]), batches)
            old_log_logits_ = np.array_split( np.reshape(self.buffer[traj][6], [-1,self.actionSize]), batches)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?

            #Staging Buffer inputs into the entries to run through the network.
            # print(td_target)
            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):

                    feed_dict = {self.s: np.squeeze(np.asarray(s[i])),
                                 self.a_his: np.asarray(a_his[i]),
                                 self.td_target_:np.asarray(td_target_[i]),
                                 self.advantage_: np.asarray(advantage_[i]),
                                 self.old_log_logits_: np.asarray(old_log_logits_[i])}
                    aLoss= self.sess.run([self.actor_loss], feed_dict)
                    aLoss, cLoss, entropy,grads, _ = self.sess.run([self.actor_loss,self.critic_loss,self.entropy,self.gradients,self.update_ops], feed_dict)

                    self.EntropyMA.append(entropy)
                    self.CriticLossMA.append(cLoss)
                    self.ActorLossMA.append(aLoss)
                    total_counter = 0
                    vanish_counter = 0
                    for grad in grads:
                        total_counter += np.prod(grad.shape)
                        vanish_counter += (np.absolute(grad)<1e-8).sum()
                    self.GradMA.append(vanish_counter/total_counter)

        self.ClearTrajectory()


    def GetStatistics(self):
        dict = {"Training Results/Entropy":self.EntropyMA(),
        "Training Results/Loss Critic":self.CriticLossMA(),
        "Training Results/Loss Actor":self.ActorLossMA(),
        "Training Results/Vanishing Gradient":self.GradMA(),}
        return dict


    def ProcessBuffer(self,traj):
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
        return self.Model.getVars("PPO_Training")
