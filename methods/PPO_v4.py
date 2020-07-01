
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
import operator

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
        self.Model_ = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)

        #Creating appropriate buffer for the method.
        self.sharedBuffer = PriorityBuffer(maxSamples=settings["MemoryCapacity"])
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
                self.td_error = tf.reduce_mean(self.td_target_ - self.v)
                self.critic_loss = tf.reduce_mean(tf.square(self.td_error), name='critic_loss')

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

                self.params = self.Model.trainable_variables
                self.target_params = self.Model_.trainable_variables
                self.gradients = self.optimizer.get_gradients(loss, self.params)
                self.update_op = self.optimizer.apply_gradients(zip(self.gradients, self.target_params))

                self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, self.target_params)]

        self.update_ops=[self.update_op]
        self.logging_ops=[self.actor_loss,self.critic_loss,self.entropy]
        self.labels = ["Loss Actor","Loss Critic","Entropy"]
        self.logging_MA = [MovingAverage(400) for i in range(len(self.logging_ops))]

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

        self.PushToBuffer()

        samples,num = self.sharedBuffer.Sample()
        if num < self.HPs["BatchSize"]:
            return
        # print("Updating")
        self.sess.run(self.pull_params_op, {}) 

        for traj in samples:
            if len(traj[0]) <= 5:
                continue

            for epoch in range(self.HPs["Epochs"]):
                #Create a feedDict from the buffer
                feedDict = {self.s: np.squeeze(np.asarray(traj[0])),
                             self.a_his: np.asarray(traj[1]),
                             self.td_target_:np.asarray(traj[2]),
                             self.advantage_: np.asarray(traj[3]),
                             self.old_log_logits_: np.asarray(traj[4])}
                out = self.sess.run(self.update_ops+self.logging_ops, feedDict)   # local grads applied to global net.
                logging = out[len(self.update_ops):]

                for i,log in enumerate(logging):
                    self.logging_MA[i].append(log)

        self.PrioritizeBuffer()


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/" + label] = self.logging_MA[i]()
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

    def PushToBuffer(self):
        #Estimating TD Difference to give priority to the data.

        for traj in range(len(self.buffer)):
            td_target, advantage = self.ProcessBuffer(traj)

            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1
            s = np.array_split( self.buffer[traj][0], batches)
            a_his = np.array_split( np.asarray(self.buffer[traj][1]).reshape(-1), batches)
            td_target_ = np.array_split( td_target, batches)
            advantage_ = np.array_split( np.reshape(advantage, [-1]), batches)
            old_log_logits_ = np.array_split( np.reshape(self.buffer[traj][6], [-1,self.actionSize]), batches)

            #Staging Buffer inputs into the entries to run through the network.
            # print(td_target)
            for i in range(batches):
                if len(np.squeeze(np.asarray(s[i])).shape) == 3:
                    continue
                feedDict = {self.s: np.squeeze(np.asarray(s[i])),
                             self.a_his: np.asarray(a_his[i]),
                             self.td_target_:np.asarray(td_target_[i]),
                             self.advantage_: np.asarray(advantage_[i]),
                             self.old_log_logits_: np.asarray(old_log_logits_[i])}
                priority = self.sess.run(self.td_error, feedDict)

                self.sharedBuffer.AddTrajectory([s[i],a_his[i],td_target_[i],advantage_[i],old_log_logits_[i]],priority)
        self.sharedBuffer.PrioritizeandPruneSamples(2048)

        self.ClearTrajectory()

    def PrioritizeBuffer(self):
        #Getting the data that needs to be assigned a new priority.
        trajs = self.sharedBuffer.GetReprioritySamples()
        priority=[]
        for traj in trajs:
            feedDict = {self.s: np.squeeze(np.asarray(traj[0])),
                         self.a_his: np.asarray(traj[1]),
                         self.td_target_:np.asarray(traj[2]),
                         self.advantage_: np.asarray(traj[3]),
                         self.old_log_logits_: np.asarray(traj[4])}
            priority.append( self.sess.run(self.td_error, feedDict))
        #Calculating the priority.
        self.sharedBuffer.UpdatePriorities(priority)

        #Pushing the priorities back to the buffer
        self.sharedBuffer.PrioritizeandPruneSamples(2048)


    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")


class PriorityBuffer():
    def __init__(self,maxSamples=10000):
        self.maxSamples = maxSamples
        self.buffer=[]
        self.priorities=[]
        self.trajLengths=[]
        self.flag = True
        self.slice=0
        self.sampleSize=0
        self.errorMA=MovingAverage(1000)

    def GetMuSigma(self):
        return self.errorMA(), self.errorMA.std()

    def AddError(self,val):
        self.errorMA.append(val)

    def AddTrajectory(self,sample,priority):
        if len(sample[0]) == 0:
            return
        self.buffer.append(sample)
        self.priorities.append(priority)
        self.trajLengths.append(len(sample[0]))

    def Sample(self):
        return self.buffer[0:self.slice] , self.sampleSize

    def PrioritizeandPruneSamples(self,sampleSize):
        if len(self.trajLengths) ==0:
            return
        if self.flag:
            self.flag=False
        self.priorities, self.buffer,self.trajLengths = (list(t) for t in zip(*sorted(zip(self.priorities, self.buffer,self.trajLengths), key = operator.itemgetter(0), reverse=True)))

        #Pruning the least favorable samples
        while sum(self.trajLengths) >= self.maxSamples:
            self.priorities.pop(-1)
            self.buffer.pop(-1)
            self.trajLengths.pop(-1)
        self.sampleSize = 0;self.slice=0
        for length in self.trajLengths:
            self.sampleSize += length
            self.slice +=1
            if self.sampleSize > sampleSize:
                break


    def UpdatePriorities(self,priorities):
        self.priorities[0:self.slice] = priorities
        self.flag = True
        return self.buffer

    def GetReprioritySamples(self):
        return self.buffer[0:self.slice]
