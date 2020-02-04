
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

    def __init__(self,Model,stateShape,actionSize,HPs,nTrajs=1,scope="PPO_Training"):
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
        self.Model = Model
        self.HPs=HPs

        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]

        #Initializing Netowrk I/O
        inputs = {"state":tf.convert_to_tensor(np.random.random([1]+stateShape), dtype=tf.float32)}
        out = self.Model(inputs)
        self.Model.summary()
        # print(out)

        self.optimizer = tf.keras.optimizers.Adam(HPs["LR"])

    @tf.function
    def ActorLoss(self,a_his,log_logits,old_log_logits,advantage):
        action_OH = tf.one_hot(a_his, self.actionSize, dtype=tf.float32)
        log_prob = tf.reduce_sum(log_logits * action_OH, 1)
        old_log_prob = tf.reduce_sum(old_log_logits * action_OH, 1)
        ratio = tf.exp(log_prob - old_log_prob)
        surrogate = ratio * advantage
        clipped_surrogate = tf.clip_by_value(ratio, 1-self.HPs["eps"], 1+self.HPs["eps"]) * advantage
        surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
        actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')
        return actor_loss

    @tf.function
    def CriticLoss(self,v,td_target):
        td_error = td_target - v
        critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
        return critic_loss

    @tf.function
    def EntropyLoss(self,a_prob):
        log_prob = tf.math.log(tf.clip_by_value(a_prob, 1e-10, 10.0))
        entropy = -tf.reduce_mean(a_prob * log_prob, name='entropy')
        return entropy

    def GetAction(self, state):
        """
        Method to run data through the neural network. Catches if malformed data is passed through the network.
        ToDo: Change data checks to be performed on the Model/Network. Have checks built into the JSON file.

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
        if len(state.shape) == 3:
            state = np.expand_dims(state,axis=0)
        inputs = {"state":tf.convert_to_tensor(state, dtype=tf.float32)}
        out = self.Model(inputs)
        probs = out["actor"]
        v = out["critic"]
        log_logits = out["log_logits"]

        actions = tf.random.categorical(probs, 1)
        return actions, [v,log_logits]

    def Update(self,HPs):
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

            clip = -1
            try:
                for j in range(1):
                    clip = self.buffer[traj][4].index(True, clip + 1)
            except:
                clip=len(self.buffer[traj][4])

            #Staging Buffer inputs into the entries to run through the network.
            if len(self.buffer[traj][0][:clip]) == 0:
                continue

            td_target, advantage = self.ProcessBuffer(HPs,traj,clip)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?

            # print(self.buffer[traj][1][:clip].shape)
            with tf.GradientTape() as tape:
                s= self.buffer[traj][0][:clip]
                a_his= np.asarray(self.buffer[traj][1][:clip]).reshape(-1)
                td_target= np.asarray(td_target[:clip]).reshape(-1)
                advantage= np.reshape(advantage, [-1])
                old_log_logits= np.reshape(self.buffer[traj][6][:clip], [-1,self.actionSize])
                inputs = {"state":tf.convert_to_tensor(s, dtype=tf.float32)}
                out = self.Model(inputs)
                v = out["critic"][:clip]
                # print(v)
                a_prob = out["actor"][:clip]
                # print(td_target)

                actor_loss = self.ActorLoss(a_his,a_prob,old_log_logits,advantage)
                critic_loss = self.CriticLoss(v,td_target)
                entropy = self.EntropyLoss(a_prob)
                actor_loss = actor_loss - entropy * HPs["EntropyBeta"]
                loss = actor_loss + critic_loss * HPs["CriticBeta"]
                gradients = tape.gradient(loss, self.Model.trainable_variables)

            # gradients = self.optimizer.get_gradients(loss, self.Model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.Model.trainable_weights))


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
