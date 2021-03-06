
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
        self.HPs = HPs

        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=8) for _ in range(nTrajs)]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):
                #Placeholders
                if len(stateShape) == 4:
                    self.s = tf.placeholder(tf.float32, [None]+stateShape[1:4], 'S')
                elif len(stateShape) == 3:
                    self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.td_target_ = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_log_logits_ = tf.placeholder(shape=[None, actionSize], dtype=tf.float32, name='old_logit_hold')

                #Introduce logic to get the hidden size and shape Also intrduce logic for multiple LSTM Cells.
                self.hidden_state_prev = tf.placeholder(tf.float32, [None,256], 'hidden_state_prev')
                self.hidden_cell_prev = tf.placeholder(tf.float32, [None,256], 'hidden_cell_prev')

                #Initializing Netowrk I/O
                inputs = {"state":self.s,
                            "hiddenState":[self.hidden_state_prev,self.hidden_cell_prev]}
                out = self.Model(inputs)
                print(out)
                self.a_prob = out["actor"]
                self.v = out["critic"]
                self.log_logits = out["log_logits"]
                self.hidden_state = out["hiddenState"]
                self.cell_state = out["hiddenCell"]
                # print("--------Hidden State",self.hidden_state)

                # Entropy
                def _log(val):
                    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))
                entropy =self.entropy= -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')

                # Critic Loss
                td_error = self.td_target_ - self.v
                critic_loss =self.critic_loss= tf.reduce_mean(tf.square(td_error), name='critic_loss')

                # Actor Loss
                action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
                log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
                old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

                # Clipped surrogate function
                ratio = tf.exp(log_prob - old_log_prob)
                surrogate = ratio * self.advantage_
                clipped_surrogate = tf.clip_by_value(ratio, 1-HPs["eps"], 1+HPs["eps"]) * self.advantage_
                surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
                actor_loss =self.actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

                actor_loss = actor_loss - entropy * HPs["EntropyBeta"]
                loss = actor_loss + critic_loss * HPs["CriticBeta"]

                # Build Trainer
                self.optimizer = tf.keras.optimizers.Adam(HPs["LearningRate"])
                self.gradients = self.optimizer.get_gradients(loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
                self.update_ops = self.optimizer.apply_gradients(zip(self.gradients, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)))
                self.EntropyMA = MovingAverage(400)
                self.CriticLossMA = MovingAverage(400)
                self.ActorLossMA = MovingAverage(400)
                self.GradMA = MovingAverage(400)

    def GetAction(self, state, episode=1, step=1):
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
        if step==0:
            #Initializing Hidden State of the LSTM
            hidden_state_prev =[np.zeros([1,256]),np.zeros([1,256])]
        else:
            hidden_state_prev = self.store.copy()

        try:
            probs,log_logits,v,hidden_state,hidden_cell = self.sess.run([self.a_prob,self.log_logits,self.v,self.hidden_state,self.cell_state], {self.s: state, self.hidden_state_prev:hidden_state_prev[0],self.hidden_cell_prev:hidden_state_prev[1]})

        except ValueError:
            probs,log_logits,v,hidden_state,hidden_cell = self.sess.run([self.a_prob,self.log_logits,self.v,self.hidden_state,self.cell_state], {self.s: np.expand_dims(state,axis=0), self.hidden_state_prev:hidden_state_prev[0],self.hidden_cell_prev:hidden_state_prev[1]})

        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])

        self.store = [hidden_state,hidden_cell]
        test = np.array(hidden_state_prev)

        # print(probs,actions)
        return actions, [v,log_logits,test]

    def Update(self,episode=1):
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



            td_target, advantage = self.ProcessBuffer(self.HPs,traj,-1)
            # print(td_target)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?
            hidden_state_prev = []
            hidden_cell_prev = []
            for element in self.buffer[traj][7]:
                hidden_state_prev.append(element[0])
                hidden_cell_prev.append(element[1])

            #Staging Buffer inputs into the entries to run through the network.
            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1
            s= np.array_split(np.squeeze(self.buffer[traj][0]), batches)
            a_his = np.array_split( np.asarray(self.buffer[traj][1]).reshape(-1), batches)
            hidden_state_prev_ = np.array_split(np.asarray(hidden_state_prev).reshape(-1,256), batches)
            hidden_cell_prev_ = np.array_split(np.asarray(hidden_cell_prev).reshape(-1,256), batches)
            td_target_ = np.array_split(td_target, batches)
            advantage_= np.array_split(np.reshape(advantage, [-1]), batches)
            old_log_logits_ = np.array_split(np.reshape(self.buffer[traj][6], [-1,self.actionSize]), batches)

            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):
                    feed_dict = {self.s: s[i],
                                 self.hidden_state_prev:hidden_state_prev_[i],
                                 self.hidden_cell_prev:hidden_cell_prev_[i],
                                 self.a_his: a_his[i],
                                 self.td_target_: td_target_[i],
                                 self.advantage_: advantage_[i],
                                 self.old_log_logits_: old_log_logits_[i]}

                    self.sess.run(self.update_ops, feed_dict)

                    ops = [self.actor_loss, self.critic_loss, self.entropy]
                    aLoss, cLoss, entropy = self.sess.run(ops, feed_dict)
                    grads = self.sess.run(self.gradients, feed_dict)
                    total_counter = 0
                    vanish_counter = 0
                    for grad in grads:
                        total_counter += np.prod(grad.shape)
                        vanish_counter += (np.absolute(grad)<1e-8).sum()

                    self.EntropyMA.append(entropy)
                    self.CriticLossMA.append(cLoss)
                    self.ActorLossMA.append(aLoss)
                    self.GradMA.append(vanish_counter/total_counter)
        self.ClearTrajectory()


    def GetStatistics(self):
        dict = {"Training Results/Entropy":self.EntropyMA(),
        "Training Results/Loss Critic":self.CriticLossMA(),
        "Training Results/Loss Actor":self.ActorLossMA(),
        "Training Results/Vanishing Gradient":self.GradMA(),}
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
