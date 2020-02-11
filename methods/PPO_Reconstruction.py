
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
                self.state_next = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='State_next')

                #Initializing Netowrk I/O
                inputs = {"state":self.s}
                out = self.Model(inputs)
                self.a_prob = out["actor"]
                self.v = out["critic"]
                self.log_logits = out["log_logits"]
                self.state_pred = out["prediction"]

                # Entropy
                def _log(val):
                    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))
                entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')

                # Critic Loss
                td_error = self.td_target_ - self.v
                self.c_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

                # Actor Loss
                action_OH = tf.one_hot(self.a_his, actionSize, dtype=tf.float32)
                log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
                old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

                # Clipped surrogate function
                ratio = tf.exp(log_prob - old_log_prob)
                surrogate = ratio * self.advantage_
                clipped_surrogate = tf.clip_by_value(ratio, 1-HPs["eps"], 1+HPs["eps"]) * self.advantage_
                surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
                actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

                # self.s_loss = tf.losses.mean_squared_error(self.state_pred,self.state_next)
                self.s_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.state_pred,self.state_next))))

                self.a_loss = actor_loss - entropy * HPs["EntropyBeta"]

                # Build Trainer
                self.c_optimizer = tf.keras.optimizers.Adam(HPs["Critic LR"])
                self.c_grads = self.c_optimizer.get_gradients(self.c_loss, self.Model.GetVariables("Critic"))
                self.update_c_op = self.c_optimizer.apply_gradients(zip(self.c_grads, self.Model.GetVariables("Critic")))

                self.a_optimizer = tf.keras.optimizers.Adam(HPs["Actor LR"])
                self.a_grads = self.a_optimizer.get_gradients(self.a_loss, self.Model.GetVariables("Actor"))
                self.update_a_op = self.a_optimizer.apply_gradients(zip(self.a_grads, self.Model.GetVariables("Actor")))

                self.s_optimizer = tf.keras.optimizers.Adam(HPs["State LR"])
                self.s_grads = self.s_optimizer.get_gradients(self.s_loss, self.Model.GetVariables("Reconstruction"))
                self.update_s_op = self.s_optimizer.apply_gradients(zip(self.s_grads, self.Model.GetVariables("Reconstruction")))

                self.update_ops = [self.update_a_op,self.update_c_op,self.update_s_op]
                self.grads = [self.a_grads,self.c_grads,self.s_grads]
                self.losses = [self.a_loss,self.c_loss,self.s_loss]

                self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
                self.loss_MA = [MovingAverage(400) for i in range(len(self.grads))]
                self.labels = ["Actor","Critic","State"]

    def GetAction(self, state, episode=0, step=0):
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

    def Update(self,HPs,episode=0,statistics=True):
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

            td_target, advantage = self.ProcessBuffer(HPs,traj,clip)

            #Create a dictionary with all of the samples?
            #Use a sampler to feed the update operation?

            #Staging Buffer inputs into the entries to run through the network.
            if len(self.buffer[traj][0][:clip]) == 0:
                continue
            feed_dict = {self.s: self.buffer[traj][0][:clip],
                         self.state_next: self.buffer[traj][3][:clip],
                         self.a_his: np.asarray(self.buffer[traj][1][:clip]).reshape(-1),
                         self.td_target_: td_target,
                         self.advantage_: np.reshape(advantage, [-1]),
                         self.old_log_logits_: np.reshape(self.buffer[traj][6][:clip], [-1,self.actionSize])}
            if not statistics:
                self.sess.run(self.update_ops, feed_dict)
            else:
                #Perform update operations
                out = self.sess.run(self.update_ops+self.losses+self.grads, feed_dict)   # local grads applied to global net.
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
                        vanish_counter += (np.absolute(grad)<1e-8).sum()
                    self.grad_MA[i].append(vanish_counter/total_counter)



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
    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
        return dict

    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")
