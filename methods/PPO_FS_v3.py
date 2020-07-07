
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory,BatchDivider,MultiBatchDivider
from .AdvantageEstimator import gae
import tensorflow as tf
import numpy as np
import scipy
from utils.record import Record
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
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)
        scope="PPO"

        #Creating appropriate buffer for the method.
        self.buffer = [Trajectory(depth=8) for _ in range(nTrajs)]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):
                #Placeholders
                self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.td_target_ = tf.placeholder(tf.float32, [None], 'TD_target')
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
                clipped_surrogate = tf.clip_by_value(ratio, 1-self.HPs["eps"], 1+self.HPs["eps"]) * self.advantage_
                surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
                actor_loss = self.actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

                loss = self.actor_loss + self.critic_loss * self.HPs["CriticBeta"]

                # Build Trainer
                if self.HPs["Optimizer"] == "Adam":
                    self.optimizerA = tf.keras.optimizers.Adam(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.Adam(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "RMS":
                    self.optimizerA = tf.keras.optimizers.RMSProp(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.RMSProp(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "Adagrad":
                    self.optimizerA = tf.keras.optimizers.Adagrad(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.Adagrad(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "Adadelta":
                    self.optimizerA = tf.keras.optimizers.Adadelta(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.Adadelta(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "Adamax":
                    self.optimizerA = tf.keras.optimizers.Adamax(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.Adamax(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "Nadam":
                    self.optimizerA = tf.keras.optimizers.Nadam(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.Nadam(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "SGD":
                    self.optimizerA = tf.keras.optimizers.SGD(self.HPs["LR Actor"])
                    self.optimizerE = tf.keras.optimizers.SGD(self.HPs["LR Entropy"])
                elif self.HPs["Optimizer"] == "Amsgrad":
                    self.optimizerA = tf.keras.optimizers.Nadam(self.HPs["LR Actor"],amsgrad=True)
                    self.optimizerE = tf.keras.optimizers.Nadam(self.HPs["LR Entropy"],amsgrad=True)
                else:
                    print("Not selected a proper Optimizer")
                    exit()
                a_params = self.Model.GetVariables("Actor")
                c_params = self.Model.GetVariables("Critic")
                self.gradients_a = self.optimizerA.get_gradients(loss, self.Model.trainable_variables)
                # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients_a]
                self.update_op_a = self.optimizerA.apply_gradients(zip(self.gradients_a , self.Model.trainable_variables))

                entropy_loss = -self.entropy * self.HPs["EntropyBeta"]
                self.gradients_e = self.optimizerE.get_gradients(entropy_loss, a_params)
                self.update_op_e = self.optimizerE.apply_gradients(zip(self.gradients_e, a_params))


                total_counter = 1
                vanish_counter = 0
                for gradient in self.gradients_a:
                    total_counter += np.prod(gradient.shape)
                    stuff = tf.reduce_sum(tf.cast(tf.math.less_equal(tf.math.abs(gradient),tf.constant(1e-8)),tf.int32))
                    vanish_counter += stuff
                self.vanishing_gradient = vanish_counter/total_counter


        self.update_ops=[self.update_op_a,self.update_op_e]
        self.logging_ops=[self.actor_loss,self.critic_loss,self.entropy,tf.reduce_mean(self.advantage_),tf.reduce_mean(ratio),loss, self.vanishing_gradient]
        self.labels = ["Loss Actor","Loss Critic","Entropy","Advantage","PPO Ratio","Loss Total","Vanishing Gradient"]
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

        if step % self.HPs["FS"] == 0:
            self.store_actions = actions
            return actions, [v,log_logits,True]
        else:
            return self.store_actions, [v,log_logits,False]

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
        #Counting number of samples.
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        for traj in range(len(self.buffer)):

            td_target_hier, advantage_hier,actions_hier,ll_hier,s_hier = self.ProcessBuffer(traj)

            for epoch in range(self.HPs["Epochs"]):
                for batch in MultiBatchDivider([s_hier,actions_hier,td_target_hier,advantage_hier,ll_hier],self.HPs["MinibatchSize"]):
                    #Staging Buffer inputs into the entries to run through the network.
                    feedDict = {self.s: np.asarray(batch[0]).squeeze(),
                                 self.a_his: np.asarray(batch[1]).squeeze(),
                                 self.td_target_: np.asarray(batch[2]).squeeze(),
                                 self.advantage_: np.reshape(batch[3], [-1]),
                                 self.old_log_logits_: np.asarray(batch[4]).squeeze()}
                    out = self.sess.run(self.update_ops+self.logging_ops, feedDict)   # local grads applied to global net.
                    logging = out[len(self.update_ops):]

                    for i,log in enumerate(logging):
                        self.logging_MA[i].append(log)

        self.ClearTrajectory()

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

        Returns
        -------
        td_target : list
            List Temporal Difference Target for particular states.
        advantage : list
            List of advantages for particular actions.
        """
        # Split into different episodes based on the "done" signal. Assumes that episode terminates at done.
        # Cannot account for instances where there are multiple done signals in a row.

        split_loc = [i+1 for i, x in enumerate(self.buffer[traj][4]) if x]

        # reward_lists = np.split(self.buffer[traj][2],split_loc)
        # value_lists = np.split(self.buffer[traj][5],split_loc)
        #
        # td_target=[]; advantage=[]
        # for rew,value in zip(reward_lists,value_lists):
        #     td_target_i, advantage_i = gae(rew.reshape(-1).tolist(),value.reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
        #     td_target.extend(td_target_i); advantage.extend( advantage_i)
        # return td_target, advantage


        reward_lists = np.split(self.buffer[traj][2],split_loc[:-1])

        #Stuff needed for the
        HL_S_lists = np.split(self.buffer[traj][0],split_loc[:-1])
        HL_Critic_lists = np.split(self.buffer[traj][5],split_loc[:-1])
        HL_Logits_lists = np.split(self.buffer[traj][6],split_loc[:-1])
        HL_action_lists = np.split(self.buffer[traj][1],split_loc[:-1])
        HL_flag_lists = np.split(self.buffer[traj][7],split_loc[:-1])

        td_target_hier=[]; advantage_hier=[]
        ll=[];actions=[];s=[]

        for rew,HL_critic,HL_ll,HL_a,HL_flag,HL_s in zip(reward_lists,HL_Critic_lists,HL_Logits_lists,HL_action_lists,HL_flag_lists,HL_S_lists):
            #Colapsing different trajectory lengths for the hierarchical controller
            split_loc_ = [i for i, x in enumerate(HL_flag[:-1]) if x][1:]
            rew_hier = [np.sum(l) for l in np.split(rew,split_loc_)]
            value_hier = [l[0] for l in np.split(HL_critic,split_loc_)]
            actions.extend([l[0] for l in np.split(HL_a,split_loc_)])
            ll.extend([l[0] for l in np.split(HL_ll,split_loc_)])
            s.extend([l[0] for l in np.split(HL_s,split_loc_)])
            #Calculating the td_target and advantage for the hierarchical controller.
            td_target_i_, advantage_i_ = gae(np.asarray(rew_hier).reshape(-1).tolist(),np.asarray(value_hier).reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
            td_target_hier.extend(td_target_i_); advantage_hier.extend( advantage_i_)

        return td_target_hier, advantage_hier,actions,ll,s


    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")
