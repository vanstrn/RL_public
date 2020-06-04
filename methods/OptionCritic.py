
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
from utils.utils import MovingAverage
import random

from networks.common import NetworkBuilder

def _log(val):
    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

class OptionCritic(Method):

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
        self.Model = NetworkBuilder(networkConfig=settings["NetworkConfig"],netConfigOverride=netConfigOverride,actionSize=actionSize)
        self.method = "Confidence" #Create input for this.
        self.HPs=settings["NetworkHPs"]
        self.subReward = False
        self.UpdateSubpolicies = True
        self.nTrajs = nTrajs

        #Creating buffer
        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]
        #[s0,a,r,s1,done]+[HL_action]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope("OptionCritic"):
                #Generic placeholders
                self.batch_size = tf.placeholder(tf.int32, 1, 'BS')
                self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.actions = tf.placeholder(tf.int32, [None, ], 'A')
                self.rewards = tf.placeholder(tf.float32, [None], 'R')
                # self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.options = tf.placeholder(shape=[None], dtype=tf.int32, name="options")

                batch_indexer = tf.range(tf.reshape(self.batch_size, []))

                #Initializing Netowrk I/O
                inputs = {"state":self.s}
                out = self.Model(inputs)
                self.term = out["metaTermination"]
                self.q = out["metaCritic"]

                self.sub_a_prob = out["subActor"]
                self.sub_log_logits = out["subLogLogits"]

                self.nPolicies = len(self.sub_a_prob)

                # Creating the Loss and update calls for the Hierarchical policy
                # Indexers
                self.responsible_options = tf.stack([batch_indexer, self.options], axis=1)
                self.responsible_actions = tf.stack([batch_indexer, self.actions], axis=1)
                self.network_indexer = tf.stack([self.options, batch_indexer], axis=1)

                # Q Values OVER options
                self.disconnected_q_vals = tf.stop_gradient(self.q)

                # Q values of each option that was taken
                self.responsible_opt_q_vals = tf.gather_nd(params=self.q, indices=self.responsible_options) # Extract q values for each option
                self.disconnected_q_vals_option = tf.gather_nd(params=self.disconnected_q_vals, indices=self.responsible_options)

                # Termination probability of each option that was taken
                self.terminations = tf.gather_nd(params=self.term, indices=self.responsible_options)

                # Q values for each action that was taken
                relevant_networks = tf.gather_nd(params=self.sub_a_prob, indices=self.network_indexer)
                relevant_networks = tf.nn.softmax(relevant_networks, dim=1)

                self.action_values = tf.gather_nd(params=relevant_networks, indices=self.responsible_actions)

                # Weighted average value
                option_eps = 0.001
                self.value = tf.reduce_max(self.q) * (1 - option_eps) + (option_eps * tf.reduce_mean(self.q))
                disconnected_value = tf.stop_gradient(self.value)

                # Losses; TODO: Why reduce sum vs reduce mean?
                vf_coef = 0.5
                self.value_loss = vf_coef * tf.reduce_mean(vf_coef * 0.5 * tf.square(self.rewards - self.responsible_opt_q_vals))
                self.policy_loss = tf.reduce_mean(_log(self.action_values) * (self.rewards - self.disconnected_q_vals_option))
                self.deliberation_costs=0.020
                self.termination_loss = tf.reduce_mean(self.terminations * ((self.disconnected_q_vals_option - disconnected_value) + self.deliberation_costs) )

                ent_coef = 0.01
                action_probabilities = self.sub_a_prob
                self.entropy = ent_coef * tf.reduce_mean(action_probabilities * _log(action_probabilities))

                self.loss = -self.policy_loss - self.entropy - self.value_loss - self.termination_loss

                variables = self.Model.getVars()
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "OptionCritic")
                optimizer = tf.keras.optimizers.Adam(self.HPs["LR"])
                gradients = optimizer.get_gradients(self.loss,variables)
                self.update_op = optimizer.apply_gradients(zip(gradients,variables))

            #Creating Variables for teh purpose of logging.
            self.SubpolicyDistribution = MovingAverage(1000)

    def GetAction(self, state, step,episode=0):
        """
        Method to run data through hierarchical network

        First run the state through the meta network to select subpolicy to use.
        Second run the state through the proper Subpolicy

        ToDo: Check if faster to run the entire network and select appropriate subpolicy afterwards. or run only the required bit.

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
        #Determine number of steps and whether to initiate confidence based on the length of the Buffer.
        if step == 0:
            self.pastActions = [None]*self.nTrajs

        # Run the Meta and Sub-policy Networks
        targets = [self.q,self.term]+self.sub_a_prob+self.sub_log_logits
        res = self.sess.run(targets, {self.s: state})
        q=res[0]
        terminations =res[1]
        sub_probs = res[2:3+self.nPolicies]
        sub_log_logits = res[2+self.nPolicies:2+2*self.nPolicies]
        HL_actions = []
        for i,term in enumerate(terminations):
            if step==0:
                action = np.argmax(q[i])
                HL_actions.append(action)
                self.pastActions[i] = action
            elif random.uniform(0,1) < term[self.pastActions[i]]:
                # action = np.argmax(q[i])
                action = random.randint(0,2)
                HL_actions.append(action)
                self.pastActions[i] = action
            else:
                action = random.randint(0,2)
                HL_actions.append(action)
                # HL_actions.append(self.pastActions[i])
        self.traj_action = HL_actions
        print(q,HL_actions)

        # Run the Subpolicy Network
        actions = np.array([np.random.choice(self.actionSize, p=sub_probs[mod][idx] / sum(sub_probs[mod][idx])) for idx, mod in enumerate(HL_actions)])
        logits = [sub_log_logits[mod][idx] for idx, mod in enumerate(HL_actions)]

        return actions, [HL_actions,q]

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
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        for traj in range(len(self.buffer)):
            advantage = self.ProcessBuffer(traj)
            # Updating the Hierarchical Controller
            for epoch in range(self.HPs["Epochs"]):
                for batch in MultiBatchDivider([self.buffer[traj][0],self.buffer[traj][1],advantage,self.buffer[traj][5]],self.HPs["MinibatchSize"]):

                    feed_dict = {self.batch_size: [np.asarray(batch[0]).squeeze().shape[0]],
                                 self.s: np.asarray(batch[0]).squeeze(),
                                 self.actions: np.asarray(batch[1]).squeeze(),
                                 self.rewards: np.asarray(batch[2]).squeeze(),
                                 self.options: np.reshape(batch[3], [-1])}
                    self.sess.run(self.update_op, feed_dict)
            self.SubpolicyDistribution.extend(np.asarray(self.buffer[traj][5]))
            self.ClearTrajectory()

    def GetStatistics(self):
        stats={}
        for i in range(self.nPolicies):
            length = len(self.SubpolicyDistribution.tolist())
            if length == 0:
                length=1
            stats["Subpolicy Use/"+str(i)] = self.SubpolicyDistribution.tolist().count(i)/length
        return stats

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
        #Splitting the buffer into different episodes based on the done tag.
        split_loc = [i+1 for i, x in enumerate(self.buffer[traj][4]) if x]
        #Stuff need to be processed for the Low Level Controllers
        reward_lists = np.split(self.buffer[traj][2],split_loc[:-1])
        value_lists = np.split(self.buffer[traj][6],split_loc[:-1])

        HL_action_lists = np.split(self.buffer[traj][5],split_loc[:-1])

        td_target=[]; advantage=[]

        for rew,value,options in zip(reward_lists,value_lists,HL_action_lists):
            # Calculating the per step advantage of each of the different sections
            val = []
            for i,option in enumerate(options):
                val.append(value[i,0,option])
            td_target_i, advantage_i = gae(rew.reshape(-1).tolist(),np.asarray(val).reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
            td_target.extend(td_target_i); advantage.extend( advantage_i)


        return advantage


    @property
    def getVars(self):
        return self.Model.getVars("PPO_Training")

def SubpolicyIterator(sortingList, dataLists):
    list = np.asarray(sortingList).squeeze().tolist()
    for num in set(list):
        res = []
        for dataList in dataLists:
            index_first = list.index(num)
            index_last = len(list) - 1 - list[::-1].index(num)
            res.append(dataList[index_first:index_last])

        yield num, res
