
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


def _log(val):
    return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

class PPO_Hierarchy(Method):

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
        self.HPs=settings["HPs"]
        self.subReward = False
        self.UpdateSubpolicies = True
        self.nTrajs = nTrajs
        self.method = self.HPs["Method"]

        #Creating two buffers to separate information between the different levels of the network.
        if self.subReward:
            self.buffer = [Trajectory(depth=12) for _ in range(nTrajs)]
            #[s0,a,r,r_sub,s1,done]+[HL_actions, HL_log_logits, HL_v, flag, critics, logits]
        else:
            self.buffer = [Trajectory(depth=11) for _ in range(nTrajs)]
            #[s0,a,r,s1,done]+[HL_action, HL_log_logits, HL_v, flag, critics, logits]

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):
                #Generic placeholders
                self.s = tf.placeholder(tf.float32, [None]+stateShape, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.td_target_ = tf.placeholder(tf.float32, [None], 'Vtarget')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')

                #Initializing Netowrk I/O
                inputs = {"state":self.s}
                out = self.Model(inputs)
                self.a_prob = out["metaActor"]
                self.v = out["metaCritic"]
                self.log_logits = out["metaLogLogits"]

                self.sub_a_prob = out["subActor"]
                self.sub_log_logits = out["subLogLogits"]
                self.sub_v = out["subCritic"]

                self.nPolicies = len(self.sub_a_prob)

                #Placeholder for the Hierarchical Policy
                self.old_log_logits_ = tf.placeholder(shape=[None, self.nPolicies], dtype=tf.float32, name='old_logit_hold')
                #Placeholder for the Sub-Policies
                self.old_log_logits_sub_ = tf.placeholder(shape=[None, actionSize], dtype=tf.float32, name='old_logit_sub_hold')

                # Creating the Loss and update calls for the Hierarchical policy
                self.hierarchicalLoss = self.CreateLossPPO(self.a_prob,self.td_target_,self.v,self.a_his,self.log_logits,self.old_log_logits_,self.advantage_,self.nPolicies)
                variables = self.Model.getHierarchyVariables()
                self.hierarchyUpdater = self.CreateUpdater(self.hierarchicalLoss,variables)

                # Creating the Losses updaters for the Sub-policies.
                self.subpolicyLoss = []
                self.subpolicyUpdater = []
                for i in range(self.nPolicies):
                    loss = self.CreateLossPPO(self.sub_a_prob[i],self.td_target_,self.sub_v[i],self.a_his,self.sub_log_logits[i],self.old_log_logits_sub_,self.advantage_,self.actionSize)
                    self.subpolicyLoss.append(loss)
                    variables = self.Model.getSubpolicyVariables(i)
                    self.subpolicyUpdater.append(self.CreateUpdater(loss,variables))

            #Creating Variables for teh purpose of logging.
            self.SubpolicyDistribution = MovingAverage(1000)

    def CreateUpdater(self,loss,variables):
        optimizer = tf.keras.optimizers.Adam(self.HPs["LR"])
        gradients = optimizer.get_gradients(loss,variables)
        return optimizer.apply_gradients(zip(gradients,variables))

    def CreateLossPPO(self,a_prob,td_target_,v,a_his,log_logits,old_log_logits_,advantage_,actionSize):
        # Entropy
        entropy = -tf.reduce_mean(a_prob * _log(a_prob), name='entropy')

        # Critic Loss
        td_error = td_target_ - v
        critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

        # Actor Loss
        action_OH = tf.one_hot(a_his, actionSize, dtype=tf.float32)
        log_prob = tf.reduce_sum(log_logits * action_OH, 1)
        old_log_prob = tf.reduce_sum(old_log_logits_ * action_OH, 1)

        # Clipped surrogate function
        ratio = tf.exp(log_prob - old_log_prob)
        surrogate = ratio * advantage_
        clipped_surrogate = tf.clip_by_value(ratio, 1-self.HPs["eps"], 1+self.HPs["eps"]) * advantage_
        surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
        actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

        actor_loss = actor_loss - entropy * self.HPs["EntropyBeta"]
        loss = actor_loss + critic_loss * self.HPs["CriticBeta"]
        return loss

    def InitiateEpisode(self):
        if self.method == "Greedy":
            pass
        elif self.method == "Fixed Step":
            self.counter = 1
            self.nStep = 4

        elif self.method == "Constant":
            pass

        elif self.method == "Confidence":
            self.pastActions = [None]*self.nTrajs

        elif self.method == "Probabilistic Confidence":
            pass

        else:
            pass
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
            self.InitiateEpisode()

        # Run the Meta and Sub-policy Networks
        targets = [self.a_prob,self.log_logits,self.v]+self.sub_a_prob+self.sub_log_logits+self.sub_v
        res = self.sess.run(targets, {self.s: state})
        LL_probs=res[0]
        HL_log_logits =res[1]
        HL_v = res[2]
        sub_probs = res[3:3+self.nPolicies]
        sub_log_logits = res[3+self.nPolicies:3+2*self.nPolicies]
        sub_v = res[3+2*self.nPolicies:]

        if self.method == "Greedy":
            HL_actions = np.array([np.random.choice(LL_probs.shape[1], p=prob / sum(prob)) for prob in LL_probs])
            flag=[True]*state.shape[0]
        elif self.method == "Fixed Step":
            if self.counter == self.nStep:
                #Reseting Step counter and selecting New option
                self.counter = 1
            if self.counter == 1:
                HL_actions = np.array([np.random.choice(LL_probs.shape[1], p=prob / sum(prob)) for prob in LL_probs])
                self.traj_action = HL_actions
                flag=[True]*state.shape[0]
            else:
                HL_actions = self.traj_action
                flag=[False]*state.shape[0]
            self.counter +=1

        elif self.method == "Confidence":
            flag = []
            HL_actions = []
            confids = -np.mean(LL_probs * np.log(LL_probs), axis=1)
            for i,confid in enumerate(confids):
                if confid < 0.1 or step==0:
                    action = np.random.choice(LL_probs.shape[1], p=LL_probs[i] / sum(LL_probs[i]))
                    HL_actions.append(action)
                    self.pastActions[i] = action
                    flag.append(True)
                else:
                    HL_actions.append(self.pastActions[i])
                    flag.append(True)
            self.traj_action = HL_actions

        elif self.method == "Probabilistic Confidence":
            pass
        else:
            pass

        # Run the Subpolicy Network
        actions = np.array([np.random.choice(self.actionSize, p=sub_probs[mod][idx] / sum(sub_probs[mod][idx])) for idx, mod in enumerate(HL_actions)])
        critics = [sub_v[mod][idx] for idx, mod in enumerate(HL_actions)]
        logits = [sub_log_logits[mod][idx] for idx, mod in enumerate(HL_actions)]

        return actions, [HL_actions, HL_log_logits, HL_v, flag, critics, logits]

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

            td_target, advantage, td_target_hier, advantage_hier,actions_hier,ll_hier,s_hier = self.ProcessBuffer(HPs,traj)
            # Updating the Hierarchical Controller
            for epoch in range(self.HPs["Epochs"]):
                for batch in MultiBatchDivider([s_hier,actions_hier,td_target_hier,advantage_hier,ll_hier],self.HPs["MinibatchSize"]):

                    feed_dict = {self.s: np.asarray(batch[0]).squeeze(),
                                 self.a_his: np.asarray(batch[1]).squeeze(),
                                 self.td_target_: np.asarray(batch[2]).squeeze(),
                                 self.advantage_: np.reshape(batch[3], [-1]),
                                 self.old_log_logits_: np.asarray(batch[4]).squeeze()}
                    self.sess.run(self.hierarchyUpdater, feed_dict)

            if self.UpdateSubpolicies:
                #Collecting the data into different sub-Policies
                if self.subReward:
                    tmp, l1, l2, l3, l4, l5 = (list(t) for t in zip(*sorted(zip(self.buffer[traj][6], self.buffer[traj][0], self.buffer[traj][1], td_target, advantage, self.buffer[traj][10]),key=lambda x: x[0]))) #Sorting by the value in the actions_hier
                    #dividing at the splits
                    for subpolicyNum,data in SubpolicyIterator(tmp,[l1, l2, l3, l4, l5]):
                        #Updating each of the sub-policies.
                        for epoch in range(self.HPs["Epochs"]):
                            for batch in MultiBatchDivider(data,self.HPs["MinibatchSize"]):

                                feed_dict = {self.s: np.asarray(batch[0]).squeeze(),
                                            self.a_his: np.asarray(batch[1]).squeeze(),
                                            self.td_target_: np.asarray(batch[2]).squeeze(),
                                            self.advantage_: np.reshape(batch[3], [-1]),
                                            self.old_log_logits_sub_: np.asarray(batch[4]).squeeze()}
                                self.sess.run(self.subpolicyUpdater[subpolicyNum], feed_dict)
                    self.SubpolicyDistribution.extend(np.asarray(self.buffer[traj][6]))
                else:
                    tmp, l1, l2, l3, l4, l5 = (list(t) for t in zip(*sorted(zip(self.buffer[traj][5], self.buffer[traj][0], self.buffer[traj][1], td_target, advantage, self.buffer[traj][10]),key=lambda x: x[0]))) #Sorting by the value in the actions_hier
                    #dividing at the splits
                    for subpolicyNum,data in SubpolicyIterator(tmp,[l1, l2, l3, l4, l5]):
                    #Updating each of the sub-policies.
                        for epoch in range(self.HPs["Epochs"]):
                            for batch in MultiBatchDivider(data,self.HPs["MinibatchSize"]):

                                feed_dict = {self.s: np.asarray(batch[0]).squeeze(),
                                             self.a_his: np.asarray(batch[1]).squeeze(),
                                             self.td_target_: np.asarray(batch[2]).squeeze(),
                                             self.advantage_: np.reshape(batch[3], [-1]),
                                             self.old_log_logits_sub_: np.asarray(batch[4]).squeeze()}
                                self.sess.run(self.subpolicyUpdater[subpolicyNum], feed_dict)
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

    def ProcessBuffer(self,HPs,traj):
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
        if self.subReward:
            #Stuff need to be processed for the Low Level Controllers
            reward_lists = np.split(self.buffer[traj][2],split_loc[:-1])
            sub_reward_lists = np.split(self.buffer[traj][3],split_loc[:-1])
            value_lists = np.split(self.buffer[traj][10],split_loc[:-1])

            #Stuff needed for the
            HL_Critic_lists = np.split(self.buffer[traj][8],split_loc[:-1])
            HL_Logits_lists = np.split(self.buffer[traj][7],split_loc[:-1])
            HL_action_lists = np.split(self.buffer[traj][6],split_loc[:-1])
            HL_flag_lists = np.split(self.buffer[traj][9],split_loc[:-1])

            td_target=[]; advantage=[]
            td_target_hier=[]; advantage_hier=[]
            ll=[];actions=[]

            for rew,s_rew,value,HL_critic,HL_ll,HL_a,HL_flag,HL_s in zip(reward_lists,sub_reward_lists,value_lists,HL_Critic_lists,HL_Logits_lists,HL_action_lists,HL_flag_lists,HL_S_lists):
                # Calculating the per step advantage of each of the different sections
                td_target_i, advantage_i = gae(s_rew.reshape(-1).tolist(),value.reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
                td_target.extend(td_target_i); advantage.extend( advantage_i)
                #Colapsing different trajectory lengths for the hierarchical controller
                split_loc_ = [i+1 for i, x in enumerate(HL_flag[:-1]) if x]
                rew_hier = [np.sum(l) for l in np.split(rew,split_loc_)]
                value_hier = [l[0] for l in np.split(HL_critic,split_loc_)]
                actions.extend([l[0] for l in np.split(HL_a,split_loc_)])
                ll.extend([l[0] for l in np.split(HL_ll,split_loc_)])
                s.extend([l[0] for l in np.split(HL_s,split_loc_)])
                #Calculating the td_target and advantage for the hierarchical controller.
                td_target_i_, advantage_i_ = gae(np.asarray(rew_hier).reshape(-1).tolist(),np.asarray(value_hier).reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
                td_target_hier.extend(td_target_i_); advantage_hier.extend( advantage_i_)

            return td_target, advantage, td_target_hier, advantage_hier,actions,ll
        else:

            #Stuff need to be processed for the Low Level Controllers
            reward_lists = np.split(self.buffer[traj][2],split_loc[:-1])
            value_lists = np.split(self.buffer[traj][9],split_loc[:-1])

            #Stuff needed for the
            HL_S_lists = np.split(self.buffer[traj][0],split_loc[:-1])
            HL_Critic_lists = np.split(self.buffer[traj][7],split_loc[:-1])
            HL_Logits_lists = np.split(self.buffer[traj][6],split_loc[:-1])
            HL_action_lists = np.split(self.buffer[traj][5],split_loc[:-1])
            HL_flag_lists = np.split(self.buffer[traj][8],split_loc[:-1])

            td_target=[]; advantage=[]
            td_target_hier=[]; advantage_hier=[]
            ll=[];actions=[];s=[]

            for rew,value,HL_critic,HL_ll,HL_a,HL_flag,HL_s in zip(reward_lists,value_lists,HL_Critic_lists,HL_Logits_lists,HL_action_lists,HL_flag_lists,HL_S_lists):
                # Calculating the per step advantage of each of the different sections
                td_target_i, advantage_i = gae(rew.reshape(-1).tolist(),value.reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
                td_target.extend(td_target_i); advantage.extend( advantage_i)
                #Colapsing different trajectory lengths for the hierarchical controller
                split_loc_ = [i+1 for i, x in enumerate(HL_flag[:-1]) if x]
                rew_hier = [np.sum(l) for l in np.split(rew,split_loc_)]
                value_hier = [l[0] for l in np.split(HL_critic,split_loc_)]
                actions.extend([l[0] for l in np.split(HL_a,split_loc_)])
                ll.extend([l[0] for l in np.split(HL_ll,split_loc_)])
                s.extend([l[0] for l in np.split(HL_s,split_loc_)])
                #Calculating the td_target and advantage for the hierarchical controller.
                td_target_i_, advantage_i_ = gae(np.asarray(rew_hier).reshape(-1).tolist(),np.asarray(value_hier).reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
                td_target_hier.extend(td_target_i_); advantage_hier.extend( advantage_i_)

            return td_target, advantage, td_target_hier, advantage_hier,actions,ll,s

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
