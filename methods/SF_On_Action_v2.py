
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
from utils.utils import MovingAverage
from utils.record import Record

class OffPolicySF(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,nTrajs=1):
        """
        Off policy Successor Representation using neural networks
        Does not create an action for the

        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.actionSize =actionSize
        self.HPs = HPs
        self.sess=sess
        self.scope=scope
        self.Model = sharedModel
        self.s = tf.placeholder(tf.float32, [None] + stateShape, 'S')
        self.a = tf.placeholder(tf.float32, [None,self.actionSize], 'A')
        self.s_next = tf.placeholder(tf.float32, [None] + stateShape, 'S_next')
        self.reward = tf.placeholder(tf.float32, [None, ], 'R')
        self.td_target = tf.placeholder(tf.float32, [None,self.Model.data["DefaultParams"]["SFSize"]], 'TDtarget')
        self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
        self.old_log_logits_ = tf.placeholder(shape=[None, actionSize], dtype=tf.float32, name='old_logit_hold')

        input = {"state":self.s,
                 "action":self.a}
        out = self.Model(input)
        self.value_pred = out["critic"]
        self.state_pred = out["prediction"]
        self.reward_pred = out["reward_pred"]
        self.phi = out["phi"]
        self.psi = out["psi"]
        self.a_prob = out["actor"]
        self.log_logits = out["log_logits"]

        self.buffer = [Trajectory(depth=7) for _ in range(nTrajs)]

        self.params = self.Model.getVars()

        with tf.name_scope('loss'):
            sf_error = tf.subtract(self.td_target, self.psi, name='TD_error')
            sf_error = tf.square(sf_error)
            self.c_loss = tf.reduce_mean(sf_error,name="sf_loss")

            if HPs["Loss"] == "MSE":
                self.s_loss = tf.losses.mean_squared_error(self.state_pred,self.s_next)
            elif HPs["Loss"] == "KL":
                self.s_loss = tf.losses.KLD(self.state_pred,self.s_next)
            elif HPs["Loss"] == "M4E":
                self.s_loss = tf.reduce_mean((self.state_pred-self.s_next)**4)

            self.r_loss = tf.losses.mean_squared_error(self.reward,tf.squeeze(self.reward_pred))

            # Entropy
            def _log(val):
                return tf.log(tf.clip_by_value(val, 1e-10, 10.0))
            entropy = self.entropy = -tf.reduce_mean(self.a_prob * _log(self.a_prob), name='entropy')
            # Actor Loss
            action_OH = tf.one_hot(self.a, actionSize, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(self.old_log_logits_ * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            surrogate = ratio * self.advantage_
            clipped_surrogate = tf.clip_by_value(ratio, 1-HPs["eps"], 1+HPs["eps"]) * self.advantage_
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            self.actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            self.loss = self.a_loss - entropy * HPs["EntropyBeta"] + self.s_loss + HPs["CriticBeta"]*self.c_loss + HPs["RewardBeta"]*self.r_loss

        if HPs["Optimizer"] == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(HPs["LR"])
        elif HPs["Optimizer"] == "RMS":
            self.optimizer = tf.keras.optimizers.RMSprop(HPs["LR"])
        elif HPs["Optimizer"] == "Adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(HPs["LR"])
        elif HPs["Optimizer"] == "Adadelta":
            self.optimizer = tf.keras.optimizers.Adadelta(HPs["LR"])
        elif HPs["Optimizer"] == "Adamax":
            self.optimizer = tf.keras.optimizers.Adamax(HPs["LR"])
        elif HPs["Optimizer"] == "Nadam":
            self.optimizer = tf.keras.optimizers.Nadam(HPs["LR"])
        elif HPs["Optimizer"] == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(HPs["LR"])
        elif HPs["Optimizer"] == "SGD-Nesterov":
            self.optimizer = tf.keras.optimizers.SGD(HPs["LR"],nesterov=True)
        elif HPs["Optimizer"] == "Amsgrad":
            self.optimizer = tf.keras.optimizers.Nadam(HPs["LR"],amsgrad=True)
        else:
            print("Not selected a proper Optimizer")
            exit()

        with tf.name_scope('local_grad'):
            self.grads = self.optimizer.get_gradients(self.loss, self.params)

        with tf.name_scope('update'):
            self.update_op = self.optimizer.apply_gradients(zip(self.grads, self.params))


        self.update_ops = [self.update_op]
        self.grads = [self.grads]
        self.losses = [self.c_loss,self.s_loss,self.r_loss,self.a_loss]

        self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
        self.loss_MA = [MovingAverage(400) for i in range(len(self.losses))]
        self.Gradlabels = ["Total"]
        self.Losslabels = ["Critic","State","Reward"]

        self.clearBuffer = False

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
            probs,log_logits,v,phi,psi = self.sess.run([self.a_prob,self.log_logits,self.value_pred,self.phi, self.psi], {self.s: state})
        except ValueError:
            probs,log_logits,v,phi,psi = self.sess.run([self.a_prob,self.log_logits,self.value_pred,self.phi, self.psi], {self.s: np.expand_dims(state,axis=0)})
        actions = np.array([np.random.choice(probs.shape[1], p=prob / sum(prob)) for prob in probs])
        return actions, [v,log_logits,phi,psi]

    def PredictValue(self,state):
        s = state
        # s = state[np.newaxis,:]
        out = self.sess.run(self.value_pred, {self.s: s})
        return out
    def PredictState(self,state,action):
        s = state
        # s = state[np.newaxis,:]
        out = self.sess.run(self.state_pred, {self.s: s,self.a:action})
        return out
    def PredictReward(self,state):
        s = state
        # s = state[np.newaxis,:]
        out = self.sess.run(self.reward_pred, {self.s: s})
        return out

    def Update(self,episode=0,statistics=True):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Process the data from the buffer
        samples=0
        for i in range(len(self.buffer)):
            samples +=len(self.buffer[i])
        if samples < self.HPs["BatchSize"]:
            return

        for traj in range(len(self.buffer)):

            clip = -1
            td_target, advantage_, psi_target_ = self.ProcessBuffer(traj)

            acts = np.zeros([np.asarray(self.buffer[traj][1][:clip]).size,self.actionSize])
            acts[np.arange(np.asarray(self.buffer[traj][1][:clip]).size),np.asarray(self.buffer[traj][1][:clip]).reshape(-1)] = 1

            batches = len(self.buffer[traj][0][:clip])//self.HPs["MinibatchSize"]+1
            if "StackedDim" in self.HPs:
                if self.HPs["StackedDim"] > 1:
                    s_next = np.array_split(np.squeeze(self.buffer[traj][3][:clip])[:,:,:,-self.HPs["StackedDim"]:],3,batches)
                else:
                    s_next = np.array_split(np.expand_dims(np.stack(self.buffer[traj][3][:clip])[:,:,:,-self.HPs["StackedDim"]],3),batches)
            else:
                s_next = np.array_split(self.buffer[traj][3][:clip],batches)
            s = np.array_split(self.buffer[traj][0][:clip], batches)
            reward = np.array_split(np.asarray(self.buffer[traj][2][:clip]).reshape(-1),batches)
            actions = np.array_split(acts,batches)
            psi_target = np.array_split(np.squeeze(psi_target_),batches)
            advantage = np.array_split(np.squeeze(advantage_),batches)
            old_log_logits = np.array_split(self.buffer[traj][5][:clip], batches)

            for epoch in range(self.HPs["Epochs"]):
                #Create a feedDict from the buffer
                for i in range(batches):
                    feedDict = {
                        self.s: s[i],
                        self.a: actions[i],
                        self.reward: reward[i],
                        self.s_next: s_next[i],
                        self.td_target: psi_target[i],
                        self.advantage_: advantage[i],
                        self.old_log_logits_: old_log_logits[i]
                    }

                    if not statistics:
                        self.sess.run(self.update_ops, feedDict)   # local grads applied to global net.
                    else:
                        #Perform update operations
                        try:
                            out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)   # local grads applied to global net.
                            update_ops = out.pop(0)
                            grads=out.pop(-1)
                            losses = out

                            for i,loss in enumerate(losses):
                                self.loss_MA[i].append(loss)

                            for i,grads_i in enumerate(grads):
                                total_counter = 0
                                vanish_counter = 0
                                for grad in grads_i:
                                    total_counter += np.prod(grad.shape)
                                    vanish_counter += (np.absolute(grad)<1e-8).sum()
                                self.grad_MA[i].append(vanish_counter/total_counter)
                        except:
                            out = self.sess.run(self.update_ops+self.losses, feedDict)   # local grads applied to global net.
                            out = np.array_split(out,2)
                            losses = out[1]

                            for i,loss in enumerate(losses):
                                self.loss_MA[i].append(loss)

        self.ClearTrajectory()


    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.Gradlabels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
        for i,label in enumerate(self.Losslabels):
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
        return dict


    def ProcessBuffer(self,traj):
        """
        Process the buffer to calculate td_target.

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

        phi_lists = np.split(self.buffer[traj][7],split_loc)
        psi_lists = np.split(self.buffer[traj][8],split_loc)

        td_target=[]; advantage=[]; psi_target=[]
        for rew,value,phi,psi in zip(reward_lists,value_lists,phi_lists,psi_lists):
            td_target_i, advantage_i = gae(rew,value.reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
            psi_target_i, _ = gae(phi,psi, np.zeros_like(phi),self.HPs["Gamma"],self.HPs["lambda"])
            td_target.extend(td_target_i); advantage.extend( advantage_i); psi_target.extend( psi_target_i)
        return td_target, advantage,psi_target

    @property
    def getVars(self):
        return self.Model.getVars(self.scope)

    @property
    def getAParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ 'Actor')

    @property
    def getCParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')
