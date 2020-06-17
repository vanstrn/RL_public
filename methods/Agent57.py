
"""
To Do:
-Add an optional input for the networks so they can be defined in a main run script.
-Test
-Combine Training Operation
"""
from .method import Method
from .buffer import Trajectory
from .AdvantageEstimator import MultiStepDiscountProcessing
import tensorflow as tf
import numpy as np
from utils.utils import MovingAverage
from utils.record import Record
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import random
from environments import CreateEnvironment
from networks.common import NetworkBuilder
import json

class Agent57(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,sharedBuffer,globalNet=None,nTrajs=1,LSTM=False,sharedBandit=None):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        #Placeholders
        self.bandit=sharedBandit
        self.LSTM = LSTM
        self.sess=sess
        self.scope=scope
        self.Model = sharedModel
        self.sharedBuffer=sharedBuffer
        #Common Stuff Between the networks:
        self.HPs = HPs
        #Creating the different values of beta
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        self.betas = []; self.gammas = [];
        for i in range(self.HPs["N"]):
            if i ==0:
                self.gammas.append(self.HPs["Gamma0"])
            elif i < 7:
                self.gammas.append(self.HPs["Gamma1"]+(self.HPs["Gamma0"]-self.HPs["Gamma1"])*sigmoid(10.0*(2.0*float(i)-6.0)/6.0))
            elif i==7:
                self.gammas.append(self.HPs["Gamma1"])
            else:
                self.gammas.append(1.0-np.exp(((self.HPs["N"]-9.0)*np.log(1.0-self.HPs["Gamma1"])+(float(i)-8.0)*np.log(1-self.HPs["Gamma2"]))/(self.HPs["N"]-9.0)))

        for i in range(self.HPs["N"]):
            if i ==0:
                self.betas.append(0.0)
            elif i == self.HPs["N"]-1:
                self.betas.append(self.HPs["betaMax"])
            else:
                self.betas.append(self.HPs["betaMax"]*sigmoid((2.0*float(i)+2.0-self.HPs["N"])/(self.HPs["N"]-2.0)))

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.name_scope(scope):

                #Specifying placeholders for Tensorflow Networks
                if len(stateShape) == 4:
                    self.states_ = tf.placeholder(shape=[None]+stateShape[1:4], dtype=tf.float32, name='states')
                    self.next_states_ = tf.placeholder(shape=[None]+stateShape[1:4], dtype=tf.float32, name='next_states')
                else:
                    self.states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='states')
                    self.next_states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='next_states')
                self.actions_ = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_hold')
                self.done_ = tf.placeholder(shape=[None], dtype=tf.float32, name='done_hold')
                self.rewards_ = tf.placeholder(shape=[None], dtype=tf.float32, name='total_reward')
                self.bandit_one_hot = tf.placeholder(shape=[None,self.HPs["N"]], dtype=tf.int32, name='beta_bandit')
                self.action_past = tf.placeholder(shape=[None], dtype=tf.int32, name='action_past')
                self.reward_i_past = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_i_past')
                self.reward_e_past = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_e_past')
                self.reward_i_current = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_i_current')
                self.reward_e_current = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_e_current')
                g = tf.constant(np.asarray(self.gammas),dtype=tf.float32)
                # Creating the IO for the entire network
                input = {   "state":self.states_,
                            "state_next":self.next_states_,
                            "bandit_one_hot":self.bandit_one_hot,
                            "beta":tf.math.reduce_sum(tf.cast(self.bandit_one_hot,tf.float32)*g,axis=1),
                            "action_past":self.action_past,
                            "reward_i_past":self.reward_i_past,
                            "reward_e_past":self.reward_e_past,
                            }
                out = self.Model(input)
                self.q = out["Q"]
                self.a_pred = out["action_prediction"]
                self.latent = out["latent_space"]
                self.rnd_random = out["RND_random"]
                self.rnd_predictor = out["RND_predictor"]

                input2 = {  "state":self.next_states_,
                            "state_next":self.next_states_, #Used as a placeholder in network
                            "bandit_one_hot":self.bandit_one_hot,
                            "beta":tf.math.reduce_sum(tf.cast(self.bandit_one_hot,tf.float32)*g,axis=1),
                            "action_past":self.actions_,
                            "reward_i_past":self.reward_i_current,
                            "reward_e_past":self.reward_e_current,
                            }
                out2 = self.Model(input2)
                q_next = out["Q"]
                with tf.name_scope('q_learning'):
                    #Current Q
                    oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                    curr_q = tf.reduce_sum(tf.multiply(self.q, oh_action), axis=-1) # [?, num_agent]
                    #Next Q
                    max_next_q = tf.reduce_max(q_next, axis=-1)
                    #TD Error
                    td_target = self.rewards_  + tf.reduce_sum(tf.cast(self.bandit_one_hot,tf.float32)*self.gammas) * max_next_q * (1. - self.done_)
                    self.td_error=loss = tf.keras.losses.MSE(td_target, curr_q)
                    softmax_q = tf.nn.softmax(curr_q)
                    self.entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q))
                    self.loss = loss + HPs["EntropyBeta"] * self.entropy

                self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
                self.int_params = self.Model.GetVariables("Intrinsic")
                self.critic_params = self.Model.GetVariables("Critic")

                if globalNet is None: #Creating the Training instance of the network.
                    with tf.name_scope('embedding_network'):
                        oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                        self.embedding_loss = tf.reduce_mean(tf.keras.losses.MSE(oh_action, self.a_pred))

                    with tf.name_scope('life_long_curiosity'):
                        self.llc_loss = tf.reduce_mean(tf.keras.losses.MSE(self.rnd_random, self.rnd_predictor))

                    loss_critic = tf.reduce_mean(self.loss)
                    optimizer = tf.keras.optimizers.Adam(HPs["LearningRate"])
                    self.gradients = optimizer.get_gradients(loss_critic, self.critic_params)
                    self.update_op = optimizer.apply_gradients(zip(self.gradients, self.critic_params))
                    #
                    loss_intrinsic = tf.reduce_mean( self.llc_loss+self.embedding_loss)
                    optimizer2 = tf.keras.optimizers.Adam(HPs["LearningRateEmbedding"])
                    self.embedding_gradients = optimizer2.get_gradients(loss_intrinsic, self.int_params)
                    self.embedding_update = optimizer2.apply_gradients(zip(self.embedding_gradients, self.int_params))

                    total_counter = 1
                    vanish_counter = 0
                    for gradient in self.gradients:
                        total_counter += np.prod(gradient.shape)
                        stuff = tf.reduce_sum(tf.cast(tf.math.less_equal(tf.math.abs(gradient),tf.constant(1e-8)),tf.int32))
                        vanish_counter += stuff
                    self.vanishing_gradient = vanish_counter/total_counter

                    self.update_ops=[self.update_op]
                    self.logging_ops=[loss,self.embedding_loss,self.llc_loss,self.entropy,self.vanishing_gradient]
                    self.logging_MA = [MovingAverage(400) for i in range(len(self.logging_ops))]
                    self.labels = ["Total Loss","Embedding Loss","Life Long Curiosity Loss","Entropy","Vanishing Gradient"]

                else: #Creating a Actor Instance for the Network.
                    #Creating the Episodic Memory, which compares samples
                    self.episodicMemory = EpisodicMemory()
                    #Creating Local Buffer to store data until it is ready to push to sample buffer
                    self.buffer = [Trajectory(depth=10) for _ in range(nTrajs)]
                    #Creating a pull operation to synch network parameters
                    with tf.name_scope('sync'):
                        self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, globalNet.params)]
                        self.pull_ops = [self.pull_params_op]


    def GetAction(self, state, a_past,r_i_past,r_e_past, episode=None, step=0):
        """
        Contains the code to run the network based on an input.
        """
        #Fixing the state shape if there is somethinf wrong
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]

        #Selecting new beta if the begining of the episode
        #Also bootstrapping rewards/actions for the
        if step == 0:
            self.currBetaSel = self.bandit.GetBanditDecision()
            oh = np.zeros(self.HPs["N"])
            oh[self.currBetaSel] = 1
            self.betaSelect = oh
            self.currBeta = self.betas[self.currBetaSel]
            self.currGamma = self.gammas[self.currBetaSel]

        feedDict={ self.states_: state,
            self.bandit_one_hot:self.betaSelect[np.newaxis, :],
            self.action_past:np.asarray(a_past),
            self.reward_i_past:np.asarray(r_i_past),
            self.reward_e_past:np.asarray(r_e_past)}
        q = self.sess.run(self.q, feedDict)

        actions = np.argmax(q, axis=-1)
        return actions ,[self.currBeta,self.betaSelect]  # return a int and extra data that needs to be fed to buffer.

    def Encode(self,state):
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        return self.sess.run(self.latent,{self.states_:state})

    def RNDPredictionError(self,state):
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        random,predictor = self.sess.run([self.rnd_random,self.rnd_predictor],{self.states_:state})
        return np.linalg.norm(random-predictor)

    def GetIntrinsicReward(self,state_prev,state,episode=None, step=0):
        #Clearing the episodic buffer
        if step==0:
            self.episodicMemory.Clear()
            self.episodicMemory.Add(self.Encode(state_prev))

        #Adding Sample to the buffer
        encodedState = self.Encode(state)
        stateError = self.RNDPredictionError(state)
        self.sharedBuffer.AddError(stateError)

        #####Calculating the episodic reward factor
        #-finding k nearest neighbors in buffer and distance to them
        K = self.episodicMemory.NearestNeighborsDist(encodedState,num=5)
        r_episodic = 1.0/np.sqrt(K+0.001)

        #Calculating alpha
        stateError_Average,stateError_std=self.sharedBuffer.GetMuSigma()
        alpha = 1.0 + (stateError - stateError_Average) / stateError_std

        #Calculating the intrinsic reward
        r_i = r_episodic * min(max(1.0,alpha),5.0)

        #adding the sample to the buffer after nearest neighbors has been calculated.
        self.episodicMemory.Add(encodedState)
        return r_i

    def Update(self,HPs,episode=0,statistics=True):
        """
        The main update function for A3C. The function pushes gradients to the global AC Network.
        The second function is to Pull
        """
        #Process the data from the buffer
        samples,num = self.sharedBuffer.Sample()
        if num < self.HPs["BatchSize"]:
            return

        priorities = []
        for traj in samples:
            if len(traj[0]) <= 5:
                continue

            for epoch in range(self.HPs["Epochs"]):
                #Create a feedDict from the buffer
                feedDict = {
                    self.states_ : np.squeeze(np.asarray(traj[0])),
                    self.actions_ : np.squeeze(np.asarray(traj[1])),
                    self.rewards_ : np.squeeze(np.asarray(traj[2])),
                    self.next_states_ : np.squeeze(np.asarray(traj[3])),
                    self.done_ : np.squeeze(np.asarray(traj[4],dtype=float)),
                    self.action_past:np.squeeze(np.asarray(traj[5])),
                    self.reward_i_past:np.squeeze(np.asarray(traj[6])),
                    self.reward_e_past:np.squeeze(np.asarray(traj[7])),
                    self.bandit_one_hot:np.squeeze(np.asarray(traj[8])),
                }
                out = self.sess.run(self.update_ops+self.logging_ops, feedDict)   # local grads applied to global net.
                logging = out[len(self.update_ops):]

                for i,log in enumerate(logging):
                    self.logging_MA[i].append(log)

    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/" + label] = self.logging_MA[i]()
        return dict

    def PushToBuffer(self,total_reward):
        # Updating the UCB bandit
        self.bandit.InformBandit(self.currBetaSel,total_reward)
        self.sess.run(self.pull_ops)

        #Packaging samples in manner that requires modification on the learner end.
        #Estimating TD Difference to give priority to the data.
        for traj in range(len(self.buffer)):
            g,s_n=MultiStepDiscountProcessing(np.asarray(self.buffer[traj][2]),self.buffer[traj][3],np.sum(self.buffer[traj][9][0]*self.gammas),self.HPs["MultiStep"])
            batches = len(self.buffer[traj][0])//self.HPs["MinibatchSize"]+1
            s = np.array_split(self.buffer[traj][0], batches)
            a_his = np.array_split( self.buffer[traj][1], batches)
            r = np.array_split( np.asarray(g), batches)
            s_next = np.array_split( s_n, batches)
            done = np.array_split( self.buffer[traj][4], batches)

            action_past =  np.array_split(self.buffer[traj][5],batches)
            reward_i_past =  np.array_split(self.buffer[traj][6],batches)
            reward_e_past =  np.array_split(self.buffer[traj][7],batches)
            bandit_one_hot =  np.array_split(self.buffer[traj][9],batches)
            for i in range(batches):
                feedDict = {
                    self.states_ : np.squeeze(np.asarray(s[i])),
                    self.next_states_ : np.squeeze(np.asarray(s_next[i])),
                    self.actions_ : np.squeeze(np.asarray(a_his[i])),
                    self.rewards_ : np.squeeze(np.asarray(r[i])),
                    self.done_ : np.squeeze(np.asarray(done[i],dtype=float)),
                    self.bandit_one_hot : np.squeeze(np.asarray( bandit_one_hot[i])),
                    self.action_past : np.squeeze(np.asarray( action_past[i] )),
                    self.reward_i_past : np.squeeze(np.asarray( reward_i_past[i] )),
                    self.reward_e_past : np.squeeze(np.asarray( reward_e_past[i] )),
                }
                priority = self.sess.run(self.td_error, feedDict)

                self.sharedBuffer.AddTrajectory([s[i],a_his[i],r[i],s_next[i],done[i],action_past[i],reward_i_past[i],reward_e_past[i],bandit_one_hot[i]],priority)
        self.ClearTrajectory()

    def PrioritizeBuffer(self):
        #Updating the network weights before calculating new priorities
        self.sess.run(self.pull_ops)
        #Getting the data that needs to be assigned a new priority.
        trajs = self.sharedBuffer.GetReprioritySamples()
        priority=[]
        for traj in trajs:

            feedDict = {
                self.states_ : np.squeeze(np.asarray(traj[0])),
                self.actions_ : np.squeeze(np.asarray(traj[1])),
                self.rewards_ : np.squeeze(np.asarray(traj[2])),
                self.next_states_ : np.squeeze(np.asarray(traj[3])),
                self.done_ : np.squeeze(np.asarray(traj[4],dtype=float)),
                self.action_past:np.squeeze(np.asarray(traj[5])),
                self.reward_i_past:np.squeeze(np.asarray(traj[6])),
                self.reward_e_past:np.squeeze(np.asarray(traj[7])),
                self.bandit_one_hot:np.squeeze(np.asarray(traj[8])),
            }
            priority.append( self.sess.run(self.td_error, feedDict))
        #Calculating the priority.
        self.sharedBuffer.UpdatePriorities(priority)

        #Pushing the priorities back to the buffer
        self.sharedBuffer.PrioritizeandPruneSamples(2048)


    @property
    def getVars(self):
        return self.Model.getVars(self.scope)
class WorkerPrioritizer(object):
    def __init__(self,localNetwork,sess,global_step,settings):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess=sess
        self.net = localNetwork
        self.global_step = global_step
        self.settings =settings

    def work(self,COORD,render=False):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.settings["MaxEpisodes"]:
            self.net.PrioritizeBuffer()
class WorkerSlave(object):
    def __init__(self,localNetwork,env,sess,global_step,global_step_next,settings,
                    progbar,writer,MODEL_PATH,saver):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess=sess
        self.env=env
        self.net = localNetwork
        self.global_step = global_step
        self.global_step_next = global_step_next
        self.settings =settings
        self.progbar =progbar
        self.writer=writer
        self.MODEL_PATH=MODEL_PATH
        self.saver=saver

    def work(self,COORD,render=False):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.settings["MaxEpisodes"]:

            self.sess.run(self.global_step_next)

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'logEnv')
            saving = interval_flag(self.sess.run(self.global_step), self.settings["SaveFreq"], 'save')

            #Initializing environment and storage variables:
            s0 = self.env.reset()
            a_past = [0]
            r_i_past = [0.0]
            r_e_past = [0.0]
            r_episode = 0.0
            r_int_episode = 0.0

            for j in range(self.settings["MaxEpisodeSteps"]+1):

                a,networkData = self.net.GetAction(state = s0,episode=self.sess.run(self.global_step),step=j,a_past=a_past,r_i_past=r_i_past,r_e_past=r_e_past)
                #networkData is expected to be [betaVal,betaOH]

                s1,r,done,_ = self.env.step(a)
                if render:
                    self.env.render()

                #Calculating Intrinsic Reward of the state:
                r_intrinsic = self.net.GetIntrinsicReward(s0,s1)
                r_total = r + networkData[0]*r_intrinsic

                #Adding to the trajectory
                self.net.AddToTrajectory([s0,a,r_total,s1,done,a_past,r_i_past,r_e_past]+networkData)

                #Updating the storage variables.
                s0 = s1
                a_past = a
                r_i_past = [r_intrinsic]
                r_e_past = [r]
                r_episode+=r_total
                r_int_episode+=networkData[0]*r_intrinsic

                #Pushing entire trajectory to the buffer
                if done or j == self.settings["MaxEpisodeSteps"]:
                    self.net.PushToBuffer(r_episode)
                    break

            self.progbar.update(self.sess.run(self.global_step))
            if logging:
                Record({"Reward/Intrinsic":r_int_episode,"Reward/Total":r_episode}, self.writer, self.sess.run(self.global_step))
                loggingDict = self.env.getLogging()
                Record(loggingDict, self.writer, self.sess.run(self.global_step))
            if saving:
                self.saver.save(self.sess, self.MODEL_PATH+'/ctf_policy.ckpt', global_step=self.sess.run(self.global_step))

class WorkerMaster(object):
    def __init__(self,localNetwork,sess,global_step,global_step_next,settings,
                    progbar,writer,MODEL_PATH,saver):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess=sess
        self.net = localNetwork
        self.global_step = global_step
        self.global_step_next = global_step_next
        self.settings =settings
        self.progbar =progbar
        self.writer=writer
        self.MODEL_PATH=MODEL_PATH
        self.saver=saver

    def work(self,COORD,render=False):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.settings["MaxEpisodes"]:

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'logNet')
            self.net.Update(self.settings["NetworkHPs"],self.sess.run(self.global_step))

            if logging:
                loggingDict = self.net.GetStatistics()
                Record(loggingDict, self.writer, self.sess.run(self.global_step))


class EpisodicMemory():
    def __init__(self):
        self.buffer=[]
    def Add(self,sample):
        self.buffer.append(sample)
    def Clear(self):
        self.buffer=[]
    def NearestNeighborsDist(self,sample,num=5):
        #If there isn't enough samples then just calculate distance based all smaples
        K = 0
        if len(self.buffer) <= num:
            for neighbor in self.buffer:
                dist=np.linalg.norm(neighbor-sample)
                K += 0.001/(dist+0.001)
        else:
            #finding the distance to all points
            allDist=[]
            for neighbor in self.buffer:
                allDist.append(0.001/(np.linalg.norm(neighbor-sample)+0.001))
            #Calculating the distances to the n clostest neighbors
            K = sum(allDist.argsort()[:num])
        return K

class SlidingWindowUCBBandit():
    def __init__(self,N=16,WindowSize=90):
        self.counter = 0
        self.N = N
        self.WindowSize = WindowSize
        self.buffer = Trajectory(depth=2)
        self.beta= 1.0
        self.epsiolon=0.5

        self.thing = 1.0

    def GetBanditDecision(self):
        if self.counter < self.N: #Taking the first N actions
            tmp = self.counter
            self.counter+=1
            return tmp
        else:
            tmp = self.ChooseBandit()
            return tmp
    def ChooseBandit(self):
        count = [0]*self.N
        R_total = [0]*self.N
        mu = [0]*self.N
        length = len(self.buffer[0])
        for i in range(length):
            count[self.buffer[0][i]] += 1
            R_total[self.buffer[0][i]] += self.buffer[1][i]
        for i in range(self.N):
            if count[i] ==0:
                mu[i] = self.beta*np.sqrt(np.log(length)/self.thing)
            else:
                mu[i] = R_total[i]/count[i] + self.beta*np.sqrt(np.log(length)/count[i])

        return np.argmax(np.asarray(mu))


    def InformBandit(self,bandit,reward):
        self.buffer.append([bandit,reward])
        self.buffer.trim(self.WindowSize)

class Agent57Buffer():
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
        self.priorities, self.buffer,self.trajLengths = (list(t) for t in zip(*sorted(zip(self.priorities, self.buffer,self.trajLengths), key=lambda x: x[0],reverse=True)))

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


def Agent57Workers(sess,settings,netConfigOverride):
    #Created Here if there is something to save images
    EXP_NAME = settings["RunName"]
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME
    CreatePath(LOG_PATH)
    CreatePath(MODEL_PATH)

    for (dirpath, dirnames, filenames) in os.walk("configs/environment"):
        for filename in filenames:
            if settings["EnvConfig"] == filename:
                envConfigFile = os.path.join(dirpath,filename)
                break
    with open(envConfigFile) as json_file:
        envSettings = json.load(json_file)

    sharedBuffer = Agent57Buffer()
    sharedBandit = SlidingWindowUCBBandit()
    _,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings,multiprocessing=1)

    progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    workers = []

    network = NetworkBuilder(settings["NetworkConfig"],netConfigOverride,scope="Global",actionSize=nActions)
    Updater = Agent57(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"],sharedBuffer=sharedBuffer)
    Updater.Model.summary()
    saver = tf.train.Saver(max_to_keep=3, var_list=Updater.getVars+[global_step])
    Updater.InitializeVariablesFromFile(saver,MODEL_PATH)
    workers.append(WorkerMaster(Updater,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

    network = NetworkBuilder(settings["NetworkConfig"],netConfigOverride,scope="prioritizer",actionSize=nActions)
    localNetwork = Agent57(network,sess,stateShape=dFeatures,actionSize=nActions,scope="prioritizer",HPs=settings["NetworkHPs"],globalNet=Updater,nTrajs=nTrajs,sharedBuffer=sharedBuffer)
    localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
    workers.append(WorkerPrioritizer(localNetwork,sess,global_step,settings))

    # Create workers
    for i in range(settings["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = NetworkBuilder(settings["NetworkConfig"],netConfigOverride,scope=i_name,actionSize=nActions)
        localNetwork = Agent57(network,sess,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalNet=Updater,nTrajs=nTrajs,sharedBuffer=sharedBuffer,sharedBandit=sharedBandit)
        localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
        env,_,_,_ = CreateEnvironment(envSettings,multiprocessing=1)
        workers.append(WorkerSlave(localNetwork,env,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

    return workers
