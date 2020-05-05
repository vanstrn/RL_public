
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
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction
from utils.record import Record,SaveHyperparams
import random


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
                self.beta = tf.placeholder(shape=[None], dtype=tf.float32, name='beta')
                self.action_past = tf.placeholder(shape=[None], dtype=tf.int32, name='action_past')
                self.reward_i_past = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_i_past')
                self.reward_e_past = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_e_past')
                self.reward_i_current = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_i_current')
                self.reward_e_current = tf.placeholder(shape=[None], dtype=tf.float32, name='reward_e_current')

                # Creating the IO for the entire network
                input = {   "state":self.states_,
                            "state_next":self.next_states_,
                            "bandit_one_hot":self.bandit_one_hot,
                            "beta":self.beta,
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
                            "beta":self.beta,
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
                    td_target = self.rewards_  + HPs["Gamma"] * max_next_q * (1. - self.done_)
                    self.td_error=loss = tf.keras.losses.MSE(td_target, curr_q)
                    softmax_q = tf.nn.softmax(curr_q)
                    self.entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q))
                    self.loss = loss + HPs["EntropyBeta"] * self.entropy

                self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

                if globalNet is None: #Creating the Training instance of the network.
                    with tf.name_scope('embedding_network'):
                        oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                        self.embedding_loss = tf.keras.losses.MSE(oh_action, self.a_pred)

                    with tf.name_scope('life_long_curiosity'):
                        self.llc_loss = tf.keras.losses.MSE(self.rnd_random, self.rnd_predictor)
                    loss = tf.reduce_mean(self.loss + self.llc_loss + self.embedding_loss)

                    optimizer = tf.keras.optimizers.Adam(HPs["LearningRate"])

                    self.gradients = optimizer.get_gradients(loss, self.params)
                    self.update_op = optimizer.apply_gradients(zip(self.gradients, self.params))

                    self.grads=[self.gradients]
                    self.losses=[loss]
                    self.update_ops=[self.update_op]

                    self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
                    self.loss_MA = [MovingAverage(400) for i in range(len(self.losses))]
                    self.entropy_MA = MovingAverage(400)
                    self.labels = ["Total"]
                    self.HPs = HPs

                else: #Creating a Actor Instance for the Network.
                    #Creating the Episodic Memory, which compares samples
                    self.episodicMemory = EpisodicMemory()
                    #Creating Local Buffer to store data until it is ready to push to sample buffer
                    self.buffer = [Trajectory(depth=11) for _ in range(nTrajs)]
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
            self.beta:np.asarray([self.currBeta]),
            self.reward_i_past:np.asarray(r_i_past),
            self.reward_e_past:np.asarray(r_e_past)}
        q = self.sess.run(self.q, feedDict)

        actions = np.argmax(q, axis=-1)
        return actions ,[[self.currBeta],self.currGamma,self.betaSelect]  # return a int and extra data that needs to be fed to buffer.

    def Encode(self,state):
        return self.sess.run(self.latent,{self.states_:state})

    def RNDPredictionError(self,state):
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
            batches = len(traj[0])//self.HPs["MinibatchSize"]+1
            s = np.array_split( traj[0], batches)
            a_his = np.array_split( np.asarray(traj[1]).reshape(-1), batches)
            r = np.array_split( np.asarray(traj[2]).reshape(-1), batches)
            s_next = np.array_split( traj[3], batches)
            done = np.array_split( traj[4], batches)
            bandit_one_hot = np.array_split( traj[8], batches)
            action_past = np.array_split( traj[5], batches)
            reward_i_past = np.array_split( traj[6], batches)
            reward_e_past = np.array_split( traj[7], batches)
            beta = np.array_split( traj[9], batches)

            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):
                #Create a feedDict from the buffer
                    if len(np.squeeze(np.asarray(s[i])).shape)==3:
                        continue
                    feedDict = {
                        self.states_ : np.squeeze(np.asarray(s[i])),
                        self.next_states_ : np.squeeze(np.asarray(s_next[i])),
                        self.actions_ : np.squeeze(np.asarray(a_his[i])),
                        self.rewards_ : np.squeeze(np.asarray(r[i])),
                        self.done_ : np.squeeze(np.asarray(done[i],dtype=float)),
                        self.bandit_one_hot:np.squeeze(np.asarray(bandit_one_hot[i])),
                        self.action_past:np.squeeze(np.asarray(action_past[i])),
                        self.reward_i_past:np.squeeze(np.asarray(reward_i_past[i])),
                        self.reward_e_past:np.squeeze(np.asarray(reward_e_past[i])),
                        self.beta:np.squeeze(np.asarray(beta[i])),
                    }
                    out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)   # local grads applied to global net.
                    out = np.array_split(out,3)
                    losses = out[1]
                    grads = out[2]

                    for i,loss in enumerate(losses):
                        self.loss_MA[i].append(loss)

                    for i,grads_i in enumerate(grads):
                        total_counter = 1
                        vanish_counter = 0
                        for grad in grads_i:
                            total_counter += np.prod(grad.shape)
                            vanish_counter += (np.absolute(grad)<1e-8).sum()
                        self.grad_MA[i].append(vanish_counter/total_counter)

                    ent = self.sess.run(self.entropy, feedDict)   # local grads applied to global net.
                    entropy = np.average(np.asarray(ent))
                    self.entropy_MA.append(entropy)
                feedDict = {
                    self.states_ : np.squeeze(np.asarray(traj[0])),
                    self.next_states_ : np.squeeze(np.asarray(traj[3])),
                    self.actions_ : traj[1],
                    self.rewards_ : traj[2],
                    self.done_ : np.squeeze(np.asarray(traj[4],dtype=float)),
                    self.bandit_one_hot : np.asarray( traj[8]),
                    self.action_past : np.squeeze(np.asarray( traj[5], )),
                    self.reward_i_past : np.squeeze(np.asarray( traj[6], )),
                    self.reward_e_past : np.squeeze(np.asarray( traj[7], )),
                    self.beta : np.squeeze(np.asarray( traj[9], )),
                }
                priorities.append(self.sess.run(self.td_error, feedDict))

        self.sharedBuffer.UpdatePriorities(priorities)

    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
            dict["Training Results/Entropy"] = self.entropy_MA()
        return dict

    def PushToBuffer(self,total_reward):

        # Updating the UCB bandit
        self.bandit.InformBandit(self.currBetaSel,total_reward)
        #Packaging samples in manner that requires modification on the learner end.

        #Estimating TD Difference to give priority to the data.
        for traj in range(len(self.buffer)):
            s = self.buffer[traj][0]
            a_his = np.asarray(self.buffer[traj][1]).reshape(-1)
            r =  np.asarray(self.buffer[traj][2]).reshape(-1)
            s_next = self.buffer[traj][3]
            done =  self.buffer[traj][4]
            action_past =  self.buffer[traj][5]
            reward_i_past =  self.buffer[traj][6]
            reward_e_past =  self.buffer[traj][7]
            beta =  self.buffer[traj][8]
            bandit_one_hot =  self.buffer[traj][10]

                #Create a feedDict from the buffer
            feedDict = {
                self.states_ : np.squeeze(np.asarray(s)),
                self.next_states_ : np.squeeze(np.asarray(s_next)),
                self.actions_ : np.squeeze(np.asarray(a_his)),
                self.rewards_ : np.squeeze(np.asarray(r)),
                self.done_ : np.squeeze(np.asarray(done,dtype=float)),
                self.beta : np.squeeze(np.asarray( beta)),
                self.bandit_one_hot : np.squeeze(np.asarray( bandit_one_hot)),
                self.action_past : np.squeeze(np.asarray( action_past )),
                self.reward_i_past : np.squeeze(np.asarray( reward_i_past )),
                self.reward_e_past : np.squeeze(np.asarray( reward_e_past )),
            }
            priority = self.sess.run(self.td_error, feedDict)

        self.sharedBuffer.AddTrajectory([s,a_his,r,s_next,done,action_past,reward_i_past,reward_e_past,bandit_one_hot,beta],priority)
        self.sharedBuffer.PrioritizeandPruneSamples(2048)
        self.ClearTrajectory()
        self.sess.run(self.pull_ops)


    @property
    def getVars(self):
        return self.Model.getVars(self.scope)

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

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'log')
            saving = interval_flag(self.sess.run(self.global_step), self.settings["SaveFreq"], 'save')

            #Initializing environment and storage variables:
            s0 = self.env.reset()
            a_past = [0]
            r_i_past = [0.0]
            r_e_past = [0.0]
            r_episode = 0.0

            for j in range(self.settings["MaxEpisodeSteps"]+1):

                a,networkData = self.net.GetAction(state = s0,episode=self.sess.run(self.global_step),step=j,a_past=a_past,r_i_past=r_i_past,r_e_past=r_e_past)
                #networkData is expected to be [betaVal,betaOH]

                s1,r,done,_ = self.env.step(a)
                if render:
                    self.env.render()

                #Calculating Intrinsic Reward of the state:
                r_intrinsic = self.net.GetIntrinsicReward(s0,s1)
                r_total = r + networkData[0][0]*r_intrinsic

                #Adding to the trajectory
                self.net.AddToTrajectory([s0,a,r_total,s1,done,a_past,r_i_past,r_e_past]+networkData)

                #Updating the storage variables.
                s0 = s1
                a_past = a
                r_i_past = [r_intrinsic]
                r_e_past = r
                r_episode+=r_total

                #Pushing entire trajectory to the buffer
                if done or j == self.settings["MaxEpisodeSteps"]:
                    self.net.PushToBuffer(r_episode)
                    break

            self.progbar.update(self.sess.run(self.global_step))
            if logging:
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

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'log')

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
    def __init__(self,N=32,WindowSize=90):
        self.counter = 0
        self.N = 32
        self.WindowSize = WindowSize
        self.buffer = Trajectory(depth=2)
        self.beta= 1.0
        self.epsiolon=0.5

        self.thing = 1.0

    def GetBanditDecision(self):
        if self.counter < 32: #Taking the first N actions
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
        self.priorities, self.buffer,self.trajLengths = (list(t) for t in zip(*sorted(zip(self.priorities, self.buffer,self.trajLengths), reverse=True)))

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
