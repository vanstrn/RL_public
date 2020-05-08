
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
from utils.utils import MovingAverage, GetFunction,CreatePath,interval_flag
from utils.record import Record
import time
import random
from environments import CreateEnvironment
import math

class ApeX(Method):

    def __init__(self,sharedModel,sess,stateShape,actionSize,scope,HPs,sharedBuffer,globalAC=None,nTrajs=1,targetNetwork=None):
        """
        Initializes I/O placeholders used for Tensorflow session runs.
        Initializes and Actor and Critic Network to be used for the purpose of RL.
        """
        self.sess=sess
        self.scope=scope
        self.sharedBuffer=sharedBuffer
        self.actionSize=actionSize

        #Creating the General I/O of the network
        self.Model = sharedModel
        with self.sess.as_default(), self.sess.graph.as_default():
            self.states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='states')
            self.next_states_ = tf.placeholder(shape=[None]+stateShape, dtype=tf.float32, name='next_states')
            self.actions_ = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_hold')
            self.rewards_ = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards_hold')
            self.done_ = tf.placeholder(shape=[None], dtype=tf.float32, name='done_hold')

            if targetNetwork is not None:
                with tf.name_scope("target"):
                    out2 = targetNetwork({"state":self.next_states_})
                    q_next = out2["Q"]
                    self.targetParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target")

            with tf.name_scope(scope):

                input = {"state":self.states_}
                out = self.Model(input)
                self.q = out["Q"]

                # GettingVariables for the specified network.
                with tf.variable_scope(scope):
                    self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

                    # Creating the Global Actor that does all of the learning
                    if globalAC is None:
                        out2 = self.Model({"state":self.next_states_})
                        q_next = out2["Q"]
                        with tf.variable_scope(scope+"_update"):

                            with tf.name_scope('current_Q'):
                                oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                                curr_q = tf.reduce_sum(tf.multiply(self.q, oh_action), axis=-1) # [?, num_agent]

                            with tf.name_scope('target_Q'):
                                max_next_q = tf.reduce_max(q_next, axis=-1)
                                td_target = self.rewards_  + HPs["Gamma"] * max_next_q * (1. - self.done_)

                            with tf.name_scope('td_error'):
                                self.td_error=loss = tf.keras.losses.MSE(td_target, curr_q)
                                softmax_q = tf.nn.softmax(curr_q)
                                self.entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q+ 1e-5))
                                self.loss=total_loss = loss + HPs["EntropyBeta"] * self.entropy
                            optimizer = tf.keras.optimizers.Adam(HPs["LearningRate"])
                            self.gradients = optimizer.get_gradients(total_loss, self.params)
                            self.update_op = optimizer.apply_gradients(zip(self.gradients, self.params))

                            self.grads=[self.gradients]
                            self.losses=[self.loss]
                            self.update_ops=[self.update_op]

                            self.push_ops = [l_p.assign(g_p) for l_p, g_p in zip(self.targetParams, self.params)]

                        self.grad_MA = [MovingAverage(400) for i in range(len(self.grads))]
                        self.loss_MA = [MovingAverage(400) for i in range(len(self.grads))]
                        self.entropy_MA = MovingAverage(400)
                        self.labels = ["Critic",]

                    # Creating the local networks that only pull parameters and run experiments.
                    else:
                        out2 = self.Model({"state":self.next_states_})
                        q_next = out2["Q"]
                        #Creating local Buffer that
                        self.buffer = [Trajectory(depth=5) for _ in range(nTrajs)]
                        with tf.variable_scope(scope+"_priority"):

                            with tf.name_scope('current_Q'):
                                oh_action = tf.one_hot(self.actions_, actionSize, dtype=tf.float32) # [?, num_agent, action_size]
                                curr_q = tf.reduce_sum(tf.multiply(self.q, oh_action), axis=-1) # [?, num_agent]

                            with tf.name_scope('target_Q'):
                                max_next_q = tf.reduce_max(q_next, axis=-1)
                                td_target = self.rewards_  + HPs["Gamma"] * max_next_q * (1. - self.done_)

                            with tf.name_scope('td_error'):
                                self.td_error = tf.keras.losses.MSE(td_target, curr_q)

                        with tf.name_scope('sync'):
                            self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, globalAC.params)]

                            self.pull_ops = [self.pull_params_op]

                self.HPs = HPs

    def GetAction(self, state,episode,step):
        """
        Contains the code to run the network based on an input.
        """
        if len(state.shape) == 3:
            state = state[np.newaxis, :]
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        q = self.sess.run(self.q, {self.states_: state})
        if "Exploration" in self.HPs:
            if self.HPs["Exploration"]=="EGreedy":
                prob = self.HPs["ExploreSS"] + (1-self.HPs["ExploreSS"])*(np.exp(-episode/self.HPs["ExplorationDecay"]))
                if random.uniform(0, 1) < prob:
                    actions = random.randint(0,self.actionSize-1)
                else:
                    actions = np.argmax(q, axis=-1)
            else:
                actions = np.argmax(q, axis=-1)
        else:
            actions = np.argmax(q, axis=-1)
        return actions ,[]  # return a int and extra data that needs to be fed to buffer.

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

            g,s_n=MultiStepDiscountProcessing(traj[2],traj[3],self.HPs["Gamma"],self.HPs["MultiStep"])
            batches = len(traj[0])//self.HPs["MinibatchSize"]+1
            s = np.array_split(traj[0], batches)
            a_his = np.array_split( traj[1], batches)
            r = np.array_split( np.asarray(g), batches)
            s_next = np.array_split( s_n, batches)
            done = np.array_split( traj[4], batches)
            for epoch in range(self.HPs["Epochs"]):
                for i in range(batches):
                #Create a feedDict from the buffer
                    if len(np.squeeze(np.asarray(s[i])).shape)==3 or len(np.squeeze(np.asarray(s[i])).shape)==1:
                        continue
                    feedDict = {
                        self.states_ : np.squeeze(np.asarray(s[i])),
                        self.next_states_ : np.squeeze(np.asarray(s_next[i])),
                        self.actions_ : a_his[i],
                        self.rewards_ : r[i],
                        self.done_ : np.squeeze(np.asarray(done[i],dtype=float))

                    }
                    out = self.sess.run(self.update_ops+self.losses+self.grads, feedDict)   # local grads applied to global net.
                    out = np.array_split(out,3)
                    losses = out[1]
                    grads = out[2]

                    for i,loss in enumerate(losses):
                        if math.isnan(loss):
                            continue
                        self.loss_MA[i].append(loss)

                    for i,grads_i in enumerate(grads):
                        total_counter = 1
                        vanish_counter = 0
                        for grad in grads_i:
                            total_counter += np.prod(grad.shape)
                            vanish_counter += (np.absolute(grad)<1e-8).sum()
                            if math.isnan(vanish_counter/total_counter):
                                continue
                        self.grad_MA[i].append(vanish_counter/total_counter)

        #Updating the Priorities of the samples.
            feedDict = {
                self.states_ : np.squeeze(np.asarray(traj[0])),
                self.next_states_ : np.squeeze(np.asarray(traj[3])),
                self.actions_ : traj[1],
                self.rewards_ : traj[2],
                self.done_ : np.squeeze(np.asarray(traj[4],dtype=float))
            }
            priorities.append(self.sess.run(self.td_error, feedDict))

        self.sharedBuffer.UpdatePriorities(priorities)
        self.sess.run(self.push_ops)


    def PushToBuffer(self):
        #Packaging samples in manner that requires modification on the learner end.

        #Estimating TD Difference to give priority to the data.

        for traj in range(len(self.buffer)):
            s = self.buffer[traj][0]
            a_his = np.asarray(self.buffer[traj][1]).reshape(-1)
            r =  np.asarray(self.buffer[traj][2]).reshape(-1)
            s_next = self.buffer[traj][3]
            done =  self.buffer[traj][4]

                #Create a feedDict from the buffer
            feedDict = {
                self.states_ : np.squeeze(np.asarray(s)),
                self.next_states_ : np.squeeze(np.asarray(s_next)),
                self.actions_ : np.squeeze(np.asarray(a_his)),
                self.rewards_ : np.squeeze(np.asarray(r)),
                self.done_ : np.squeeze(np.asarray(done,dtype=float))
            }
            priority = self.sess.run(self.td_error, feedDict)   # local grads applied to global net.

        self.sharedBuffer.AddTrajectory([s,a_his,r,s_next,done],priority)
        self.sharedBuffer.PrioritizeandPruneSamples(2048)
        self.ClearTrajectory()
        self.sess.run(self.pull_ops)

    def GetStatistics(self):
        dict ={}
        for i,label in enumerate(self.labels):
            dict["Training Results/Vanishing Gradient " + label] = self.grad_MA[i]()
            dict["Training Results/Loss " + label] = self.loss_MA[i]()
        return dict


    def ProcessBuffer(self,HPs,traj):
        """
        Process the buffer to calculate td_target.

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
        return self.Model.getVars(self.scope)

    @property
    def getAParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ 'Actor')

    @property
    def getCParams(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Model.scope + '/Shared') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Model.scope+ '/Critic')

import operator
class ApexBuffer():
    def __init__(self,maxSamples=10000):
        self.maxSamples = maxSamples
        self.buffer=[]
        self.priorities=[]
        self.trajLengths=[]
        self.flag = True
        self.slice=0
        self.sampleSize=0

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
        self.priorities, self.buffer,self.trajLengths = (list(t) for t in zip(*sorted(zip(self.priorities, self.buffer,self.trajLengths),key = operator.itemgetter(0), reverse=True)))

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

            s0 = self.env.reset()

            for j in range(self.settings["MaxEpisodeSteps"]+1):

                a,networkData = self.net.GetAction(state = s0,episode=self.sess.run(self.global_step),step=j)
                s1,r,done,_ = self.env.step(a)
                if render:
                    self.env.render()

                self.net.AddToTrajectory([s0,a,r,s1,done]+networkData)

                s0 = s1
                if done or j == self.settings["MaxEpisodeSteps"]:   # update global and assign to local net
                    self.net.PushToBuffer()
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

            logging = interval_flag(self.sess.run(self.global_step), self.settings["LogFreq"], 'logNet')

            self.net.Update(self.settings["NetworkHPs"],self.sess.run(self.global_step))
            if logging:
                loggingDict = self.net.GetStatistics()
                Record(loggingDict, self.writer, self.sess.run(self.global_step))

def BuildWorkers(sess,networkBuilder,settings,envSettings,netConfigOverride):

    EXP_NAME = settings["RunName"]
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME
    CreatePath(LOG_PATH)
    CreatePath(MODEL_PATH)

    progbar = tf.keras.utils.Progbar(None, unit_name='Training',stateful_metrics=["Reward"])
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    workers = []

    sharedBuffer = ApexBuffer()
    _,dFeatures,nActions,nTrajs = CreateEnvironment(envSettings,multiprocessing=1)

    network = networkBuilder(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    targetNetwork = networkBuilder(settings["NetworkConfig"],nActions,netConfigOverride,scope="Global")
    Method = GetFunction(settings["Method"])
    Updater = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope="Global",HPs=settings["NetworkHPs"],sharedBuffer=sharedBuffer,targetNetwork=targetNetwork)
    Updater.Model.summary()
    saver = tf.train.Saver(max_to_keep=3, var_list=Updater.getVars+[global_step])
    Updater.InitializeVariablesFromFile(saver,MODEL_PATH)
    workers.append(WorkerMaster(Updater,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

    # Create workers
    for i in range(settings["NumberENV"]):
        i_name = 'W_%i' % i   # worker name
        network = networkBuilder(settings["NetworkConfig"],nActions,netConfigOverride,scope=i_name)
        Method = GetFunction(settings["Method"])
        localNetwork = Method(network,sess,stateShape=dFeatures,actionSize=nActions,scope=i_name,HPs=settings["NetworkHPs"],globalAC=Updater,nTrajs=nTrajs,sharedBuffer=sharedBuffer)
        localNetwork.InitializeVariablesFromFile(saver,MODEL_PATH)
        env,_,_,_ = CreateEnvironment(envSettings,multiprocessing=1)
        workers.append(WorkerSlave(localNetwork,env,sess,global_step,global_step_next,settings,progbar,writer,MODEL_PATH,saver))

    return workers
