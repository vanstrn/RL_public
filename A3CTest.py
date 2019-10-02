"""
Framework for setting up an experiment.
"""

import numpy as np
import gym
import tensorflow as tf
import threading


from networks.network import Network
from networks.dnn_1out import DNN10ut,DNN10ut_
from networks.dnn_2out import DNN2Out
from methods.A3C import A3C,A3C_s
from utils.utils import InitializeVariables
from utils.record import Record,SaveHyperparams

class Worker(object):
    def __init__(self, name, localNetwork, sess, globalAC):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.env = gym.make('CartPole-v0')
        self.name = name
        self.AC = localNetwork
        self.sess =sess

    def work(self):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing the
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and sess.run(global_step) < HPs["MAX_EP"]:
            s = self.env.reset()
            ep_r = 0
            while True:

                a = self.AC.GetAction(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % HPs['UPDATE_GLOBAL_ITER'] == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + HPs["GAMMA"] * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.UpdateGlobal(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.PullGlobal()

                s = s_
                total_step += 1
                if done:
                    if GLOBAL_RUNNING_R is None:  # record running episode reward
                        GLOBAL_RUNNING_R = ep_r
                    else:
                        GLOBAL_RUNNING_R = 0.99 * GLOBAL_RUNNING_R + 0.01 * ep_r
                        print("episode:", sess.run(global_step), "  reward:", int(GLOBAL_RUNNING_R))

                    if sess.run(global_step) % HPs["SAVE_FREQ"] == 0:
                        saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=sess.run(global_step))
                        pass
                    if sess.run(global_step) % HPs["LOG_FREQ"] == 0:
                        tag = 'Training Results/'
                        Record({
                            tag+'Reward': GLOBAL_RUNNING_R,
                            }, writer, i)

                    sess.run(global_step_next)
                    break

if __name__ == "__main__":
    #Defining parameters and Hyperparameters for the run.
    HPs = {
        "MAX_EP_STEPS" : 1000,
        "MAX_EP" : 10000,
        "SAVE_FREQ" : 100,
        "LOG_FREQ" : 10,
        "Critic LR": 1E-3,
        "Actor LR": 1E-4,
        "N_WORKERS":4,
        "UPDATE_GLOBAL_ITER":10,
        "GAMMA":.9
        }
    EXP_NAME = 'Test13'
    MODEL_PATH = './models/'+EXP_NAME
    LOG_PATH = './logs/'+EXP_NAME

    #Creating the Environment
    sess = tf.Session()
    env = gym.make('CartPole-v0')
    env.seed(1)  # Create a consistent seed so results are reproducible.
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    GLOBAL_RUNNING_R = None

    global_episodes = 0
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step,1)

    #Creating the Networks and Methods of the Run.
    # criticNetwork = DNN10ut("critic",1,networkName="Critic")
    # actorNetwork = DNN10ut("actor",N_A,)
    # net = A2C(actorNetwork,criticNetwork,sess,stateShape=[1,N_F],actionSize=N_A)
    network = DNN2Out("Global",N_A,1,networkName="Test")
    GLOBAL_AC = A3C_s(network,sess,stateShape=N_F,actionSize=N_A,scope="Global",lr_c=HPs["Critic LR"],lr_a=HPs["Actor LR"])
    workers = []
    # Create worker
    for i in range(HPs["N_WORKERS"]):
        i_name = 'W_%i' % i   # worker name
        network = DNN2Out(i_name,N_A,1,networkName="Test")
        localNetwork = A3C_s(network,sess,stateShape=N_F,actionSize=N_A,scope=i_name,globalAC=GLOBAL_AC,lr_c=HPs["Critic LR"],lr_a=HPs["Actor LR"])
        workers.append(Worker(i_name,localNetwork, sess, GLOBAL_AC))


    #Creating Auxilary Functions for logging and saving.
    writer = tf.summary.FileWriter(LOG_PATH,graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.getVars+[global_step])
    SaveHyperparams(writer,HPs)
    GLOBAL_AC.InitializeVariablesFromFile(saver,MODEL_PATH)
    InitializeVariables(sess) #Included to catch if there are any uninitalized variables.

    COORD = tf.train.Coordinator()
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
