# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import tensorflow as tf
import gym, gym_minigrid
from utils.record import Record
from utils.utils import interval_flag,GetFunction

class Worker(object):
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

            s0 = self.env.reset()

            for j in range(self.settings["MaxEpisodeSteps"]+1):

                a,networkData = self.net.GetAction(state = s0)

                s1,r,done,_ = self.env.step(a)
                if render:
                    self.env.render()

                self.net.AddToTrajectory([s0,a,r,s1,done]+networkData)

                s0 = s1

                if done or j == self.settings["MaxEpisodeSteps"]:   # update global and assign to local net
                    self.net.Update(self.settings["NetworkHPs"],self.sess.run(self.global_step))
                    break

            self.progbar.update(self.sess.run(self.global_step))

            if saving:
                self.saver.save(self.sess, self.MODEL_PATH+'/ctf_policy.ckpt', global_step=self.sess.run(self.global_step))

            if logging:
                loggingDict = self.env.getLogging()
                dict = self.net.GetStatistics()
                loggingDict.update(dict)
                Record(loggingDict, self.writer, self.sess.run(self.global_step))
