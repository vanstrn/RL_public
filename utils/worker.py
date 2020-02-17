# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import tensorflow as tf
import gym, gym_minigrid
from utils.record import Record
from utils.utils import interval_flag,GetFunction

class Worker(object):
    def __init__(self, name, localNetwork,sess, HPs,global_step,global_step_next,settings,envSettings,progbar):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.sess =sess
        for functionString in envSettings["StartingFunctions"]:
            StartingFunction = GetFunction(functionString)
            self.env,_,_,_ = StartingFunction(settings,envSettings,self.sess)
        self.name = name
        self.net = localNetwork
        self.HPs=HPs
        self.global_step = global_step
        self.global_step_next = global_step_next
        self.settings =settings
        self.envSettings =envSettings
        self.progbar =progbar

    def work(self,COORD,writer,MODEL_PATH,settings,envSettings,saver,GLOBAL_RUNNING_R,GLOBAL_EP_LEN):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.HPs["MAX_EP"]:

            self.sess.run(self.global_step_next)

            logging = interval_flag(self.sess.run(self.global_step), settings["EnvHPs"]["LOG_FREQ"], 'log')
            saving = interval_flag(self.sess.run(self.global_step), settings["EnvHPs"]["SAVE_FREQ"], 'save')

            for functionString in envSettings["BootstrapFunctions"]:
                BootstrapFunctions = GetFunction(functionString)
                s0, loggingDict = BootstrapFunctions(self.env,settings,envSettings,self.sess)

            for functionString in envSettings["StateProcessingFunctions"]:
                StateProcessing = GetFunction(functionString)
                s0 = StateProcessing(s0,self.env,envSettings,self.sess)

            for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):
                updating = interval_flag(j, settings["EnvHPs"]['UPDATE_GLOBAL_ITER'], 'update')

                a,networkData = self.net.GetAction(state = s0)

                for functionString in envSettings["ActionProcessingFunctions"]:
                    ActionProcessing = GetFunction(functionString)
                    a = ActionProcessing(a,self.env,envSettings,self.sess)

                s1,r,done,_ = self.env.step(a)
                for functionString in envSettings["StateProcessingFunctions"]:
                    StateProcessing = GetFunction(functionString)
                    s1 = StateProcessing(s1,self.env,envSettings,self.sess)

                for functionString in envSettings["RewardProcessingFunctions"]:
                    RewardProcessing = GetFunction(functionString)
                    r,done = RewardProcessing(s1,r,done,self.env,envSettings,self.sess)

                self.net.AddToTrajectory([s0,a,r,s1,done]+networkData)

                for functionString in envSettings["LoggingFunctions"]:
                    LoggingFunctions = GetFunction(functionString)
                    loggingDict = LoggingFunctions(loggingDict,s1,r,done,self.env,envSettings,self.sess)

                s0 = s1

                if updating or done.all() or j == settings["EnvHPs"]["MAX_EP_STEPS"]:   # update global and assign to local net
                    self.net.Update(settings["NetworkHPs"],self.sess.run(self.global_step))
                if done.all() or j == settings["EnvHPs"]["MAX_EP_STEPS"]:
                    self.net.ClearTrajectory()
                    break

            #Closing Functions that will be executed after every episode.
            for functionString in envSettings["EpisodeClosingFunctions"]:
                EpisodeClosingFunction = GetFunction(functionString)
                finalDict = EpisodeClosingFunction(loggingDict,self.env,settings,envSettings,self.sess,self.progbar,GLOBAL_RUNNING_R,GLOBAL_EP_LEN)

            self.progbar.update(self.sess.run(self.global_step))

            if saving:
                saver.save(self.sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=self.sess.run(self.global_step))

            if logging:
                dict = self.net.GetStatistics()
                finalDict.update(dict)
                Record(finalDict, writer, self.sess.run(self.global_step))
