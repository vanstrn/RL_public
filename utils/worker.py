# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import tensorflow as tf
import gym
from utils.record import Record

class Worker(object):
    def __init__(self, name, localNetwork, HPs,global_step):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        for functionString in envSettings["StartingFunctions"]:
            StartingFunction = GetFunction(functionString)
            self.env,_,_,_ = StartingFunction(settings,envSettings)
        self.name = name
        self.AC = localNetwork
        self.HPs=HPs
        self.global_step = global_step

    def work(self,COORD,writer,MODEL_PATH,settings,envSettings):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        while not COORD.should_stop() and int(self.global_step) < self.HPs["MAX_EP"]:

            self.global_step+= 1

            logging = interval_flag(int(self.global_step), settings["EnvHPs"]["LOG_FREQ"], 'log')
            saving = interval_flag(int(self.global_step), settings["EnvHPs"]["SAVE_FREQ"], 'save')

            for functionString in envSettings["BootstrapFunctions"]:
                BootstrapFunctions = GetFunction(functionString)
                s0, loggingDict = BootstrapFunctions(self.env,settings,envSettings)
            for functionString in envSettings["StateProcessingFunctions"]:
                StateProcessing = GetFunction(functionString)
                s0 = StateProcessing(s0,self.env,envSettings)

            for j in range(settings["EnvHPs"]["MAX_EP_STEPS"]+1):
                updating = interval_flag(j, settings["EnvHPs"]['UPDATE_GLOBAL_ITER'], 'update')

                a,networkData = self.net.GetAction(state = s0)

                for functionString in envSettings["ActionProcessingFunctions"]:
                    ActionProcessing = GetFunction(functionString)
                    a = ActionProcessing(a,self.env,envSettings)

                s1,r,done,_ = self.env.step(a)
                for functionString in envSettings["StateProcessingFunctions"]:
                    StateProcessing = GetFunction(functionString)
                    s1 = StateProcessing(s1,self.env,envSettings)

                for functionString in envSettings["RewardProcessingFunctions"]:
                    RewardProcessing = GetFunction(functionString)
                    r,done = RewardProcessing(s1,r,done,self.env,envSettings)

                self.net.AddToBuffer([s,a,r,s_,done]+data)

                s0 = s1

                if updating or done.all():   # update global and assign to local net
                    net.Update(settings["NetworkHPs"])
                if done.all() or j == settings["EnvHPs"]["MAX_EP_STEPS"]:
                    net.Update(settings["NetworkHPs"])
                    net.ClearTrajectory()
                if done.all():
                    break

            #Closing Functions that will be executed after every episode.
            for functionString in envSettings["EpisodeClosingFunctions"]:
                EpisodeClosingFunction = GetFunction(functionString)
                finalDict = EpisodeClosingFunction(loggingDict,self.env,settings,envSettings)

            progbar.update(int(self.global_step))

            if saving:
                net.SaveModel(MODEL_PATH,self.global_step)

            if logging:
                Record(finalDict, writer, int(self.global_step))
