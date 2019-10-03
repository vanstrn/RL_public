# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import tensorflow as tf
import gym
from utils.record import Record

class Worker(object):
    def __init__(self, name, localNetwork, sess, HPs,global_step,global_step_next):
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
        self.HPs=HPs
        self.global_step = global_step
        self.global_step_next = global_step_next

    def work(self,COORD,GLOBAL_RUNNING_R,saver,writer,MODEL_PATH):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing access to the global variables.
        total_step = 1
        while not COORD.should_stop() and self.sess.run(self.global_step) < self.HPs["MAX_EP"]:
            s = self.env.reset()
            ep_r = 0
            while True:

                a,data = self.AC.GetAction(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r

                self.AC.AddToBuffer([s,a,r,s_,done]+data)

                if total_step % self.HPs['UPDATE_GLOBAL_ITER'] == 0 or done:   # update global and assign to local net

                    self.AC.Update(self.HPs)

                s = s_
                total_step += 1
                if done:
                    GLOBAL_RUNNING_R.append(ep_r)
                    print("episode:", self.sess.run(self.global_step), "  reward:", int(GLOBAL_RUNNING_R()))

                    if self.sess.run(self.global_step) % self.HPs["SAVE_FREQ"] == 0:
                        saver.save(self.sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=self.sess.run(self.global_step))
                        pass

                    if self.sess.run(self.global_step) % self.HPs["LOG_FREQ"] == 0:
                        tag = 'Training Results/'
                        Record({
                            tag+'Reward': GLOBAL_RUNNING_R(),
                            }, writer, self.sess.run(self.global_step))

                    self.sess.run(self.global_step_next)
                    break
