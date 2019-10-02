
"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
Capture the Flag implementation.
Based on: https://morvanzhou.github.io/tutorials/

"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

from Utils import record,initialize_uninitialized_vars

GAME = 'cap-v0'
LOG_DIR = './log'
N_WORKERS = 1#multiprocessing.cpu_count()
MAX_GLOBAL_EP = 10000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 150
GAMMA = 0.99
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GLOBAL_RUNNING_R = None
GLOBAL_EP = 0

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, sess, scope, globalAC=None):
        """
        Creates an actor critic network that is compatible with A3C methods.

        Inputs:
        sess            - Tenorflow session that the network is run on.
        scope           - Scope of the actor-critic. Controls whether the network is
                            the global network that is updated or the worker networks
                            which serve to collect data.
        globalAC (Opt)  - Gives the name of the global Actor-critic which updates
                            are performed on.
        """
        self.sess=sess
        self.scope=scope

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(shape=in_size,dtype=tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        """Creating the network structure. This can be offloaded to external function or class.

        Inputs:
        Scope   - Used to distinguish different networks.
        """

        w_init = tf.random_normal_initializer(0., .1)

        # Creating a shared network portion. In This is used to extract features that can be used
        # by both the Actor and Critic. With each updating the feature extractor it allows for faster convergence
        # than separated networks.
        with tf.variable_scope('shared'):
            layer = layers.conv2d(self.s, 32, [3, 3],
                                  activation_fn=tf.nn.relu,
                                  weights_initializer=layers.xavier_initializer_conv2d(),
                                  biases_initializer=tf.zeros_initializer(),
                                  padding='SAME')
            layer = layers.max_pool2d(layer, [2, 2])
            layer = layers.conv2d(layer, 64, [2, 2],
                                  activation_fn=tf.nn.relu,
                                  weights_initializer=layers.xavier_initializer_conv2d(),
                                  biases_initializer=tf.zeros_initializer(),
                                  padding='SAME')
            layer = layers.flatten(layer)

        #Creating the subsection of the network which is used as the actor.
        with tf.variable_scope('actor'):
            actor = layers.fully_connected(layer, 64)
            actor = layers.fully_connected(self.actor, self.action_size,
                                           activation_fn=tf.nn.softmax)

        #Creating the subsection of the network which is used as the critic.
        with tf.variable_scope('critic'):
            critic = layers.fully_connected(layer, 1,
                                            activation_fn=None)
            critic = tf.reshape(self.critic, (-1,))

        #Collecting the parameters that are in each network. These are used in the training process to specify which
        common_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/shared')
        a_params = common_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = common_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        #Returning things that are used in network. (Outputs and Trainable Data)
        return actor, critic, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    @property
    def get_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def initiate(self, saver, model_path):
        # Restore if savepoint exist. Initialize everything else
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

class Worker(object):
    def __init__(self, name, sess, globalAC):
        """Creates a worker that is used to gather smaples to update the main network.

        Inputs:
        name        - Unique name for the worker actor-critic environmnet.
        sess        - Session Name
        globalAC    - Name of the Global actor-critic which the updates are based around.
        """
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(sess, name, globalAC)
        self.sess =sess

    def work(self):
        """Main function of the Workers. This runs the environment and the experience
        is used to update the main Actor Critic Network.
        """
        #Allowing the
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:

                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if GLOBAL_RUNNING_R is None:  # record running episode reward
                        GLOBAL_RUNNING_R = ep_r
                    else:
                        GLOBAL_RUNNING_R = 0.99 * GLOBAL_RUNNING_R + 0.01 * ep_r

                    if GLOBAL_EP % 1000 == 0:
                        saver.save(self.sess, SAVE_PATH+'/ctf_policy.ckpt', global_step=GLOBAL_EP)
                    if GLOBAL_EP % 100 == 0:
                        #Adding Summary to Tensorboard
                        tag = 'baseline_training/'
                        record({
                            tag+'moving_reward': GLOBAL_RUNNING_R,
                            tag+'ep_reward': ep_r,
                        }, writer, GLOBAL_EP)

                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":

    TRAIN_NAME = "CP4"
    LOG_PATH = './logs/'+TRAIN_NAME
    SAVE_PATH = './model/'+TRAIN_NAME
    LOAD_PATH = './model/'+TRAIN_NAME

    ## Launch TF session and create Graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(config=config)



    with tf.device("/gpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(sess,GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, sess, GLOBAL_AC))

    #Creating the tensorboard logging and model saving.
    writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    saver = tf.train.Saver(max_to_keep=3, var_list=GLOBAL_AC.get_vars)

    ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Load Model : ", ckpt.model_checkpoint_path)
        initialize_uninitialized_vars(sess)
    else:
        sess.run(tf.global_variables_initializer())
        print("Initialized Variables")

    COORD = tf.train.Coordinator()
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
