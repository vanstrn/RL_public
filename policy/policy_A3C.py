""" Reinforce Learning Policy Generator

This module calls any pre-trained tensorflow model and generates the policy.
It includes method to switch between the network and weights.

For use: generate policy to simulate along with CTF environment.

for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

Last Modified:
    Seung Hyun Kim
    created :Wed Oct 24 12:21:34 CDT 2018
"""

import numpy as np
import tensorflow as tf

from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import store_args



class PolicyGen:
    """Policy generator class for CtF env.

    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action  : Required method to generate a list of actions.
        load_model  : Load pre-defined model (*.meta file). Only TensorFlow model supported
        load_weight : Load/reload weight to the model.
    """

    @store_args
    def __init__(self,
                 free_map=None,
                 agent_list=None,
                 model_dir='./model/A3C_pretrained/',
                 input_name='global/state:0',
                 output_name='global/actor/Softmax:0',
                 import_scope=None,
                 vision_radius=19,
                 trainable=False,
                 name='policy',
                 *args,
                 **kwargs
             ):
        """Constuctor for policy class.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """

        # reset_network
        if not trainable:
            config = tf.ConfigProto(device_count = {'GPU': 0})  # Only use CPU

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if not ckpt or not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise NameError('Error : Graph is not loaded')
        # Reset the weight to the newest saved weight.
        print('Policy using TF pretrained model called:')
        print('    model path : {}'.format(ckpt.model_checkpoint_path))
        print('    input_name : {}'.format(input_name))
        print('    output_name : {}'.format(output_name))
        self.graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta', clear_devices=True)
        self.state, self.action = self.reset_network_weight()
        print('    TF policy loaded. {}'.format(name) )

    def gen_action(self, agent_list, observation, free_map=None, centered_obs=False):
        """Action generation method.

        This is a required method that generates list of actions corresponding
        to the list of units.

        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).

        Returns:
            action_out (list): list of integers as actions selected for team.

        Note:
            The graph is not updated in this session.
            It only returns action for given input.
        """

        if not centered_obs:
            observation = one_hot_encoder(state=observation,
                    agents=agent_list, vision_radius=self.vision_radius)
        logit = self.sess.run(self.action, feed_dict={self.state: observation})  # Action Probability
        action_out = [np.random.choice(5, p=logit[x] / sum(logit[x])) for x in range(len(agent_list))]

        return action_out

    def reset_network_weight(self):
        """
        Reload the weight from the TF meta data
        """
        input_name = self.input_name
        output_name = self.output_name
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            state = self.graph.get_tensor_by_name(input_name)
            try:
                action = self.graph.get_operation_by_name(output_name)
            except ValueError:
                action = self.graph.get_tensor_by_name(output_name)
        return state, action
