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

import policy.policy_A3C

from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import store_args

from method.base import initialize_uninitialized_vars as iuv


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
                 model_dir='./model/meta/',
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
            iuv(self.sess)
        self.state, self.action = self.reset_network_weight()
        print('    TF policy loaded. {}'.format(name) )

        # Subpolicy
        subpol1 = policy.policy_A3C.PolicyGen(
                model_dir='./model/golub_attacker_a3c',
                input_name='global/state:0',
                output_name='global/actor/Softmax:0',
                name='attacker'
            )
        subpol2 = policy.policy_A3C.PolicyGen(
                model_dir='./model/golub_scout_a3c',
                input_name='global/state:0',
                output_name='global/actor/Softmax:0',
                name='scout'
            )
        subpol3 = policy.policy_A3C.PolicyGen(
                model_dir='./model/golub_defense_a3c',
                input_name='global/state:0',
                output_name='global/actor/Softmax:0',
                name='defense'
            )
        self.subpol = [subpol1, subpol2, subpol3]

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
        with self.graph.as_default():
            logit = self.sess.run(self.action, feed_dict={self.state: observation})  # Action Probability
        option = [np.random.choice(3, p=logit[x] / sum(logit[x])) for x in range(len(agent_list))]

        action = []
        for opt, agent, state in zip(o1, envs.get_team_blue(), states):
            action = subpol[opt].gen_action([agent], state[np.newaxis,:], centered_obs=True)[0]

        return np.array(action)

    def reset_network_weight(self, input_name=None, output_name=None):
        """
        Reload the weight from the TF meta data
        """
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
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
