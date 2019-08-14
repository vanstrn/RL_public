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
from tensorflow.python.tools import inspect_checkpoint as chkp
from utility.dataModule import one_hot_encoder

class PolicyGen:
    import tensorflow as tf
    """Policy generator class for CtF env.
    
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action  : Required method to generate a list of actions.
        load_model  : Load pre-defined model (*.meta file). Only TensorFlow model supported
        load_weight : Load/reload weight to the model. 
    """
    
    def __init__(self, free_map, agent_list, model_dir='./model/VANILLA', color='blue', input_name='state:0', output_name='action:0', import_scope=None):
        """Constuctor for policy class.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """

        # Switches 
        self.deterministic = False
        self.full_observation = True
        self.is_blue = color == 'blue'
        
        
        self.model_dir = model_dir # Default
        self.input_name = input_name
        self.output_name = output_name
        self.reset_network(self.input_name, self.output_name, import_scope)

    def gen_action(self, agent_list, observation, free_map=None):
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

        obs = one_hot_encoder(observation, agent_list, self.input_shape, reverse=not self.is_blue)
        action_prob = [self.sess.run(self.action, feed_dict={self.state : obs[i:i+1,]})[0] for i in range(len(agent_list))]
        #action_prob = self.sess.run(self.action, feed_dict={self.state:obs}) # Action Probability

        # If the policy is deterministic policy, return the argmax
        # The parameter can be changed with set_deterministic(bool)
        if self.deterministic:
            action_out = np.argmax(action_prob, axis=1).tolist()
        else: 
            action_out = [np.random.choice(5, p=action_prob[x]/sum(action_prob[x])) for x in range(len(agent_list))]

        return action_out
    
    def reset_network_weight(self, input_name='state:0', output_name='action:0'):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Weight is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            print('Error : Graph is not loaded')
    
    def reset_network(self, input_name = 'state:0', output_name = 'action:0', im_scope=None):
        # Reset the weight to the newest saved weight.
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('path exist')
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta', import_scope=im_scope, clear_devices=True);
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                #:print([n.name for n in self.graph.as_graph_def().node])
            
                self.state = self.graph.get_tensor_by_name(input_name)
                try:
                    self.action = self.graph.get_operation_by_name(output_name)
                except ValueError:
                    self.action = self.graph.get_tensor_by_name(output_name)
                    #print([n.name for n in self.graph.as_graph_def().node])
            
            self.input_shape = 9
            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            raise NameError
            print('Error : Graph is not loaded')

    def set_directory(self, model_dir):
        self.model_dir = model_dir

    def set_deterministic(self, b):
        self.deterministic = b
