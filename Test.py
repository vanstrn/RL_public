"""
Executes Test functions for the repository
"""

import glob
from networks.networkHierarchy import HierarchicalNetwork
from methods.PPO_Hierarchy import PPO_Hierarchy
import tensorflow as tf
import numpy as np
from utils.utils import InitializeVariables, CreatePath, interval_flag, GetFunction


if __name__ == "__main__":
    import numpy as np
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(config=config)
    with tf.device("/gpu:0"):
        test = HierarchicalNetwork(configFile="networks/hierarchyTest.json",actionSize=4)
        # s = tf.placeholder(tf.float32, [None,39,39,6], 'S')
        # state={"state":s}
        # out = test(state)
        HPs = { "eps":0.2,
                "EntropyBeta":0.00,
                "CriticBeta":0.3,
                "LR":0.0001,
                "Gamma":0.99,
                "lambda":0.9,
                "Optimizer":"Adam",
                "Epochs":5,
                "BatchSize":1024,
                "FS":2,
                "MinibatchSize":32}
        test2 = PPO_Hierarchy(test,sess,[10,10,3],4,HPs)
    InitializeVariables(sess)
    s = np.random.rand(10,10,10,3)
    a,stuff=test2.GetAction(s,0)
