"""
Executes Test functions for the repository
"""

import glob
from networks.network import Network
import tensorflow as tf
import numpy as np

def NetworkCreationTest(path="configs/network"):
    """
    """
    filenames = glob.glob(path+"/*.json")
    for filename in filenames:
        sess = tf.Session()
        with sess.as_default(), sess.graph.as_default():
            print("filename",filename)
            test = Network(configFile=filename,actionSize=4)
            if test.built:
                input={}
                for inputName,inputDimension in test.testInputs.items():
                    if inputName == "hiddenState":
                        input[inputName] = [tf.placeholder(tf.float32, [None, inputDimension], 'HiddenStateSize'),tf.placeholder(tf.float32, [None, inputDimension], 'HiddenCellSize')]
                    else:
                        input[inputName] = tf.placeholder(tf.float32, inputDimension, 'HiddenCellSize')
                print(test(input))
        tf.keras.backend.clear_session()




if __name__ == "__main__":
    NetworkCreationTest()
