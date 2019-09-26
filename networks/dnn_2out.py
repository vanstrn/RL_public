"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""
from netwok import Network
import tensorflow as tf

class DNN20ut(Network):
    def __init__(self, namespace, lr=0.01):
        super(DNN20ut,self).__init__()

        self.dense1 = KL.dense( 20,
                                activation=tf.nn.relu,
                                name='Fully Connected 1')

        self.dense2 = KL.dense( 1,
                                activation=tf.nn.relu,
                                name='Fully Connected 2')

        self.dense3 = KL.dense( 1,
                                activation=tf.nn.relu,
                                name='Fully Connected 2')


    def call(self,inputs):
        x = self.dense1(inputs)

        return self.dense2(x), self.dense3(x)

    def Learn(self, s, r, s_):
        pass

    def ChooseAction(self,a):
        pass

    def SaveStatistics(self,a):
        pass
