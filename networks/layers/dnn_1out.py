"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""
import tensorflow as tf
import tensorflow.keras.layers as KL

networks = {
    "Default": {"Layers"        :[20],
                "Activations"   :[tf.nn.relu,tf.nn.softmax],
                "KernalInit"    :[tf.random_normal_initializer(0., .1), tf.random_normal_initializer(0., .1)],
                "BiasInit"      :[tf.constant_initializer(.1), tf.constant_initializer(.1)],
                },
    "Critic": {"Layers"        :[20],
                "Activations"   :[tf.nn.relu,None],
                "KernalInit"    :[tf.random_normal_initializer(0., .1), tf.random_normal_initializer(0., .1)],
                "BiasInit"      :[tf.constant_initializer(.1), tf.constant_initializer(.1)],
                },
    # "Default": {"Layers"        :[20],
    #             "Activations"   :[tf.nn.relu,tf.nn.softmax],
    #             "KernalInit"    :[tf.random_normal_initializer(0., .1), tf.random_normal_initializer(0., .1)],
    #             "BiasInit"      :[tf.constant_initializer(.1), tf.constant_initializer(.1)],
    #             },

    }

class DNN10ut(tf.keras.Model):
    def __init__(self, namespace,actionSize,networkDict=None,networkName=None):
        """

        """
        super(DNN10ut,self).__init__(name=namespace)
        self.scope=namespace
        with tf.variable_scope(self.scope):
            if networkDict is None and networkName is None:
                layerSizes = networks["Default"]["Layers"]
                layerActivations = networks["Default"]["Activations"]
                kernInit = networks["Default"]["KernalInit"]
                biasInit = networks["Default"]["BiasInit"]
            elif networkDict is None and networkName is not None:
                layerSizes = networks[networkName]["Layers"]
                layerActivations = networks[networkName]["Activations"]
                kernInit = networks[networkName]["KernalInit"]
                biasInit = networks[networkName]["BiasInit"]
            else:
                layerSizes = networkDict["Layers"]
                layerActivations = networkDict["Activations"]
                kernInit = networkDict["KernalInit"]
                biasInit = networkDict["BiasInit"]

            self.layerList=[]
            for i in range(len(layerSizes)):
                self.layerList.append(KL.Dense( layerSizes[i],
                                        activation=layerActivations[i],
                                        kernel_initializer=kernInit[i],  # weights
                                        bias_initializer=biasInit[i],  # biases
                                        name='fc'+str(i+1))
                    )
            #Creating Last Layer
            self.layerList.append(KL.Dense( actionSize,
                                    activation=layerActivations[i+1],
                                    kernel_initializer=kernInit[i+1],  # weights
                                    bias_initializer=biasInit[i+1],  # biases
                                    name='fc'+str(i+2))
                )

    def call(self,inputs):
        x = inputs
        for i in range(len(self.layerList)):
            x = self.layerList[i](x)
        return x
    @property
    def getVars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)


class DNN10ut_(tf.keras.Model):
    def __init__(self, namespace,actionSize):
        super(DNN10ut_,self).__init__(name=namespace)

        self.dense1 = KL.Dense( 40,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                bias_initializer=tf.constant_initializer(0.1),  # biases
                                name='fc1')


        self.dense2 = KL.Dense( actionSize,
                                activation=None,
                                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                bias_initializer=tf.constant_initializer(0.1),  # biases
                                name='fc2')


    def call(self,inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

if __name__ == "__main__":
    sess = tf.Session()
    test = DNN10ut(namespace="Test",actionSize=4)
