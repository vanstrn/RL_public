"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""
import tensorflow as tf
import tensorflow.keras.layers as KL

networks = {
    "Default": {"Shared":{"Layers"      :[20],
                    "Activations"       :[tf.nn.relu],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1)],
                    },
                "Critic": {"Layers"     :[20],
                    "Activations"       :[tf.nn.relu,tf.nn.softmax],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1), tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1), tf.constant_initializer(.1)],
                    },
                "Actor": {"Layers"      :[20],
                    "Activations"       :[tf.nn.relu,tf.nn.softmax],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1), tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1), tf.constant_initializer(.1)],
                    },
        },
        "Test": {"Shared":{"Layers"     :[20],
                    "Activations"       :[tf.nn.relu],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1)],
                    },
                "Critic": {"Layers"     :[],
                    "Activations"       :[None],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1)],
                    },
                "Actor": {"Layers"      :[],
                    "Activations"       :[tf.nn.softmax],
                    "KernalInit"        :[tf.random_normal_initializer(0., .1)],
                    "BiasInit"          :[tf.constant_initializer(.1)],
                    },
        },
    }


class DNN2Out(tf.keras.Model):
    def __init__(self, namespace,actionSize1,actionSize2,networkDict=None,networkName=None):
        """

        """
        super(DNN2Out,self).__init__(name=namespace)
        if networkDict is None and networkName is None:
            layerSizes1 = networks["Default"]["Shared"]["Layers"];layerSizes2 = networks["Default"]["Actor"]["Layers"];layerSizes3 = networks["Default"]["Critic"]["Layers"];
            layerActivations1 = networks["Default"]["Shared"]["Activations"];layerActivations2 = networks["Default"]["Actor"]["Activations"];layerActivations3 = networks["Default"]["Critic"]["Activations"];
            kernInit1 = networks["Default"]["Shared"]["KernalInit"];kernInit2 = networks["Default"]["Actor"]["KernalInit"];kernInit3 = networks["Default"]["Critic"]["KernalInit"];
            biasInit1 = networks["Default"]["Shared"]["BiasInit"];biasInit2 = networks["Default"]["Actor"]["BiasInit"];biasInit3 = networks["Default"]["Critic"]["BiasInit"];
        elif networkDict is None and networkName is not None:
            layerSizes1 = networks[networkName]["Shared"]["Layers"];layerSizes2 = networks[networkName]["Actor"]["Layers"];layerSizes3 = networks[networkName]["Critic"]["Layers"];
            layerActivations1 = networks[networkName]["Shared"]["Activations"];layerActivations2 = networks[networkName]["Actor"]["Activations"];layerActivations3 = networks[networkName]["Critic"]["Activations"];
            kernInit1 = networks[networkName]["Shared"]["KernalInit"];kernInit2 = networks[networkName]["Actor"]["KernalInit"];kernInit3 = networks[networkName]["Critic"]["KernalInit"];
            biasInit1 = networks[networkName]["Shared"]["BiasInit"];biasInit2 = networks[networkName]["Actor"]["BiasInit"];biasInit3 = networks[networkName]["Critic"]["BiasInit"];
        else:
            layerSizes1 = networkName["Shared"]["Layers"];layerSizes2 = networkName["Actor"]["Layers"];layerSizes3 = networkName["Critic"]["Layers"];
            layerActivations1 = networkName["Shared"]["Activations"];layerActivations2 = networkName["Actor"]["Activations"];layerActivations3 = networkName["Critic"]["Activations"];
            kernInit1 = networkName["Shared"]["KernalInit"];kernInit2 = networkName["Actor"]["KernalInit"];kernInit3 = networkName["Critic"]["KernalInit"];
            biasInit1 = networkName["Shared"]["BiasInit"];biasInit2 = networkName["Actor"]["BiasInit"];biasInit3 = networkName["Critic"]["BiasInit"];

        self.scope=namespace
        with tf.variable_scope(self.scope):
            self.layerList1=[]
            self.layerList2=[]
            self.layerList3=[]
            with tf.name_scope('Shared'):
                for i in range(len(layerSizes1)):
                    self.layerList1.append(KL.Dense( layerSizes1[i],
                                            activation=layerActivations1[i],
                                            kernel_initializer=kernInit1[i],  # weights
                                            bias_initializer=biasInit1[i],  # biases
                                            name='shared_fc'+str(i+1))
                        )

            with tf.name_scope('Actor'):
                i=0
                for i in range(len(layerSizes2)):
                    self.layerList2.append(KL.Dense( layerSizes2[i],
                                            activation=layerActivations2[i],
                                            kernel_initializer=kernInit2[i],  # weights
                                            bias_initializer=biasInit2[i],  # biases
                                            name='actor_fc'+str(i+1))
                        )

                if i != 0:
                    i += 1
                self.layerList2.append(KL.Dense( actionSize1,
                        activation=layerActivations2[i],
                        kernel_initializer=kernInit2[i],  # weights
                        bias_initializer=biasInit2[i],  # biases
                        name='actor_fc'+str(i+1))
                        )

            with tf.name_scope('Actor'):
                i=0
                for i in range(len(layerSizes2)):
                    self.layerList3.append(KL.Dense( layerSizes3[i],
                                            activation=layerActivations3[i],
                                            kernel_initializer=kernInit3[i],  # weights
                                            bias_initializer=biasInit3[i],  # biases
                                            name='critic_fc'+str(i+1))
                        )

                #Creating Last Critic Layer
                if i != 0:
                    i += 1
                self.layerList3.append(KL.Dense( actionSize2,
                                        activation=layerActivations3[i],
                                        kernel_initializer=kernInit3[i],  # weights
                                        bias_initializer=biasInit3[i],  # biases
                                        name='critic_fc'+str(i+1))
                )

    def call(self,inputs):
        x = inputs
        for i in range(len(self.layerList1)):
            x = self.layerList1[i](x)
        y,z =x,x
        for i in range(len(self.layerList2)):
            y = self.layerList2[i](y)
        for i in range(len(self.layerList3)):
            z = self.layerList3[i](z)
        return y, z
    @property
    def getVars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)


class DNN20ut_(tf.keras.Model):
    def __init__(self, namespace,actionSize1,actionSize2):
        super(DNN20ut_,self).__init__(name=namespace)

        self.dense1 = KL.Dense( 20,
                                activation=tf.nn.relu,
                                name='Fully Connected 1')

        self.dense2 = KL.Dense( actionSize1,
                                activation=tf.nn.relu,
                                name='Fully Connected 2')

        self.dense3 = KL.Dense( actionSize2,
                                activation=tf.nn.relu,
                                name='Fully Connected 2')


    def call(self,inputs):
        x = self.dense1(inputs)

        return self.dense2(x), self.dense3(x)


if __name__ == "__main__":
    sess = tf.Session()
    test = DNN20ut(namespace="Test",actionSize1=4,actionSize2=1)
