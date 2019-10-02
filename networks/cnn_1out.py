"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.
Should contain reference image of the network in Github Repo.
"""
import tensorflow as tf
import tensorflow.keras.layers as KL

class CNN10ut(tf.keras.Model):
    def __init__(self, namespace):
        super(CNN10ut,self).__init__(name=namespace)

        self.sep_conv2d = keras_layers.SeparableConv2D(
                filters=32,
                kernel_size=4,
                strides=2,
                padding='valid',
                depth_multiplier=4,
                activation='relu',
            )
        self.non_local = Non_local_nn(16)
        self.conv1 = keras_layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')
        self.conv2 = keras_layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu')
        self.flat  = keras_layers.Flatten()
        self.dense1 = keras_layers.Dense(units=128)


    def call(self,inputs):
        net = inputs
        _layers = {'input': net}

        # Block 1 : Separable CNN
        net = self.sep_conv2d(net)
        _layers['sepCNN1'] = net

        # Block 2 : Attention (with residual connection)
        net = self.non_local(net)
        _layers['attention'] = self.non_local._attention_map
        _layers['NLNN'] = net

        # Block 3 : Convolution
        net = self.conv1(net)
        _layers['CNN1'] = net
        net = self.conv2(net)
        _layers['CNN2'] = net

        # Block 4 : Feature Vector
        net = self.flat(net)
        _layers['flat'] = net
        net = self.dense1(net)
        _layers['dense1'] = net

        self._layers_snapshot = _layers

        if self.trainable:
            return net
        else:
            return tf.stop_gradient(net)

if __name__ == "__main__":
    sess = tf.Session()
    test = DNN20ut(namespace="Test",actionSize1=4)
