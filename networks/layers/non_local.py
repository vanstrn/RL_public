import tensorflow as tf

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as KL
import numpy as np
from tensorflow.keras.regularizers import l2

class Non_local_nn(tf.keras.layers.Layer):
    def __init__(self, channels, pool=False, residual=True, train_gamma=False, name='non_local'):
        super(Non_local_nn, self).__init__(name=name)

        self.channels = channels
        self.residual = residual
        self.pool = pool
        self.train_gamma = train_gamma

    def build(self, input_shape):
        # Define layers
        with tf.variable_scope(self.name):
            self.f_conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, name='f_conv')
            self.f_conv_pool = tf.keras.layers.MaxPool2D(name='f_conv_pool')
            self.g_conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, name='g_conv')
            self.h_conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, name='h_conv')
            self.h_conv_pool = tf.keras.layers.MaxPool2D(name='h_conv_pool')
            self.att_conv = tf.keras.layers.Conv2D(filters=input_shape[-1].value, kernel_size=1, strides=1, name='att_conv')
            self.gamma = tf.Variable(1.0, self.train_gamma, name='att_gamma', dtype=tf.float32)

    def call(self, input_layer, normalize=True):
        batch_size, height, width, num_channels = input_layer.get_shape().as_list()

        f = self.f_conv(input_layer)
        g = self.g_conv(input_layer)
        h = self.h_conv(input_layer)
        if self.pool:
            f = self.f_conv_pool(f)
            h = self.h_conv_pool(h)

        _, x, y, channel = f.get_shape().as_list()
        f = tf.reshape(f, [-1, x*y, channel])
        _, x, y, channel = g.get_shape().as_list()
        g = tf.reshape(g, [-1, x*y, channel])
        _, x, y, channel = h.get_shape().as_list()
        h = tf.reshape(h, [-1, x*y, channel])
        dot = tf.matmul(g, f, transpose_b=True)  # [bs, N, N]
        if normalize:
            d_k = tf.cast(tf.shape(f)[-1], dtype=tf.float32)
            dot = tf.divide(dot, tf.sqrt(d_k))

        beta = tf.nn.softmax(dot)  # attention map

        o = tf.matmul(beta, h)  # [bs, N, C]
        o = tf.reshape(o, shape=[-1, height, width, self.channels])  # [bs, h, w, C]
        if self.channels != num_channels:
            o = self.att_conv(o)
        self._attention_map = o

        if self.residual:
            return self.gamma * o + input_layer
        else:
            return o
