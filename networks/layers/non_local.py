import tensorflow as tf

import numpy as np

from .core import layer_normalization

def soft_attention(h_prev, a, num_input, hidden_size):
    def weight_variable(name, shape):
        return tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # h_prev: output from lstm of previous time step (shape: [batch_size, lstm_size])
    # a: Result of CNN [batch_size, conv_size * conv_size, channel_size]

    Wa = weight_variable('Wa', [num_input, 1])
    Wh = weight_variable('Wh', [hidden_size, 1])

    m_list = [tf.tanh(tf.matmul(a[i], Wa) + tf.matmul(h_prev, Wh)) for i in range(len(a))]
    m_concat = tf.concat([m_list[i] for i in range(len(a))], axis = 1)
    alpha = tf.nn.softmax(m_concat)
    z_list = [tf.multiply(a[i], tf.slice(alpha, (0, i), (-1, 1))) for i in range(len(a))]
    z_stack = tf.stack(z_list, axis = 2)
    z = tf.reduce_sum(z_stack, axis = 2)

    return alpha, z

def self_attention(data, hidden_dim, output_dim, residual=True):
    # Motivated from 'Attention is all you need'
    # data shape : [T,C]
    # output_dim : V
    # output shape : [T, C+V]
    def scaled_dot_product(Q, K, scaled_=True, masked_=False):
        # Scaled-dot product
        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        if masked_:
            raise NotImplementedError

        attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
        return attention

    Q = tf.layers.dense(data, hidden_dim, name='query')  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, hidden_dim, name='key')  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, output_dim, name='value')  # [batch_size, sequence_length, output_dim]

    attention = scaled_dot_product(Q, K)  # [batch_size, sequence_length, sequence_length]
    output = tf.matmul(attention, V)  # [batch_size, sequence_length, output_dim]

    if residual:
        #output = data + output
        output = tf.concat([data, output], axis=1)

    return output

def non_local_nn_2d(data, hidden_dim=None, pool=False, use_dense=False, normalize=False, name='non_local', return_layers=False):
    # data shape : [Batch, H, W, Channel]
    # output dim : [Batch, H, W, Channel]
    _layers = {'input': data} # monitoring layer (input, attention, output)
    with tf.variable_scope(name):
        def scaled_dot_product(Q, K, scaled_=True, masked_=False):
            # Scaled-dot product
            attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

            if scaled_:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

            if masked_:
                raise NotImplementedError

            attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
            return attention

        nbatch, h, w, output_dim = data.get_shape().as_list()
        if hidden_dim is None: hidden_dim = output_dim

        if use_dense:
            flattened = tf.reshape(data, [-1, h*w, output_dim])
            Q = tf.contrib.layers.fully_connected(data, hidden_dim)
            K = tf.contrib.layers.fully_connected(data, hidden_dim)
            V = tf.contrib.layers.fully_connected(data, hidden_dim)
        else:
            Q = tf.contrib.layers.convolution(data, hidden_dim, 1)
            Q = tf.reshape(Q, [-1, h*w, hidden_dim])
            if pool:
                K = tf.contrib.layers.convolution(data, hidden_dim, 1)
                K = tf.contrib.layers.max_pool2d(K, 2)
                K = tf.reshape(K, [-1, (h//2)*(w//2), hidden_dim])
                V = tf.contrib.layers.convolution(data, hidden_dim, 1)
                V = tf.contrib.layers.max_pool2d(V, 2)
                V = tf.reshape(V, [-1, (h//2)*(w//2), hidden_dim])
            else:
                K = tf.contrib.layers.convolution(data, hidden_dim, 1)
                K = tf.reshape(K, [-1, h*w, hidden_dim])

                V = tf.contrib.layers.convolution(data, hidden_dim, 1)
                V = tf.reshape(V, [-1, h*w, hidden_dim])

        if normalize:
            Q = layer_normalization(Q)
            K = layer_normalization(K)
            V = layer_normalization(V)

        dot = scaled_dot_product(Q, K)
        output = tf.matmul(dot, V)  # [batch_size, sequence_length, output_dim]
        output = tf.reshape(output, [-1,h,w,hidden_dim])
        if hidden_dim != output_dim:
            output = tf.contrib.layers.convolution(output, output_dim, 1)
        _layers['attention'] = output

        output = output + data  # Residual
        _layers['output'] = output

    if return_layers:
        return output, _layers
    else:
        return output

def non_local_nn_3d(data, hidden_dim, pool=False, name='non_local'):
    # data shape : [Batch, T, H, W, Channel]
    # output dim : [Batch, T, H, W, Channel]
    with tf.variable_scope(name):
        def scaled_dot_product(Q, K, scaled_=True, masked_=False):
            # Scaled-dot product
            attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

            if scaled_:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

            if masked_:
                raise NotImplementedError

            attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
            return attention

        nbatch, t, h, w, output_dim = data.get_shape().as_list()
        Q = tf.contrib.layers.convolution(data, hidden_dim, 1)
        Q = tf.reshape(Q, [-1, t*h*w, hidden_dim])

        if pool:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.nn.max_pool3d(K, [1,2,2,1,1], strides=[1,2,2,1,1], padding='VALID')
            K = tf.reshape(K, [-1, (t//2)*(h//2)*w, hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.nn.max_pool3d(V, [1,2,2,1,1], strides=[1,2,2,1,1], padding='VALID')
            V = tf.reshape(V, [-1, (t//2)*(h//2)*w, hidden_dim])
        else:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.reshape(K, [-1, t*h*w, hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.reshape(V, [-1, t*h*w, hidden_dim])

        attention = scaled_dot_product(Q, K)
        output = tf.matmul(attention, V)  # [batch_size, sequence_length, output_dim]
        output = tf.reshape(output, [-1,t,h,w,hidden_dim])
        output = tf.contrib.layers.convolution(output, output_dim, 1)
        output = output + data  # Residual

        return output

def multiheaded_attention(data, hidden_dim, att_output_dim, output_dim, num_attention_layer=8):
    each_attention = []
    for _ in range(num_attention_layer):
        output = self_attention(data, hidden_dim, att_output_dim, residual=False)
        each_attention.append(output)

    output = tf.concat(each_attention, axis=1)
    output = tf.layers.dense(output, output_dim)
    return output

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
