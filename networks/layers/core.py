import tensorflow as tf

def layer_normalization(x):
    feature_size = x.get_shape()[-1:]

    mean, variance = tf.nn.moments(x, [2], keep_dims=True)
    beta = tf.Variable(tf.zeros(feature_size), trainable=False)
    gamma = tf.Variable(tf.ones(feature_size), trainable=False)

    return gamma * (x - mean) / tf.sqrt(variance + 1e-8) + beta

