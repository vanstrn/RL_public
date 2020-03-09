from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as KL
import numpy as np
from tensorflow.keras.regularizers import l2

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    model=Sequential()
    model.add(ReverseInception())
    x=tf.convert_to_tensor(np.random.random([1,40,40,3]), dtype=tf.float32)
    print(x)
    print(model(x))
