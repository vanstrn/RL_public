from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as KL
import numpy as np
from tensorflow.keras.regularizers import l2

class LSTM_Reshape(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,name='lstmreshape',**kwargs):
        super(LSTM_Reshape, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, inputs):
        bs = inputs.shape[0]*inputs.shape[1]
        return K.reshape(inputs,[-1,inputs.shape[2],inputs.shape[3],inputs.shape[4]])

    def compute_output_shape(self, input_shape):
        return (input_shape[2],input_shape[3],self.nFilters)

    def get_config(self):
        config = {}
        base_config = super(LSTM_Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class LSTM_Unshape(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,dim=4,name='lstmreshape',**kwargs):
        super(LSTM_Unshape, self).__init__(**kwargs)
        self.dim=dim

    def build(self,input_shape):
        pass

    def call(self, inputs):
        # bs = int(inputs.shape[0]//self.dim)
        return K.reshape(inputs,[-1,self.dim,inputs.shape[1],inputs.shape[2],inputs.shape[3]])

    def compute_output_shape(self, input_shape):
        return (input_shape[2],input_shape[3],self.nFilters)

    def get_config(self):
        config = {}
        base_config = super(LSTM_Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    model=Sequential()
    model.add(LSTM_Reshape())
    model.add(LSTM_Unshape(dim=2))
    x=tf.convert_to_tensor(np.random.random([1,2,40,40,3]), dtype=tf.float32)
    # print(x)
    print(model(x))
