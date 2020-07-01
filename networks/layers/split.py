from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as KL
import numpy as np
from tensorflow.keras.regularizers import l2

class Split(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,units,name='denseSplit',**kwargs):
        super(Split, self).__init__(**kwargs)
        self.units=units

    def build(self,input_shape):
        pass

    def call(self, inputs):

        return inputs[:,:self.units],inputs[:,self.units:]

    def compute_output_shape(self, input_shape):
        return [(input_shape[2],input_shape[3],self.nFilters)]

    def get_config(self):
        config = {}
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BasicCNNSplit(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,division,name='cnnSplit',**kwargs):
        super(BasicCNNSplit, self).__init__(**kwargs)
        self.division=division

    def build(self,input_shape):
        pass

    def call(self, inputs):

        return inputs[:,:,:,:self.division],inputs[:,:,:,self.division:]

    def compute_output_shape(self, input_shape):
        return [(input_shape[2],input_shape[3],self.nFilters)]

    def get_config(self):
        config = {}
        base_config = super(BasicCNNSplit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ComplexCNNSplit(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,divisions,name='cnnSplit',**kwargs):
        super(ComplexCNNSplit, self).__init__(**kwargs)
        self.divisions=divisions

    def build(self,input_shape):
        pass

    def call(self, inputs):
        outList = []
        for division in self.divisions:
            outList.append(inputs[:,:,:,np.r_[division]])
        return outList

    def compute_output_shape(self, input_shape):
        return [(input_shape[2],input_shape[3],self.nFilters)]

    def get_config(self):
        config = {}
        base_config = super(ComplexCNNSplit, self).get_config()
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
