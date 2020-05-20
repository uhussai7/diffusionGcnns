from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import activations
import numpy as np
import group_d6 as d6




class groupConv(Layer):

    def __init__(self, filters,
                 #kernel_size,
                 #strides=(1, 1),
                 #padding='valid',
                 #data_format=None,
                 #dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 shells=None,
                 **kwargs):
        super(groupConv, self).__init__(**kwargs)
        kernel_size = 7
        rank = 1
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.deep = 1
        if shells is None:
            shells=3
        self.shells=shells

    def build(self, input_shape):
        """
        :param input_shape: The number of channels go last
        :return:
        """

        #need to differentiate between first layer and deep layer
        channel_axis=-1 #channels always go last
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if input_dim==self.shells:
            self.deep=0
            kernel_shape = self.kernel_size + (input_dim, self.filters)
        elif self.deep==1 and input_shape[channel_axis]%12 ==0:
            input_dim=int(input_shape[channel_axis]/12)
            kernel_shape = self.kernel_size + (12,input_dim, self.filters)
        else:
            print("input_dim=%d" % (input_dim))
            raise ValueError('The number of channels in deep layer '
                             'not divisible by order of group '
                             'something is wrong')

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        super(groupConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)