from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import copy
import types as python_types
import warnings

from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer as TLayer
from keras.legacy import interfaces
from tensorflow.python.framework.ops import Tensor
from cliMLe.dataProcessing import Parser
import tensorflow as tf

class Layers(object):

    @staticmethod
    def serialize( layers ):
        return ";".join( [ layer.serialize() for layer in layers ] )

    @staticmethod
    def deserialize( spec ):
        if spec is None: return None
        elif isinstance(spec,basestring):
            return [ Layer.deserialize(x) for x in spec.split(";") ]
        else: return spec

class Layer(object):

    def __init__(self, _type, _dim, **kwargs):
        self.type = _type.lower()
        self.dim = int( _dim )
        self.parms = kwargs

    def serialize(self):
        return "|".join([self.type,str(self.dim),Parser.sdict(self.parms)])

    @staticmethod
    def deserialize( spec ):
        toks = spec.split("|")
        return Layer( toks[0], toks[1], **Parser.rdict(toks[2]) )

    def instance(self, **kwargs):
        self.parms.update(kwargs)
        if self.type == "dense":
            return Dense( units=self.dim, **self.parms )
        elif self.type == "solu":
            return SOLU( units=self.dim, **self.parms )
        else:
            raise Exception( "Unrecognized layer type: " + self.type )

class SOLU(TLayer):
    """Second Order Learning Unit:

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SOLU, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim_1 = input_shape[-1]
        input_dim_2 =  input_dim_1 * input_dim_1

        self.kernel1 = self.add_weight(shape=(input_dim_1, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel2 = self.add_weight(shape=(input_dim_2, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim_1})
        self.built = True

    def o2( self, input_vector ):
        # type: (Tensor) -> Tensor
        inputs2 = K.dot(   K.transpose( input_vector ), input_vector )
        return tf.reshape( inputs2, [1,-1] )

    def call( self, inputs, mask=None ):
        # type: (tf.Tensor) -> tf.Tensor

        def fn( input_tensor ):
            input_vector = tf.reshape( input_tensor, [1, -1] )
            output1 = K.dot(input_vector, self.kernel1 )
            input_vector2 = self.o2(input_vector)
            output2 = K.dot(input_vector2, self.kernel2 )
            output = tf.add(output1, output2)
            if self.use_bias:
                output = K.bias_add(output, self.bias )
            if self.activation is not None:
                output = self.activation(output)
            return output

        result = tf.map_fn( fn, inputs )  # type: tf.Tensor
        return tf.squeeze(result)


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(SOLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))