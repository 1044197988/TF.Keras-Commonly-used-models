import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow.keras.layers import InputSpec, Layer
import tensorflow as tf

# From: https://arxiv.org/pdf/1911.09737v1.pdf
# Implementation in the paper.


def frn_layer_paper(x, tau, beta, gamma, epsilon=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # tau, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
    # Perform FRN.
    x = x * tf.math.rsqrt(nu2 + tf.abs(epsilon))
    # Return after applying the Offset-ReLU non-linearity.
    return tf.maximum(gamma * x + beta, tau)


def frn_layer_keras(x, tau, beta, gamma, epsilon=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # tau, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = K.mean(K.square(x), axis=[1, 2], keepdims=True)
    # Perform FRN.
    x = x * 1 / K.sqrt(nu2 + K.abs(epsilon))
    # Return after applying the Offset-ReLU non-linearity.
    return K.maximum(gamma * x + beta, tau)


class FRN(Layer):

    def __init__(self,
                 epsilon=1e-6,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 tau_initializers='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 tau_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 tau_constraint=None,
                 **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.tau_initializer = initializers.get(tau_initializers)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.tau_regularizer = regularizers.get(tau_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.tau_constraint = constraints.get(tau_constraint)
        self.tau = None
        self.gamma = None
        self.beta = None
        self.axis = -1

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)
        self.tau = self.add_weight(shape=shape,
                                   name='tau',
                                   initializer=self.tau_initializer,
                                   regularizer=self.tau_regularizer,
                                   constraint=self.tau_constraint)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
        self.built = True

    def call(self, inputs, training=None):
        return frn_layer_keras(
            x=inputs, tau=self.tau, beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'tau_initializer': initializers.serialize(self.tau_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'tau_regularizer': regularizers.serialize(self.tau_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'tau_constraint': constraints.serialize(self.tau_constraint)
        }
        base_config = super(FRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


