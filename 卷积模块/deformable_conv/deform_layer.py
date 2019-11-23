import tensorflow as tf
from tensorflow import keras
from deformable_conv import deform_conv


class DeformableConv2D(object):
    def __init__(self, filters, use_seperate_conv=True, **kwargs):
        self.filters = filters
        if use_seperate_conv:
            self.offset_conv = keras.layers.SeparableConv2D(filters=filters * 2, kernel_size=(3, 3), padding='same',
                                                   use_bias=False)
            self.weight_conv = keras.layers.SeparableConv2D(filters=filters, kernel_size=(3, 3), padding="same",
                                                   use_bias=False, activation=tf.nn.sigmoid)
        else:
            self.offset_conv = keras.layers.Conv2D(filters=filters*2, kernel_size=(3, 3), padding='same',
                                                   use_bias=False)
            self.weight_conv = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
                                                   use_bias=False, activation=tf.nn.sigmoid)

    def __call__(self, x):
        offsets = self.offset_conv(x)
        weights = self.weight_conv(x)
        x_shape = tf.shape(x)
        x_shape_list = x.get_shape().as_list()
        x = self._to_bc_h_w(x, x_shape)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        weights = self._to_bc_h_w(weights, x_shape)
        x_offset = deform_conv.tf_batch_map_offsets(x, offsets)
        weights = tf.expand_dims(weights, axis=1)
        weights = self._to_b_h_w_c(weights, x_shape)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)
        x_offset = tf.multiply(x_offset, weights)
        x_offset.set_shape(x_shape_list)
        return x_offset

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [x_shape[0], x_shape[3], 2, x_shape[1], x_shape[2]])
        x = tf.transpose(x, [0, 1, 3, 4, 2])
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2], 2])
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2]])
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(x, (-1, x_shape[3], x_shape[1], x_shape[2]))
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


