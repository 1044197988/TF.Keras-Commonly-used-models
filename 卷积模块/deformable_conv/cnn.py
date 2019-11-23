import tensorflow as tf
from tensorflow import keras
from deformable_conv.deform_layer import DeformableConv2D


def get_deformable_cnn():
    inputs = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    # conv11
    x = keras.layers.Conv2D(32, (3, 3), padding='same', name='conv11')(inputs)
    x = keras.layers.ReLU(name='conv11_relu')(x)
    x = keras.layers.BatchNormalization()(x)
    # conv12
    x_offset = DeformableConv2D(32)(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", strides=(2, 2))(x_offset)
    x = keras.layers.ReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    # conv21
    x_offset = DeformableConv2D(64)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x_offset)
    x = keras.layers.ReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    # conv22
    x_offset = DeformableConv2D(128)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x_offset)
    x = keras.layers.ReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    #out
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10)(x)
    output =keras.layers.Softmax()(x)
    return inputs, x, output



