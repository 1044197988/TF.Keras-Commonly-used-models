"""Squeeze and Excite Inception V3 model

Major portions of this code is adapted from the applications folder of Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference
    - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
    - []() # added when paper is published on Arxiv

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape

from se import squeeze_excite_block

WEIGHTS_PATH = ''
WEIGHTS_PATH_NO_TOP = ''


def _conv2d_bn(x,
               filters,
               num_row,
               num_col,
               padding='same',
               strides=(1, 1),
               name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def SEInceptionV3(include_top=True,
                  weights=None,
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000):
    """Instantiates the Squeeze and Excite Inception v3 architecture.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = _conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = _conv2d_bn(x, 32, 3, 3, padding='valid')
    x = _conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv2d_bn(x, 80, 1, 1, padding='valid')
    x = _conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 1: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 3: 17 x 17 x 768
    branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 4: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 128, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 160, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

        # squeeze and excite block
        x = squeeze_excite_block(x)

    # mixed 7: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 192, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = _conv2d_bn(x, 192, 1, 1)
    branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3,
                           strides=(2, 2), padding='valid')

    branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = _conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # squeeze and excite block
    x = squeeze_excite_block(x)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 320, 1, 1)

        branch3x3 = _conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

        # squeeze and excite block
        x = squeeze_excite_block(x)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
