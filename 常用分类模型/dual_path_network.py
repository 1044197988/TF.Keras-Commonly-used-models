'''
Dual Path Networks
Combines ResNeXt grouped convolutions and DenseNet dense
connections to acheive state-of-the-art performance on ImageNet

References:
    - [Dual Path Networks](https://arxiv.org/abs/1707.01629)
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Lambda,Activation,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D,add,concatenate
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.utils import conv_utils
#from tensorflow.keras.utils.data_utils import get_file
#from tensorflow.keras.engine.topology import get_source_inputs
#from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K

#__all__ = ['DualPathNetwork', 'DPN92', 'DPN98', 'DPN137', 'DPN107', 'preprocess_input', 'decode_predictions']


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
       Obtained from https://github.com/cypw/DPNs

        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.

        # Returns
            Preprocessed tensor.
        """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 104
        x[:, 1, :, :] -= 117
        x[:, 2, :, :] -= 128
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 104
        x[:, :, :, 1] -= 117
        x[:, :, :, 2] -= 124

    x *= 0.0167
    return x


def DualPathNetwork(input_shape=None,
                    initial_conv_filters=64,
                    depth=[3, 4, 20, 3],
                    filter_increment=[16, 32, 24, 128],
                    cardinality=32,
                    width=3,
                    weight_decay=0,
                    include_top=True,
                    weights=None,
                    input_tensor=None,
                    pooling=None,
                    classes=1000):
    """ Instantiate the Dual Path Network architecture for the ImageNet dataset. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                DPN-92  = [3, 4, 20, 3]
                DPN-98  = [3, 6, 20, 3]
                DPN-131 = [4, 8, 28, 3]
                DPN-107 = [4, 8, 20, 3]
            filter_increment: number of filters incremented per block, defined as a list.
                DPN-92  = [16, 32, 24, 128]
                DON-98  = [16, 32, 32, 128]
                DPN-131 = [16, 32, 32, 128]
                DPN-107 = [20, 64, 64, 128]
            cardinality: the size of the set of transformations
            width: width multiplier for the network
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
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
                - `max-avg` means that both global average and global max
                    pooling will be applied to the output of the last
                    convolution layer
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filter_increment), "The length of filter increment list must match the length " \
                                                "of the depth list."

    # Determine proper input shape
    input_shape=input_shape
    """
    input_shape = (input_shape,
                   default_size=224,
                   min_size=112,
                   data_format=K.image_data_format(),
                   require_flatten=include_top)
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_dpn(classes, img_input, include_top, initial_conv_filters,
                    filter_increment, depth, cardinality, width, weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
    #    inputs = get_source_inputs(input_tensor)
    #else:
    #    inputs = img_input
    # Create model.
    model = Model(img_input, x, name='resnext')

    # load weights

    return model


def DPN92(input_shape=None,
          include_top=True,
          weights=None,
          input_tensor=None,
          pooling=None,
          classes=1000):
    return DualPathNetwork(input_shape, include_top=include_top, weights=weights, input_tensor=input_tensor,
                           pooling=pooling, classes=classes)


def DPN98(input_shape=None,
          include_top=True,
          weights=None,
          input_tensor=None,
          pooling=None,
          classes=1000):
    return DualPathNetwork(input_shape, initial_conv_filters=96, depth=[3, 6, 20, 3], filter_increment=[16, 32, 32, 128],
                           cardinality=40, width=4, include_top=include_top, weights=weights, input_tensor=input_tensor,
                           pooling=pooling, classes=classes)


def DPN137(input_shape=None,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000):
    return DualPathNetwork(input_shape, initial_conv_filters=128, depth=[4, 8, 28, 3], filter_increment=[16, 32, 32, 128],
                           cardinality=40, width=4, include_top=include_top, weights=weights, input_tensor=input_tensor,
                           pooling=pooling, classes=classes)


def DPN107(input_shape=None,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000):
    return DualPathNetwork(input_shape, initial_conv_filters=128, depth=[4, 8, 20, 3], filter_increment=[20, 64, 64, 128],
                           cardinality=50, width=4, include_top=include_top, weights=weights, input_tensor=input_tensor,
                           pooling=pooling, classes=classes)


def _initial_conv_block_inception(input, initial_conv_filters, weight_decay=5e-4):
    ''' Adds an initial conv block, with batch norm and relu for the DPN
    Args:
        input: input tensor
        initial_conv_filters: number of filters for initial conv block
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=(2, 2))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    return x


def _bn_relu_conv_block(input, filters, kernel=(3, 3), stride=(1, 1), weight_decay=5e-4):
    ''' Adds a Batchnorm-Relu-Conv block for DPN
    Args:
        input: input tensor
        filters: number of output filters
        kernel: convolution kernel size
        stride: stride of convolution
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=stride)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def _grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
                   if K.image_data_format() == 'channels_last' else
                   lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    group_merge = Activation('relu')(group_merge)
    return group_merge


def _dual_path_block(input, pointwise_filters_a, grouped_conv_filters_b, pointwise_filters_c,
                     filter_increment, cardinality, block_type='normal'):
    '''
    Creates a Dual Path Block. The first path is a ResNeXt type
    grouped convolution block. The second is a DenseNet type dense
    convolution block.

    Args:
        input: input tensor
        pointwise_filters_a: number of filters for the bottleneck
            pointwise convolution
        grouped_conv_filters_b: number of filters for the grouped
            convolution block
        pointwise_filters_c: number of filters for the bottleneck
            convolution block
        filter_increment: number of filters that will be added
        cardinality: cardinality factor
        block_type: determines what action the block will perform
            - `projection`: adds a projection connection
            - `downsample`: downsamples the spatial resolution
            - `normal`    : simple adds a dual path connection

    Returns: a list of two output tensors - one path of ResNeXt
        and another path for DenseNet

    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    grouped_channels = int(grouped_conv_filters_b / cardinality)

    init = concatenate(input, axis=channel_axis) if isinstance(input, list) else input

    if block_type == 'projection':
        stride = (1, 1)
        projection = True
    elif block_type == 'downsample':
        stride = (2, 2)
        projection = True
    elif block_type == 'normal':
        stride = (1, 1)
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)

    if projection:
        projection_path = _bn_relu_conv_block(init, filters=pointwise_filters_c + 2 * filter_increment,
                                              kernel=(1, 1), stride=stride)
        input_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c]
                                     if K.image_data_format() == 'channels_last' else
                                     z[:, :pointwise_filters_c, :, :])(projection_path)
        input_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:]
                                  if K.image_data_format() == 'channels_last' else
                                  z[:, pointwise_filters_c:, :, :])(projection_path)
    else:
        input_residual_path = input[0]
        input_dense_path = input[1]

    x = _bn_relu_conv_block(init, filters=pointwise_filters_a, kernel=(1, 1))
    x = _grouped_convolution_block(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride)
    x = _bn_relu_conv_block(x, filters=pointwise_filters_c + filter_increment, kernel=(1, 1))

    output_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c]
                                  if K.image_data_format() == 'channels_last' else
                                  z[:, :pointwise_filters_c, :, :])(x)
    output_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:]
                               if K.image_data_format() == 'channels_last' else
                               z[:, pointwise_filters_c:, :, :])(x)

    residual_path = add([input_residual_path, output_residual_path])
    dense_path = concatenate([input_dense_path, output_dense_path], axis=channel_axis)

    return [residual_path, dense_path]


def _create_dpn(nb_classes, img_input, include_top, initial_conv_filters,
                filter_increment, depth, cardinality=32, width=3, weight_decay=5e-4, pooling=None):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        initial_conv_filters: number of features for the initial convolution
        filter_increment: number of filters incremented per block, defined as a list.
            DPN-92  = [16, 32, 24, 128]
            DON-98  = [16, 32, 32, 128]
            DPN-131 = [16, 32, 32, 128]
            DPN-107 = [20, 64, 64, 128]
        depth: number or layers in the each block, defined as a list.
            DPN-92  = [3, 4, 20, 3]
            DPN-98  = [3, 6, 20, 3]
            DPN-131 = [4, 8, 28, 3]
            DPN-107 = [4, 8, 20, 3]
        width: width multiplier for network
        weight_decay: weight_decay (l2 norm)
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
            - `max-avg` means that both global average and global max
                pooling will be applied to the output of the last
                convolution layer
    Returns: a Keras Model
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)
    base_filters = 256

    # block 1 (initial conv block)
    x = _initial_conv_block_inception(img_input, initial_conv_filters, weight_decay)

    # block 2 (projection block)
    filter_inc = filter_increment[0]
    filters = int(cardinality * width)

    x = _dual_path_block(x, pointwise_filters_a=filters,
                         grouped_conv_filters_b=filters,
                         pointwise_filters_c=base_filters,
                         filter_increment=filter_inc,
                         cardinality=cardinality,
                         block_type='projection')

    for i in range(N[0] - 1):
        x = _dual_path_block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='normal')

    # remaining blocks
    for k in range(1, len(N)):
        print("BLOCK %d" % (k + 1))
        filter_inc = filter_increment[k]
        filters *= 2
        base_filters *= 2

        x = _dual_path_block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='downsample')

        for i in range(N[k] - 1):
            x = _dual_path_block(x, pointwise_filters_a=filters,
                                 grouped_conv_filters_b=filters,
                                 pointwise_filters_c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='normal')

    x = concatenate(x, axis=channel_axis)

    if include_top:
        avg = GlobalAveragePooling2D()(x)
        max = GlobalMaxPooling2D()(x)
        x = add([avg, max])
        x = Lambda(lambda z: 0.5 * z)(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        elif pooling == 'max-avg':
            a = GlobalMaxPooling2D()(x)
            b = GlobalAveragePooling2D()(x)
            x = add([a, b])
            x = Lambda(lambda z: 0.5 * z)(x)

    return x

if __name__ == '__main__':
    #model = DPN92((224, 224, 3),classes=10)
    #model = DPN98((224, 224, 3),classes=10)
    #model = DPN107((224, 224, 3),classes=10)
    model = DPN137((224, 224, 3),classes=10)
    model.summary()
