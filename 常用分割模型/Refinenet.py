"""
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
import sys
sys.setrecursionlimit(3000)
kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)

class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
                        This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = tf.Variable(self.gamma_init(shape),trainable=True)#, name='{}_gamma'.format(self.name)
        self.beta = tf.Variable(self.beta_init(shape),trainable=True)#, name='{}_beta'.format(self.name)
        #self.trainable_weights = [self.gamma, self.beta]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size,
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = Add(name='res' + str(stage) + block)([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size,
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = Add(name='res' + str(stage) + block)([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet101_model(input_shape, weights_path=None):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=input_shape, name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding = 'same')(x)
    
    # Block 1
    x = conv_block(x, (3,3), [64, 64, 256], stage=2, block='a', strides=(1,1)) #conv2_1
    x = identity_block(x, (3,3), [64, 64, 256], stage=2, block='b') #conv2_2
    block_1_out = identity_block(x, (3,3), [64, 64, 256], stage=2, block='c') #conv2_3

    # Block 2
    x = conv_block(block_1_out, (3,3), [128, 128, 512], stage=3, block='a') #conv3_1
    for i in range(1,3):
      x = identity_block(x, (3,3), [128, 128, 512], stage=3, block='b'+str(i)) #conv3_2-3
    block_2_out = identity_block(x, (3,3), [128, 128, 512], stage=3, block='b3') #conv3_4

    # Block 3
    x = conv_block(block_2_out, (3,3), [256, 256, 1024], stage=4, block='a') #conv4_1
    for i in range(1,22):
      x = identity_block(x, (3,3), [256, 256, 1024], stage=4, block='b'+str(i)) #conv4_2-22
    block_3_out = identity_block(x, (3,3), [256, 256, 1024], stage=4, block='b22') #conv4_23

    # Block 4
    x = conv_block(block_3_out, (3,3), [512, 512, 2048], stage=5, block='a') #conv5_1
    x = identity_block(x, (3,3), [512, 512, 2048], stage=5, block='b') #conv5_2
    block_4_out = identity_block(x, (3,3), [512, 512, 2048], stage=5, block='c') #conv5_3
    
    model = Model(inputs = [img_input], outputs = [block_4_out, block_3_out, block_2_out, block_1_out])
  
    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print('Frontend weights loaded.')

    return model

def ResidualConvUnit(inputs,n_filters=256,kernel_size=3,name=''):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel
    Returns:
      Output of local residual block
    """
    
    net = ReLU(name=name+'relu1')(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = ReLU(name=name+'relu2')(net)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = Add(name=name+'sum')([net, inputs])
    
    return net

def ChainedResidualPooling(inputs,n_filters=256,name=''):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 2 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are 
    fused together with the input feature map through summation 
    of residual connections.
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    Returns:
      Double-pooled feature maps
    """
    
    net = ReLU(name=name+'relu')(inputs)
    net_out_1 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool1', data_format='channels_last')(net)
    net_out_2 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool2', data_format='channels_last')(net)
    net_out_3 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool3', data_format='channels_last')(net)
    net_out_4 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool4', data_format='channels_last')(net)
    net_out_5 = net
    
    net = Add(name=name+'sum')([net_out_1,net_out_2,net_out_3,net_out_4,net_out_5])

    return net


def MultiResolutionFusion(high_inputs=None,low_inputs=None,n_filters=256,name=''):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv
    Returns:
      Fused feature maps at higher resolution
    
    """
    
    if low_inputs is None: # RefineNet block 4
        return high_inputs

    else:
        conv_low = Conv2D(n_filters, 3, padding='same', name=name+'conv_lo', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(low_inputs)
        conv_low = BatchNormalization()(conv_low)
        conv_high = Conv2D(n_filters, 3, padding='same', name=name+'conv_hi', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high_inputs)
        conv_high = BatchNormalization()(conv_high)
        
        conv_low_up = UpSampling2D(size=2, interpolation='bilinear', name=name+'up')(conv_low)
        
        return Add(name=name+'sum')([conv_low_up, conv_high])


def RefineBlock(high_inputs=None,low_inputs=None,block=0):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
    Returns:
      RefineNet block for a single path i.e one resolution
    
    """

    if low_inputs is None: # block 4
        rcu_high = ResidualConvUnit(high_inputs, n_filters=512, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=512, name='rb_{}_rcu_h2_'.format(block))
        
        # nothing happens here
        fuse = MultiResolutionFusion(high_inputs = rcu_high,
                                     low_inputs = None,
                                     n_filters = 512,
                                     name = 'rb_{}_mrf_'.format(block))
        
        fuse_pooling = ChainedResidualPooling(fuse, n_filters = 512, name='rb_{}_crp_'.format(block))
        
        output = ResidualConvUnit(fuse, n_filters = 512, name='rb_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = K.int_shape(high_inputs)[-1]
        low_n = K.int_shape(low_inputs)[-1]
        
        rcu_high = ResidualConvUnit(high_inputs, n_filters = high_n, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters = high_n, name='rb_{}_rcu_h2_'.format(block))
        
        rcu_low = ResidualConvUnit(low_inputs, n_filters = low_n, name='rb_{}_rcu_l1_'.format(block))
        rcu_low = ResidualConvUnit(rcu_low, n_filters = low_n, name='rb_{}_rcu_l2_'.format(block))

        fuse = MultiResolutionFusion(high_inputs = rcu_high,
                                     low_inputs = rcu_low,
                                     n_filters = 256,
                                     name = 'rb_{}_mrf_'.format(block))
        fuse_pooling = ChainedResidualPooling(fuse, n_filters = 256, name='rb_{}_crp_'.format(block))
        output = ResidualConvUnit(fuse_pooling, n_filters = 256, name='rb_{}_rcu_o1_'.format(block))
        return output



def build_refinenet(input_shape, num_class, resnet_weights = None,
                    frontend_trainable = True):
    """
    Builds the RefineNet model. 
    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      frontend_trainable: Whether or not to freeze ResNet layers during training
    Returns:
      RefineNet model
    """
    
    # Build ResNet-101
    model_base = resnet101_model(input_shape, resnet_weights)

    # Get ResNet block output layers
    high = model_base.output
    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[0])
    high[1] = Conv2D(256, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[1])
    high[2] = Conv2D(256, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[2])
    high[3] = Conv2D(256, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[3])
    for h in high:
        h = BatchNormalization()(h)

    # RefineNet
    low[0] = RefineBlock(high_inputs = high[0], low_inputs = None, block=4) # Only input ResNet 1/32
    low[1] = RefineBlock(high_inputs = high[1], low_inputs = low[0], block=3) # High input = ResNet 1/16, Low input = Previous 1/16
    low[2] = RefineBlock(high_inputs = high[2], low_inputs = low[1], block=2) # High input = ResNet 1/8, Low input = Previous 1/8
    net = RefineBlock(high_inputs = high[3], low_inputs = low[2], block=1) # High input = ResNet 1/4, Low input = Previous 1/4.

    net = ResidualConvUnit(net, name='rf_rcu_o1_')
    net = ResidualConvUnit(net, name='rf_rcu_o2_')
    
    net = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(net)
    net = Conv2D(num_class, 1, activation = 'softmax', name='rf_pred')(net)
    
    model = Model(model_base.input,net)
    
    for layer in model.layers:
        if 'rb' in layer.name or 'rf_' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = frontend_trainable
    return model

A=build_refinenet((32,32,3), 16, resnet_weights = None,frontend_trainable = False)
A.summary()
