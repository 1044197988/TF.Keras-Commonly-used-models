from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing, regularizers
from tensorflow.keras  import backend as K
import tensorflow as tf
import numpy as np




def MnasNet(n_classes=1000, input_shape=(224, 224, 3), alpha=1):
	inputs = layers.Input(shape=input_shape)

	x = conv_bn(inputs, 32*alpha, 3,   strides=2)
	x = sepConv_bn_noskip(x, 16*alpha, 3,  strides=1) 
	# MBConv3 3x3
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv3 5x5
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(x, filters=80, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	x = MBConv_idskip(x, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	x = MBConv_idskip(x, filters=320, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)

	# FC + POOL
	x = conv_bn(x, filters=1152*alpha, kernel_size=1,   strides=1)
	x = layers.GlobalAveragePooling2D()(x)
	predictions = layers.Dense(n_classes, activation='softmax')(x)

	return models.Model(inputs=inputs, outputs=predictions)




# Convolution with batch normalization
def conv_bn(x, filters, kernel_size,  strides=1, alpha=1, activation=True):
	"""Convolution Block
	This function defines a 2D convolution operation with BN and relu6.
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		alpha: An integer which multiplies the filters dimensionality
		activation: A boolean which indicates whether to have an activation after the normalization 
	# Returns
		Output tensor.
	"""
	filters = _make_divisible(filters * alpha)
	x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
									use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
	x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)  
	if activation:
		x = layers.ReLU(max_value=6)(x)
	return x

# Depth-wise Separable Convolution with batch normalization 
def depthwiseConv_bn(x, depth_multiplier, kernel_size,  strides=1):
	""" Depthwise convolution 
	The DepthwiseConv2D is just the first step of the Depthwise Separable convolution (without the pointwise step).
	Depthwise Separable convolutions consists in performing just the first step in a depthwise spatial convolution 
	(which acts on each input channel separately).
	
	This function defines a 2D Depthwise separable convolution operation with BN and relu6.
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
	# Returns
		Output tensor.
	"""

	x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
									padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)  
	x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)  
	x = layers.ReLU(max_value=6)(x)
	return x

def sepConv_bn_noskip(x, filters, kernel_size,  strides=1):
	""" Separable convolution block (Block F of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)
	
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
	# Returns
		Output tensor.
	"""

	x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	x = conv_bn(x, filters=filters, kernel_size=1, strides=1)

	return x

# Inverted bottleneck block with identity skip connection
def MBConv_idskip(x_input, filters, kernel_size,  strides=1, filters_multiplier=1, alpha=1):
	""" Mobile inverted bottleneck convolution (Block b, c, d, e of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)
	
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		alpha: An integer which multiplies the filters dimensionality
	# Returns
		Output tensor.
	"""

	depthwise_conv_filters = _make_divisible(x_input.shape[3].value) 
	pointwise_conv_filters = _make_divisible(filters * alpha)

	x = conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
	x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	x = conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

	# Residual connection if possible
	if strides==1 and x.shape[3] == x_input.shape[3]:
		return  layers.add([x_input, x])
	else: 
		return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v



if __name__ == "__main__":

	model = MnasNet()
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
	model.summary()
