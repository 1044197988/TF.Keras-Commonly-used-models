from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

#@tf.keras.utils.register_keras_serializable(package='Text')

"""Gaussian error linear unit."""
def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


"""Customized Swish activation."""
def swish(features):
  """Computes the Swish activation function.
  The tf.nn.swish operation uses a custom gradient to reduce memory usage.
  Since saving custom gradients in SavedModel is currently not supported, and
  one would not be able to use an exported TF-Hub module for fine-tuning, we
  provide this wrapper that can allow to select whether to use the native
  TensorFlow swish operation, or whether to use a customized operation that
  has uses default TensorFlow gradient computation.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return features * tf.nn.sigmoid(features)

def mish(features):
  features = tf.convert_to_tensor(features)
  return features * tf.nn.tanh(tf.nn.softplus(features))
