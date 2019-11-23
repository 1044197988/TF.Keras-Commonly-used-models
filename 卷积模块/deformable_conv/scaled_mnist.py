import numpy as np
import tensorflow as tf
from tensorflow import keras
from deformable_conv.cnn import get_deformable_cnn
from deformable_conv.utils import get_gen

batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)

inputs, x, output = get_deformable_cnn()
label = tf.placeholder(shape=[None, 10], dtype=tf.float32)
loss = tf.reduce_mean(keras.losses.categorical_crossentropy(label, output))
train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
accuracy = tf.reduce_mean(keras.metrics.categorical_accuracy(label, output))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
idx = 0
for image, labels in train_scaled_gen:
    _, losses, acc = sess.run([train_op, loss, accuracy], feed_dict={inputs:image,
                                                    label:labels})
    idx += 1
    if (idx % 100 == 0):
        print("step {}".format(idx))
        print(losses)
        print(acc)