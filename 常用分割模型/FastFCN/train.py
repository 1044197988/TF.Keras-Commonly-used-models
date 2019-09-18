from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from JPU import JPU_DeepLab

print('TensorFlow', tf.__version__)


batch_size = 24
H, W = 512, 512
num_classes = 34

image_list = sorted(glob(
    'cityscapes/dataset/train_images/*'))
mask_list = sorted(glob(
    'cityscapes/dataset/train_masks/*'))

val_image_list = sorted(glob(
    'cityscapes/dataset/val_images/*'))
val_mask_list = sorted(glob(
    'cityscapes/dataset/val_masks/*'))

print('Found', len(image_list), 'training images')
print('Found', len(val_image_list), 'validation images')

for i in range(len(image_list)):
    assert image_list[i].split(
        '/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]

for i in range(len(val_image_list)):
    assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[
        0] == val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]


def get_image(image_path, img_height=800, img_width=1600, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
    return img


def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask


def load_data(image_path, mask_path, H=512, W=512):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(
        mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    return image, mask


train_dataset = tf.data.Dataset.from_tensor_slices((image_list,
                                                    mask_list))
train_dataset = train_dataset.shuffle(buffer_size=128)
train_dataset = train_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(train_dataset)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                  val_mask_list))
val_dataset = val_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = JPU_DeepLab(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)
callbacks = [mc, tb]


model.fit(train_dataset,
          steps_per_epoch=len(image_list) // batch_size,
          epochs=300,
          validation_data=val_dataset,
          validation_steps=len(val_image_list) // batch_size,
          callbacks=callbacks)
