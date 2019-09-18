import tensorflow as tf
from resnet import ResNet101
print('TensorFlow', tf.__version__)


def conv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
    x = tf.keras.layers.Conv2D(filters=num_filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               strides=strides,
                               dilation_rate=dilation_rate,
                               kernel_initializer=w_init,
                               use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def sepconv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
    x = tf.keras.layers.SeparableConv2D(filters=num_filters,
                                        depth_multiplier=1,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        strides=strides,
                                        dilation_rate=dilation_rate,
                                        depthwise_initializer=w_init,
                                        use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def JPU(endpoints: list, out_channels=512):
    h, w = endpoints[1].shape.as_list()[1:3]
    for i in range(1, 4):
        endpoints[i] = conv_block(endpoints[i], out_channels, 3)
        if i != 1:
            h_t, w_t = endpoints[i].shape.as_list()[1:3]
            scale = (h // h_t, w // w_t)
            endpoints[i] = tf.keras.layers.UpSampling2D(
                size=scale, interpolation='bilinear')(endpoints[i])
    yc = tf.keras.layers.Concatenate(axis=-1)(endpoints[1:])
    ym = []
    for rate in [1, 2, 4, 8]:
        ym.append(sepconv_block(yc, 512, 3, dilation_rate=rate))
    y = tf.keras.layers.Concatenate(axis=-1)(ym)
    return endpoints, y


def ASPP(tensor):
    dims = tensor.shape.as_list()

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = conv_block(y_pool, num_filters=256, kernel_size=1)

    h_t, w_t = y_pool.shape.as_list()[1:3]
    scale = dims[1] // h_t, dims[2] // w_t
    y_pool = tf.keras.layers.UpSampling2D(
        size=scale, interpolation='bilinear')(y_pool)

    y_1 = conv_block(tensor, num_filters=256, kernel_size=1, dilation_rate=1)
    y_6 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=6)
    y_6.set_shape([None, dims[1], dims[2], 256])
    y_12 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=12)
    y_12.set_shape([None, dims[1], dims[2], 256])
    y_18 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=18)
    y_18.set_shape([None, dims[1], dims[2], 256])

    y = tf.keras.layers.Concatenate(axis=-1)([y_pool, y_1, y_6, y_12, y_18])
    y = conv_block(y, num_filters=256, kernel_size=1)
    return y


def JPU_DeepLab(img_height=1024, img_width=1024, nclasses=19):
    base_model = ResNet101(include_top=False,
                           input_shape=[img_height, img_width, 3],
                           weights=None)#'imagenet'
    endpoint_names = ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out']
    endpoints = [base_model.get_layer(x).output for x in endpoint_names]

    _, image_features = JPU(endpoints)

    x_a = ASPP(image_features)
    h_t, w_t = x_a.shape.as_list()[1:3]
    scale = (img_height / 4) // h_t, (img_width / 4) // w_t
    x_a = tf.keras.layers.UpSampling2D(
        size=scale, interpolation='bilinear')(x_a)

    x_b = base_model.get_layer('conv2_block3_out').output
    x_b = conv_block(x_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([x_a, x_b])
    x = conv_block(x, num_filters=256, kernel_size=3)
    x = conv_block(x, num_filters=256, kernel_size=3)
    h_t, w_t = x.shape.as_list()[1:3]
    scale = img_height // h_t, img_width // w_t
    x = tf.keras.layers.UpSampling2D(size=scale, interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(nclasses, (1, 1), name='output_layer')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x, name='JPU')
    return model
