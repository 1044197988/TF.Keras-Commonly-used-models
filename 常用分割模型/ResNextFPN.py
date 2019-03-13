from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Activation, Conv2D, MaxPooling2D, Add, Input, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.regularizers import l2


TOP_DOWN_PYRAMID_SIZE = 256

"""
Implementation of Resnext FPN 
"""


def resnext_fpn(input_shape, nb_labels, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True,
                batch_momentum=0.9):
    """
    TODO: add dilated convolutions as well
    Resnext-50 is defined by (3, 4, 6, 3) [default]
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    :param input_shape:
    :param nb_labels:
    :param depth:
    :param cardinality:
    :param width:
    :param weight_decay:
    :param batch_norm:
    :param batch_momentum:
    :return:
    """
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)

    bn_axis = 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    stage_1 = x

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x

    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x

    x = bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x

    x = bottleneck_block(x, 1024, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, weight_decay=weight_decay)
    stage_5 = x

    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(stage_5)
    P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4', padding='same')(stage_4)])
    P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(stage_3)])
    P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
    # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)

    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv")(P2)
    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv_2")(head1)

    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv")(P3)
    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv_2")(head2)

    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv")(P4)
    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv_2")(head3)

    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv")(P5)
    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv_2")(head4)

    f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
    f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
    f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
    f_p5 = head1

    x = Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])
    x = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv", kernel_initializer='he_normal',
               activation='linear')(x)
    x = UpSampling2D(size=(4, 4), name="final_upsample")(x)
    x = Activation('sigmoid')(x)

    model = Model(input_tensor, x)

    return model


def grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=3)
    x = BatchNormalization(axis=3)(group_merge)
    x = Activation('relu')(x)
    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input
    grouped_channels = int(filters / cardinality)

    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=3)(x)

    x = add([init, x])
    x = Activation('relu')(x)
    return x


A=resnext_fpn((256,256,3), 10, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True,
                batch_momentum=0.9)
A.summary()
