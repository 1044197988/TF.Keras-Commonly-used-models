from tensorflow.keras.layers import Input, concatenate, add, Multiply, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, MaxPooling2D, UpSampling2D, UpSampling3D, Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


# Get neural network
def get_net(inp_shape, algorithm):
    if  algorithm == 'liver_att_resunet_2d':
        return build_res_atten_unet_2d(inp_shape, filter_num=4)
    elif algorithm == 'liver_att_resunet_3d' or algorithm == 'liver_tumor_att_resunet_3d':
        return build_res_atten_unet_3d(inp_shape)
    elif algorithm == 'brain_tumor_res_atten_unet_3d':
        return build_brain_tumor_res_atten_unet_3d(inp_shape, filter_num=8)


# ============================================================
# ======================Attention ResUnet 3D================================#
# ============================================================


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1, name='out'):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """
    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk, output_channels=output_channels)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPooling3D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPooling3D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling3D()(output_soft_mask)
        ## skip connections
        output_soft_mask = add([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling3D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv3D(input_channels, (1, 1, 1))(output_soft_mask)
    output_soft_mask = Conv3D(input_channels, (1, 1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output, name=name)

    return output


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3, 3), stride=1, name='out'):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv3D(input_channels, (1, 1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(output_channels, (1, 1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv3D(output_channels, (1, 1, 1), padding='same', strides=strides)(input)
    if name == 'out':
        x = add([x, input])
    else:
        x = add([x, input], name=name)
    return x


def build_brain_tumor_res_atten_unet_3d(input_shape, filter_num=8, merge_axis=-1):
    data = Input(shape=input_shape)
    pool_size = (2, 2, 2)
    up_size = (2, 2, 2)
    conv1 = Conv3D(filter_num * 4, 3, padding='same')(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # conv1 = Dropout(0.5)(conv1)

    pool = MaxPooling3D(pool_size=pool_size)(conv1)

    res1 = residual_block(pool, output_channels=filter_num * 8)
    # res1 = Dropout(0.5)(res1)

    pool1 = MaxPooling3D(pool_size=pool_size)(res1)

    res2 = residual_block(pool1, output_channels=filter_num * 16)
    # res2 = Dropout(0.5)(res2)

    pool2 = MaxPooling3D(pool_size=pool_size)(res2)

    res3 = residual_block(pool2, output_channels=filter_num * 32)
    # res3 = Dropout(0.5)(res3)

    pool3 = MaxPooling3D(pool_size=pool_size)(res3)

    res4 = residual_block(pool3, output_channels=filter_num * 64)
    # res4 = Dropout(0.5)(res4)

    pool4 = MaxPooling3D(pool_size=pool_size)(res4)

    res5 = residual_block(pool4, output_channels=filter_num * 64)
    res5 = residual_block(res5, output_channels=filter_num * 64)

    atb5 = attention_block(res4, encoder_depth=1, name='atten1')
    up1 = UpSampling3D(size=up_size)(res5)
    merged1 = concatenate([up1, atb5], axis=merge_axis)

    res5 = residual_block(merged1, output_channels=filter_num * 64)
    # res5 = Dropout(0.5)(res5)

    atb6 = attention_block(res3, encoder_depth=2, name='atten2')
    up2 = UpSampling3D(size=up_size)(res5)
    merged2 = concatenate([up2, atb6], axis=merge_axis)

    res6 = residual_block(merged2, output_channels=filter_num * 32)
    # res6 = Dropout(0.5)(res6)

    atb7 = attention_block(res2, encoder_depth=3, name='atten3')
    up3 = UpSampling3D(size=up_size)(res6)
    merged3 = concatenate([up3, atb7], axis=merge_axis)

    res7 = residual_block(merged3, output_channels=filter_num * 16)
    # res7 = Dropout(0.5)(res7)

    atb8 = attention_block(res1, encoder_depth=4, name='atten4')
    up4 = UpSampling3D(size=up_size)(res7)
    merged4 = concatenate([up4, atb8], axis=merge_axis)

    res8 = residual_block(merged4, output_channels=filter_num * 8)
    # res8 = Dropout(0.5)(res8)

    up = UpSampling3D(size=up_size)(res8)
    merged = concatenate([up, conv1], axis=merge_axis)
    conv9 = Conv3D(filter_num * 4, 3, padding='same')(merged)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    # conv9 = Dropout(0.5)(conv9)

    output = Conv3D(1, 3, padding='same', activation='sigmoid')(conv9)
    model = Model(data, output)
    return model


# liver network do not modify
def build_res_atten_unet_3d(input_shape, filter_num=8, merge_axis=-1, pool_size=(2, 2, 2)
                            , up_size=(2, 2, 2)):
    data = Input(shape=input_shape)

    conv1 = Conv3D(filter_num * 4, 3, padding='same')(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    pool = MaxPooling3D(pool_size=pool_size)(conv1)

    res1 = residual_block(pool, output_channels=filter_num * 4)

    pool1 = MaxPooling3D(pool_size=pool_size)(res1)

    res2 = residual_block(pool1, output_channels=filter_num * 8)

    pool2 = MaxPooling3D(pool_size=pool_size)(res2)

    res3 = residual_block(pool2, output_channels=filter_num * 16)
    pool3 = MaxPooling3D(pool_size=pool_size)(res3)

    res4 = residual_block(pool3, output_channels=filter_num * 32)

    pool4 = MaxPooling3D(pool_size=pool_size)(res4)

    res5 = residual_block(pool4, output_channels=filter_num * 64)
    res5 = residual_block(res5, output_channels=filter_num * 64)

    atb5 = attention_block(res4, encoder_depth=1, name='atten1')
    up1 = UpSampling3D(size=up_size)(res5)
    merged1 = concatenate([up1, atb5], axis=merge_axis)

    res5 = residual_block(merged1, output_channels=filter_num * 32)

    atb6 = attention_block(res3, encoder_depth=2, name='atten2')
    up2 = UpSampling3D(size=up_size)(res5)
    merged2 = concatenate([up2, atb6], axis=merge_axis)

    res6 = residual_block(merged2, output_channels=filter_num * 16)
    atb7 = attention_block(res2, encoder_depth=3, name='atten3')
    up3 = UpSampling3D(size=up_size)(res6)
    merged3 = concatenate([up3, atb7], axis=merge_axis)

    res7 = residual_block(merged3, output_channels=filter_num * 8)
    atb8 = attention_block(res1, encoder_depth=4, name='atten4')
    up4 = UpSampling3D(size=up_size)(res7)
    merged4 = concatenate([up4, atb8], axis=merge_axis)

    res8 = residual_block(merged4, output_channels=filter_num * 4)
    up = UpSampling3D(size=up_size)(res8)
    merged = concatenate([up, conv1], axis=merge_axis)
    conv9 = Conv3D(filter_num * 4, 3, padding='same')(merged)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    output = Conv3D(1, 3, padding='same', activation='sigmoid')(conv9)
    model = Model(data, output)
    return model


# ============================================================
# ======================Attention ResUnet 2D================================#
# ============================================================


def attention_block_2d(input, input_channels=None, output_channels=None, encoder_depth=1, name='at'):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """
    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block_2d(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block_2d(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPooling2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block_2d(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block_2d(output_soft_mask)
        skip_connections.append(output_skip_connection)

        ## down sampling
        output_soft_mask = MaxPooling2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block_2d(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block_2d(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = add([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block_2d(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block_2d(output, name=name)

    return output


def residual_block_2d(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1, name='out'):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
    if name == 'out':
        x = add([x, input])
    else:
        x = add([x, input], name=name)
    return x


def build_res_atten_unet_2d(input_shape, filter_num=8):
    merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=input_shape)

    conv1 = Conv2D(filter_num * 4, 3, padding='same')(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # res0 = residual_block_2d(data, output_channels=filter_num * 2)

    pool = MaxPooling2D(pool_size=(2, 2))(conv1)

    res1 = residual_block_2d(pool, output_channels=filter_num * 4)

    # res1 = residual_block_2d(atb1, output_channels=filter_num * 4)

    pool1 = MaxPooling2D(pool_size=(2, 2))(res1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(atb1)

    res2 = residual_block_2d(pool1, output_channels=filter_num * 8)

    # res2 = residual_block_2d(atb2, output_channels=filter_num * 8)
    pool2 = MaxPooling2D(pool_size=(2, 2))(res2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(atb2)

    res3 = residual_block_2d(pool2, output_channels=filter_num * 16)
    # res3 = residual_block_2d(atb3, output_channels=filter_num * 16)
    pool3 = MaxPooling2D(pool_size=(2, 2))(res3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(atb3)

    res4 = residual_block_2d(pool3, output_channels=filter_num * 32)

    # res4 = residual_block_2d(atb4, output_channels=filter_num * 32)
    pool4 = MaxPooling2D(pool_size=(2, 2))(res4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(atb4)

    res5 = residual_block_2d(pool4, output_channels=filter_num * 64)
    # res5 = residual_block_2d(res5, output_channels=filter_num * 64)
    res5 = residual_block_2d(res5, output_channels=filter_num * 64)

    atb5 = attention_block_2d(res4, encoder_depth=1, name='atten1')
    up1 = UpSampling2D(size=(2, 2))(res5)
    merged1 = concatenate([up1, atb5], axis=merge_axis)
    # merged1 = concatenate([up1, atb4], axis=merge_axis)

    res5 = residual_block_2d(merged1, output_channels=filter_num * 32)
    # atb5 = attention_block_2d(res5, encoder_depth=1)

    atb6 = attention_block_2d(res3, encoder_depth=2, name='atten2')
    up2 = UpSampling2D(size=(2, 2))(res5)
    # up2 = UpSampling2D(size=(2, 2))(atb5)
    merged2 = concatenate([up2, atb6], axis=merge_axis)
    # merged2 = concatenate([up2, atb3], axis=merge_axis)

    res6 = residual_block_2d(merged2, output_channels=filter_num * 16)
    # atb6 = attention_block_2d(res6, encoder_depth=2)

    # atb6 = attention_block_2d(res6, encoder_depth=2)
    atb7 = attention_block_2d(res2, encoder_depth=3, name='atten3')
    up3 = UpSampling2D(size=(2, 2))(res6)
    # up3 = UpSampling2D(size=(2, 2))(atb6)
    merged3 = concatenate([up3, atb7], axis=merge_axis)
    # merged3 = concatenate([up3, atb2], axis=merge_axis)

    res7 = residual_block_2d(merged3, output_channels=filter_num * 8)
    # atb7 = attention_block_2d(res7, encoder_depth=3)

    # atb7 = attention_block_2d(res7, encoder_depth=3)
    atb8 = attention_block_2d(res1, encoder_depth=4, name='atten4')
    up4 = UpSampling2D(size=(2, 2))(res7)
    # up4 = UpSampling2D(size=(2, 2))(atb7)
    merged4 = concatenate([up4, atb8], axis=merge_axis)
    # merged4 = concatenate([up4, atb1], axis=merge_axis)

    res8 = residual_block_2d(merged4, output_channels=filter_num * 4)
    # atb8 = attention_block_2d(res8, encoder_depth=4)

    # atb8 = attention_block_2d(res8, encoder_depth=4)
    up = UpSampling2D(size=(2, 2))(res8)
    # up = UpSampling2D(size=(2, 2))(atb8)
    merged = concatenate([up, conv1], axis=merge_axis)
    # res9 = residual_block_2d(merged, output_channels=filter_num * 2)

    conv9 = Conv2D(filter_num * 4, 3, padding='same')(merged)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    output = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)
    model = Model(data, output)
    return model

if __name__ == "__main__":
    model = get_net((128,128,3), 'liver_att_resunet_2d')
    model.summary()
