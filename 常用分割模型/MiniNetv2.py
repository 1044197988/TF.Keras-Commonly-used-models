import tensorflow as tf
from tensorflow.keras import layers, regularizers


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)


# convolution
def convolution(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)



# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # separableConv
        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x = self.conv(inputs, activation=False, training=training)
        x = self.dropout(x, training=training)
        x = layers.ReLU()(x + inputs)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x

class MininetV2Downsample(tf.keras.Model):
    def __init__(self, filters, depthwise=True):
        super(MininetV2Downsample, self).__init__()
        if depthwise:
            self.conv = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)
        else:
            self.conv = Conv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)

    def call(self, inputs, training=True):

        x = self.conv(inputs, training=training)
        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, last=False, training=True):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x, training=training)
            x = layers.ReLU()(x)

        return x


class MiniNetv2(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(MiniNetv2, self).__init__(**kwargs)

        self.down1 = MininetV2Downsample(16, depthwise=False)
        self.down2 = MininetV2Downsample(64, depthwise=True)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(128, depthwise=True)
        self.conv_mod_5 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_6 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_16 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.upsample1 = MininetV2Upsample(64, 3, strides=2)
        self.conv_mod_17 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_18 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_19 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.upsample2 = MininetV2Upsample(num_classes, 3, strides=2)

    def call(self, inputs, training=True):
        x = self.down1(inputs, training=training)
        x = self.down2(x, training=training)
        x = self.conv_mod_1(x, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        x = self.down3(x, training=training)
        x = self.conv_mod_5(x, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = self.conv_mod_13(x, training=training)
        x = self.conv_mod_14(x, training=training)
        x = self.conv_mod_15(x, training=training)
        x = self.conv_mod_16(x, training=training)
        x = self.upsample1(x, training=training)
        x = self.conv_mod_17(x, training=training)
        x = self.conv_mod_18(x, training=training)
        x = self.conv_mod_19(x, training=training)
        x = self.upsample2(x, last=True, training=training)
        x = tf.keras.activations.softmax(x, axis=-1)

        return x







class ShatheBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,  dilation_rate=1):
        super(ShatheBlock, self).__init__()

        self.kernel_size = kernel_size
        self.filters = filters

        self.conv1 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)

    def call(self, inputs, training=True):
        x2 = self.conv1(inputs, training=training)
        x3 = self.conv2(x2, training=training)
        x = self.conv3(x3, activation=False, training=training)
        if inputs.shape[3] == x.shape[3]:
            return layers.ReLU()(x + inputs)
        else:
            return layers.ReLU()(x2 + x)




class Segception(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception, self).__init__(**kwargs)
        base_model = tf.keras.applications.xception.Xception(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')
        output_1 = base_model.get_layer('block2_sepconv2_bn').output
        output_2 = base_model.get_layer('block3_sepconv2_bn').output
        output_3 = base_model.get_layer('block4_sepconv2_bn').output
        output_4 = base_model.get_layer('block13_sepconv2_bn').output
        output_5 = base_model.get_layer('block14_sepconv2_bn').output
        outputs = [output_5, output_4, output_3, output_2, output_1]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Decoder
        self.adap_encoder_1 = ShatheBlock(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = ShatheBlock(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2_1 = ShatheBlock(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2_2 = ShatheBlock(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2_22 = ShatheBlock(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2_3 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3_1 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3_2 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3_3 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3_4 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3_5 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)

        self.adap_encoder_4 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4_1 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4_2 = ShatheBlock(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = ShatheBlock(filters=64, kernel_size=3, dilation_rate=1)


        self.upsample1 = MininetV2Upsample(256, 3, strides=2)
        self.upsample2 = MininetV2Upsample(128, 3, strides=2)
        self.upsample3 = MininetV2Upsample(128, 3, strides=2)
        self.upsample4 = MininetV2Upsample(64, 3, strides=2)

        self.conv_logits = convolution(filters=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=True):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])

        x = self.upsample1(outputs[0], training=training)
        x = self.adap_encoder_1(x, training=training) + self.adap_encoder_2(outputs[1], training=training)
        x = self.adap_encoder_2_1(x, training=training)
        x = self.adap_encoder_2_2(x, training=training)
        x = self.adap_encoder_2_22(x, training=training)
        x = self.upsample2(x, training=training)
        x = self.adap_encoder_2_3(x, training=training)
        x += self.adap_encoder_3(outputs[2], training=training)
        x = self.adap_encoder_3_1(x, training=training)
        x = self.adap_encoder_3_2(x, training=training)
        x = self.adap_encoder_3_3(x, training=training)
        x = self.adap_encoder_3_4(x, training=training)
        x = self.upsample3(x, training=training)
        x = self.adap_encoder_3_5(x, training=training)
        x += reshape_into(self.adap_encoder_4(outputs[3]),x)
        x = self.adap_encoder_4_1(x, training=training)
        x = self.adap_encoder_4_2(x, training=training)
        x = self.upsample4(x, last=True, training=training)
        x = tf.keras.activations.softmax(x, axis=-1)

        return x
