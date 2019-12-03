"""
create_time:2019.12.3
https://github.com/1044197988
"""
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,UpSampling2D,MaxPooling2D,Concatenate

class conv_block(Model):
    """
    Convolution Block
    """
    def __init__(self, filters):
        super(conv_block, self).__init__()

        self.conv = Sequential([
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.conv(x)
        return x

class up_conv(Model):
    """
    Up Convolution Block
    """

    def __init__(self, filters):
        super(up_conv, self).__init__()
        self.up = Sequential([
            UpSampling2D(),
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.up(x)
        return x

class U_Net(Model):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self,classes=16):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = MaxPooling2D(strides=2)
        self.Maxpool2 = MaxPooling2D(strides=2)
        self.Maxpool3 = MaxPooling2D(strides=2)
        self.Maxpool4 = MaxPooling2D(strides=2)

        self.Conv1 = conv_block(filters[0])
        self.Conv2 = conv_block(filters[1])
        self.Conv3 = conv_block(filters[2])
        self.Conv4 = conv_block(filters[3])
        self.Conv5 = conv_block(filters[4])

        self.Up5 = up_conv(filters[3])
        self.Up_conv5 = conv_block(filters[3])

        self.Up4 = up_conv(filters[2])
        self.Up_conv4 = conv_block(filters[2])

        self.Up3 = up_conv(filters[1])
        self.Up_conv3 = conv_block(filters[1])

        self.Up2 = up_conv(filters[0])
        self.Up_conv2 = conv_block(filters[0])

        self.Conv = Conv2D(classes,kernel_size=1, strides=1, padding='same',activation='softmax',name='final_layer')

    def call(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = Concatenate()([e4, d5])

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = Concatenate()([e3, d4])
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = Concatenate()([e2, d3])
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = Concatenate()([e1, d2])
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

class Recurrent_block(Model):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = Sequential([
            Conv2D(out_ch, kernel_size=(3, 3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out

class RRCNN_block(Model):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = Sequential([
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        ])
        self.Conv = Conv2D(out_ch, kernel_size=(1, 1), strides=1, padding='same')

    def call(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

class R2U_Net(Model):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, classes=16, t=2):
        super(R2U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = MaxPooling2D(strides=2)
        self.Maxpool1 = MaxPooling2D(strides=2)
        self.Maxpool2 = MaxPooling2D(strides=2)
        self.Maxpool3 = MaxPooling2D(strides=2)

        self.RRCNN1 = RRCNN_block(filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[4], t=t)

        self.Up5 = up_conv(filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[3], t=t)

        self.Up4 = up_conv(filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[2], t=t)

        self.Up3 = up_conv(filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[1], t=t)

        self.Up2 = up_conv(filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[0], t=t)

        self.Conv = Conv2D(classes, kernel_size=1, strides=1, padding='same',name='final_layer')

    def call(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = Concatenate()([e4, d5])
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = Concatenate()([e3, d4])
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = Concatenate()([e2, d3])
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = Concatenate()([e1, d2])
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)
        return out

class Attention_block(Model):
    """
    Attention Block
    """

    def __init__(self, filters):
        super(Attention_block, self).__init__()

        self.W_g = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization()
        ])

        self.W_x = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization()
        ])

        self.psi = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization(),
            Activation('sigmoid')
        ])

        self.relu = Activation('relu')

    def call(self, x):
        g1 = self.W_g(x[0])
        x1 = self.W_x(x[1])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x[1] * psi
        return out

class AttU_Net(Model):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, classes=16):
        super(AttU_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = MaxPooling2D(strides=2)
        self.Maxpool2 = MaxPooling2D(strides=2)
        self.Maxpool3 = MaxPooling2D(strides=2)
        self.Maxpool4 = MaxPooling2D(strides=2)

        self.Conv1 = conv_block(filters[0])
        self.Conv2 = conv_block(filters[1])
        self.Conv3 = conv_block(filters[2])
        self.Conv4 = conv_block(filters[3])
        self.Conv5 = conv_block(filters[4])

        self.Up5 = up_conv(filters[3])
        self.Att5 = Attention_block(filters[3])
        self.Up_conv5 = conv_block(filters[3])

        self.Up4 = up_conv(filters[2])
        self.Att4 = Attention_block(filters[2])
        self.Up_conv4 = conv_block(filters[2])

        self.Up3 = up_conv(filters[1])
        self.Att3 = Attention_block(filters[1])
        self.Up_conv3 = conv_block(filters[1])

        self.Up2 = up_conv(filters[0])
        self.Att2 = Attention_block(filters[0])
        self.Up_conv2 = conv_block(filters[0])

        self.Conv = Conv2D(classes, kernel_size=1, strides=1, padding='same',activation='softmax',name='final_layer')

    def call(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5([d5,e4])
        d5 = Concatenate()([x4, d5])
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4([d4,e3])
        d4 = Concatenate()([x3, d4])
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3([d3,e2])
        d3 = Concatenate()([x2, d3])
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2([d2,e1])
        d2 = Concatenate()([x1, d2])
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

class R2AttU_Net(Model):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, classes=16, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = MaxPooling2D(strides=2)
        self.Maxpool2 = MaxPooling2D(strides=2)
        self.Maxpool3 = MaxPooling2D(strides=2)
        self.Maxpool4 = MaxPooling2D(strides=2)

        self.RRCNN1 = RRCNN_block(filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[4], t=t)

        self.Up5 = up_conv(filters[3])
        self.Att5 = Attention_block(filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[3], t=t)

        self.Up4 = up_conv(filters[2])
        self.Att4 = Attention_block(filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[2], t=t)

        self.Up3 = up_conv(filters[1])
        self.Att3 = Attention_block(filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[1], t=t)

        self.Up2 = up_conv(filters[0])
        self.Att2 = Attention_block(filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[0], t=t)

        self.Conv = Conv2D(classes, kernel_size=1, strides=1, padding='same',activation='softmax',name='final_layer')

    def call(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5([d5,e4])
        d5 = Concatenate()([e4, d5])
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4([d4,e3])
        d4 = Concatenate()([e3, d4])
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3([d3,e2])
        d3 = Concatenate()([e2, d3])
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2([d2,e1])
        d2 = Concatenate()([e1, d2])
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)
        return out

class conv_block_nested(Model):

    def __init__(self, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = Activation('relu')
        self.conv1 = Conv2D(mid_ch, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_ch, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class NestedUNet(Model):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, classes=16):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = MaxPooling2D(strides=2)
        self.Up = UpSampling2D()

        self.conv0_0 = conv_block_nested(filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0], filters[0])

        self.final = Conv2D(classes, kernel_size=1,activation='softmax',name='final_layer')

    def call(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(Concatenate()([x0_0, self.Up(x1_0)]))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(Concatenate()([x1_0, self.Up(x2_0)]))
        x0_2 = self.conv0_2(Concatenate()([x0_0, x0_1, self.Up(x1_1)]))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(Concatenate()([x2_0, self.Up(x3_0)]))
        x1_2 = self.conv1_2(Concatenate()([x1_0, x1_1, self.Up(x2_1)]))
        x0_3 = self.conv0_3(Concatenate()([x0_0, x0_1, x0_2, self.Up(x1_2)]))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(Concatenate()([x3_0, self.Up(x4_0)]))
        x2_2 = self.conv2_2(Concatenate()([x2_0, x2_1, self.Up(x3_1)]))
        x1_3 = self.conv1_3(Concatenate()([x1_0, x1_1, x1_2, self.Up(x2_2)]))
        x0_4 = self.conv0_4(Concatenate()([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)]))

        output = self.final(x0_4)
        return output


if __name__=="__main__":
    #model = U_Net()
    #model = R2U_Net()
    #model = AttU_Net()
    #model = R2AttU_Net()
    model = NestedUNet()
    model.build((None,128,128,3))
    model.summary()
