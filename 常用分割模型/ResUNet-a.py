from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Add, Lambda, Activation, GlobalAveragePooling2D, DepthwiseConv2D
import tensorflow as tf

def ResBlock(x,f,k,d,s):
    x = BatchNormalization()(x)
    x = Conv2D(f, k, s, dilation_rate=d,activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f, k, s, dilation_rate=d,activation='relu', padding='same')(x)
    return x

def ResBlock_a(x,f,k,d,s):
    if len(d)==4:
        x1 = ResBlock(x,f,k,d[0],s)
        x2 = ResBlock(x,f,k,d[1],s)
        x3 = ResBlock(x,f,k,d[2],s)
        x4 = ResBlock(x,f,k,d[3],s)
        x = Add()([x1, x2, x3, x4])
    elif len(d)==3:
        x1 = ResBlock(x,f,k,d[0],s)
        x2 = ResBlock(x,f,k,d[1],s)
        x3 = ResBlock(x,f,k,d[2],s)
        x = Add()([x1, x2, x3])
    else:
        x = ResBlock(x,f,k,d[0],s)
    return x

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    return x
    
def Get_Net(img_rows,img_cols,c,n_classes):
    inputs = Input((img_rows, img_cols, c))
    x_1 = Conv2D(32, 1, 1, dilation_rate=1,activation='relu', padding='same')(inputs)
    x_2 = ResBlock_a(x_1,32,3,[1,3,15,31],1)
    x = Conv2D(64, 1, 2, dilation_rate=1,activation='relu', padding='same')(x_2)
    x_3 = ResBlock_a(x,64,3,[1,3,15,31],1)
    x = Conv2D(128, 1, 2, dilation_rate=1,activation='relu', padding='same')(x_3)
    x_4 = ResBlock_a(x,128,3,[1,3,15],1)
    x = Conv2D(256, 1, 2, dilation_rate=1,activation='relu', padding='same')(x_4)
    x_5 = ResBlock_a(x,256,3,[1,3,15],1)
    x = Conv2D(512, 1, 2, dilation_rate=1,activation='relu', padding='same')(x_5)
    x_6 = ResBlock_a(x,512,3,[1],1)                         
    x = Conv2D(1024, 1, 2, dilation_rate=1,activation='relu', padding='same')(x_6)
    
    x = ResBlock_a(x,1024,3,[1],1)

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4 = tf.expand_dims(tf.expand_dims(b4, 1),1)
    b4 = Conv2D(256, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, shape_before[1:3], method='bilinear',align_corners=True))(b4)
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    b1 = SepConv_BN(x, 256, 'aspp1',rate=6, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 256, 'aspp2',rate=9, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 256, 'aspp3',rate=12, depth_activation=True, epsilon=1e-5)
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = UpSampling2D()(x)
    x = Concatenate()([x_6,x])
    x = ResBlock_a(x,512,3,[1],1)

    x = UpSampling2D()(x)
    x = Concatenate()([x_5,x])
    x = ResBlock_a(x,256,3,[1,3,15],1)

    x = UpSampling2D()(x)
    x = Concatenate()([x_4,x])
    x = ResBlock_a(x,128,3,[1,3,15],1)

    x = UpSampling2D()(x)
    x = Concatenate()([x_3,x])
    x = ResBlock_a(x,64,3,[1,3,15,31],1)

    x = UpSampling2D()(x)
    x = Concatenate()([x_2,x])
    x = ResBlock_a(x,32,3,[1,3,15,31],1)
    x = Concatenate()([x_1,x])

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4 = tf.expand_dims(tf.expand_dims(b4, 1),1)
    b4 = Conv2D(128, (1, 1), padding='same',use_bias=False, name='image_pooling_1')(b4)
    b4 = BatchNormalization(name='image_pooling_BN_1', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, shape_before[1:3], method='bilinear',align_corners=True))(b4)
    b0 = Conv2D(128, (1, 1), padding='same', use_bias=False, name='aspp3')(x)
    b0 = BatchNormalization(name='aspp3_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp3_activation')(b0)
    b1 = SepConv_BN(x, 128, 'aspp4',rate=6, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 128, 'aspp5',rate=9, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 128, 'aspp6',rate=12, depth_activation=True, epsilon=1e-5)
    x = Concatenate()([b4, b0, b1, b2, b3])
    
    x = Conv2D(n_classes, (1, 1), padding='same',activation='softmax')(x)

    model = Model(inputs,x)
    
    return model
    
A = Get_Net(128,128,3,16)
A.summary()
    
