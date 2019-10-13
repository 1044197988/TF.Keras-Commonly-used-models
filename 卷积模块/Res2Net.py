from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add

def Conv_Relu_BN(num_filters,kernel_size,batchnorm=True,strides=(1, 1),padding='same'):
    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,padding=padding, kernel_initializer='he_normal',strides=strides)(input_tensor)
        x = Activation('relu')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x
    return layer

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list

def res2net_block(num_filters, slice_num):
    def layer(input_tensor):
        short_cut = input_tensor
        x = Conv_Mish_BN(num_filters=num_filters, kernel_size=(1, 1))(input_tensor)
        slice_list = slice_layer(x, slice_num, x.shape[-1])
        side = Conv_Mish_BN(num_filters=num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
        z = concatenate([slice_list[0], side])   # for one and second stage
        for i in range(2, len(slice_list)):
            y = Conv_Mish_BN(num_filters=num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
            side = y
            z = concatenate([z, y])
        z = Conv_Mish_BN(num_filters=num_filters, kernel_size=(1, 1))(z)
        out = concatenate([z, short_cut])
        return out
    return layer
