from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add, LeakyReLU, ZeroPadding2D
from tensorflow.keras.applications.xception import Xception

img_w, img_h = (256, 256) 
                
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def build_model(start_neurons):
    
    backbone = Xception(input_shape=(img_h, img_w, 3), weights=None, include_top=False)
    input = backbone.input

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons*32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons*32)
    convm = residual_block(convm, start_neurons*32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons*16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)
    
    uconv4 = Conv2D(start_neurons*16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons*16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons*8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(0.1)(uconv3)
    
    uconv3 = Conv2D(start_neurons*8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons*4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons*4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons*2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons*2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    # 128 -> 256
    uconv0 = Conv2DTranspose(start_neurons*1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons*1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(0.1/2)(uconv0)
    output_layer_noActi = Conv2D(3, (1,1), padding="same", activation=None)(uconv0)
    model = Model(inputs=input, outputs=output_layer_noActi)

    return model

model = build_model(16) 
model.summary()  
