#coding=utf-8  
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D  
from tensorflow.keras.layers import add,Flatten  
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
from tensorflow.keras.optimizers import SGD  
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  
  
inpt = Input(shape=(224,224,3))  
x = ZeroPadding2D((3,3))(inpt)  
x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
  
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))  
  
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
  
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))  
x = AveragePooling2D(pool_size=(7,7))(x)  
x = Flatten()(x)  
x = Dense(1000,activation='softmax')(x)  
  
model = Model(inputs=inpt,outputs=x)  
sgd = SGD(decay=0.0001,momentum=0.9)  
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
model.summary()  
