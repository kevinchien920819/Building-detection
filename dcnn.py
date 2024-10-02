import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import engine as KE



########################################################################################################################
# 54 Automatic Classification of Ovarian Cancer Types from Cytological Images Using Deep Convolutional Neural Networks #
########################################################################################################################

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

def dcnn_net(input_tensor, train_bn):
    x = KL.Conv2D(96, (11, 11), strides=(4, 4), name='conv_1')(input_tensor)
    #print(x)
    x = BatchNorm(name='bn_conv_1')(x, training=train_bn)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    #print(x)
    x = KL.Conv2D(256, (5, 5), strides=(1, 1), name='conv_2', padding='same')(x)
    #print(x)
    x = BatchNorm(name='bn_conv_2')(x, training=train_bn)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    #print(x)
    x = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv_3', padding='same')(x)
    
    #print(x)
    x = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv_4', padding='same')(x)
    
    #print(x)
    x = KL.Conv2D(256, (3, 3), strides=(1, 1), name='conv_5', padding='same')(x)
    #print(x)
    x = BatchNorm(name='bn_conv_5')(x, training=train_bn)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    #print(x)
    return x

if __name__ =='__main__':
    print('hello DCNN !!')
    input_shape = keras.Input((227, 227, 3))
    x = dcnn_net(input_shape, True)
    #print(x)
    model = Model(inputs = input_shape, outputs=x, name='dcnn')
    model.summary()
    #model.compile()
    #print(model)
    