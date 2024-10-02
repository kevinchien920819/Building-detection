import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import engine as KE

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

def VGG8(input_shape, train_bn):
    # stage 1
    x = KL.Conv2D(64, (3, 3), strides = (1, 1), name='conv_1')(input_shape)
    x = BatchNorm(name = 'BN_conv_1')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(64, (3, 3), strides = (1, 1), name='conv_2')(x)
    x = BatchNorm(name = 'BN_conv_2')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same', name='pooling_1')(x)
    
    # stage 2
    x = KL.Conv2D(128, (3, 3), strides = (1, 1), name='conv_3')(x)
    x = BatchNorm(name = 'BN_conv_3')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(128, (3, 3), strides = (1, 1), name='conv_4')(x)
    x = BatchNorm(name = 'BN_conv_4')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same', name='pooling_2')(x)
    
    # stage 3
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), name='conv_5')(x)
    x = BatchNorm(name = 'BN_conv_5')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), name='conv_6')(x)
    x = BatchNorm(name = 'BN_conv_6')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), name='conv_7')(x)
    x = BatchNorm(name = 'BN_conv_7')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same', name='pooling_3')(x)
    
    # stage 4
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_8')(x)
    x = BatchNorm(name = 'BN_conv_8')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_9')(x)
    x = BatchNorm(name = 'BN_conv_9')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_10')(x)
    x = BatchNorm(name = 'BN_conv_10')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same', name='pooling_4')(x)
    
    # stage 5
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_11')(x)
    x = BatchNorm(name = 'BN_conv_11')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_12')(x)
    x = BatchNorm(name = 'BN_conv_12')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), name='conv_13')(x)
    x = BatchNorm(name = 'BN_conv_13')(x, training = train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same', name='pooling_5')(x)
    
    return x
    
    
    
    
    
def main():
    print('hello python!!')
    input_shape = keras.Input((224, 224, 3))
    x = VGG8(input_shape, True)
    model = Model(inputs = input_shape, outputs = x, name='VGG16_reduce_pooling')
    model.summary()
    


if __name__ == '__main__':
    main()
    