import numpy as np
import warnings


import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import engine as KE
#import tensorflow.python.keras.layers as KL
#import tensorflow.python.keras.engine as KE
#import tensorflow.python.keras.models as KM

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True): 
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'
    
    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_base_name+'2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_base_name+'2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_base_name+'2c')(x, training=train_bn)
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out' )(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'
    
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_base_name+'2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_base_name+'2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_base_name+'2c')(x, training=train_bn)
    
    short_cut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    short_cut = BatchNorm(name=bn_base_name+'1')(short_cut, training=train_bn)
    
    x = KL.Add()([x, short_cut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

    

def resnet_graph(input_image, architecture, stage5=False, train_bn=True): # 
    assert architecture in ['resnet50', 'resnet101']
    # stage1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    #print(C1.shape)
    
    #stage2
    x = conv_block(x, 3, [64, 64, 256], stage='2', block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage='2', block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage='2', block='c', train_bn=train_bn)
    
    #stage3
    x = conv_block(x, 3, [128, 128, 512], stage='3', block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage='3', block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage='3', block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage='3', block='d', train_bn=train_bn)
    
    #stage4
    x = conv_block(x, 3, [256, 256, 1024], stage='4', block='a', train_bn=train_bn)
    block_count = {'resnet50': 5, 'resnet101': 22}[architecture]
    for i in range(block_count) :
        x = identity_block(x, 3, [256, 256, 1024], stage='4', block=str(98+i), train_bn=train_bn)
    C4 = x
    
    #stage5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage='5', block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage='5', block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage='5', block='c', train_bn=train_bn)
    else:
        C5 = None
    
    return x

def resnet18(input_image, train_bn=True):
    #stage1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    #stage2
    x = KL.Conv2D(64, (3, 3), name='conv2_1a', use_bias=True)(x)
    x = BatchNorm(name='bn_conv2_1a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(64, (3, 3), name='conv2_1b', use_bias=True)(x)
    x = BatchNorm(name='bn_conv2_1b')(x, training=train_bn)
     
    
    x = KL.Conv2D(64, (3, 3), name='conv2_2a', use_bias=True)(x)
    x = BatchNorm(name='bn_conv2_2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(64, (3, 3), name='conv2_2b', use_bias=True)(x)
    x2b = x = BatchNorm(name='bn_conv2_2b')(x, training=train_bn)
   
    
    #stage3
    x = KL.Conv2D(128, (3, 3), name='conv3_1a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv3_1a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(128, (3, 3), name='conv3_1b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv3_1b')(x, training=train_bn)
    short_cut = KL.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x2b)
    short_cut = BatchNorm(name='conv3_1_sc')(short_cut, training=train_bn)
    x = KL.Add()([x, short_cut])
    x3a = x = KL.Activation('relu', name='res_conv3_1_out')(x)  
    
    x = KL.Conv2D(128, (3, 3), name='conv3_2a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv3_2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(128, (3, 3), name='conv3_2b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv3_2b')(x, training=train_bn)
    #x = KL.Add()([x, x3a])
    x3b = x = KL.Activation('relu', name='res_conv3_2_out')(x)  
    
    #stage4
    x = KL.Conv2D(256, (3, 3), name='conv4_1a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv4_1a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(256, (3, 3), name='conv4_1b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv4_1b')(x, training=train_bn)
    short_cut = KL.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x3b)
    short_cut = BatchNorm(name='conv4_1_sc')(short_cut, training=train_bn)
    x = KL.Add()([x, short_cut])
    x4a = x = KL.Activation('relu', name='res_conv4_1_out')(x)  
    
    x = KL.Conv2D(256, (3, 3), name='conv4_2a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv4_2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(256, (3, 3), name='conv4_2b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv4_2b')(x, training=train_bn)
    #x = KL.Add()([x, x4a])
    x4b = x = KL.Activation('relu', name='res_conv4_2_out')(x)
    
    #stage5
    x = KL.Conv2D(512, (3, 3), name='conv5_1a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv5_1a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), name='conv5_1b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv5_1b')(x, training=train_bn)
    short_cut = KL.Conv2D(512, (1, 1), strides=(2, 2), padding='same')(x4b)
    short_cut = BatchNorm(name='conv5_1_sc')(short_cut, training=train_bn)
    x = KL.Add()([x, short_cut])
    x5a = x = KL.Activation('relu', name='res_conv5_1_out')(x)  
    
    x = KL.Conv2D(512, (3, 3), name='conv5_2a', strides=(2, 2), use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv5_2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(512, (3, 3), name='conv5_2b', use_bias=True, padding='same')(x)
    x = BatchNorm(name='bn_conv5_2b')(x, training=train_bn)
    #x = KL.Add()([x, x5a])
    x = KL.Activation('relu', name='res_conv5_2_out')(x)
    
    return x
    
    
    
   

def main():
    print("hello resnet!!")
    input_shape=keras.Input(shape=(256, 256, 3))
    #x = resnet_graph(input_shape, 'resnet50')
    x = resnet18(input_shape, True)
    model = Model(inputs = input_shape, outputs = x, name = 'resnet')

if __name__ == '__main__':
    main()