import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import engine as KE


class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

def vgg16_net(input_tensor, train_bn, train_state):
    ## vgg16 不含batch normalization layers
    
    shape = int(np.shape(input_tensor)[1])
    #print(shape)
    input_shape = keras.Input((shape, shape, 3))
    
    x = KL.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_11')(input_shape)
    #x = BatchNorm(name='bn_conv_11')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_12')(x)
    #x = BatchNorm(name='bn_conv_12')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    ot1 = KL.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    b1 = Model(inputs= input_shape, outputs=ot1, name='vgg16_b1')
    if not train_state[0]:
        b1.trainable = False
    else:
        b1.trainable = True
    #b1.summary()   
    shape = int((shape)/2)
    
    input_shape = keras.Input((shape, shape, 64))
    
    x = KL.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_21')(input_shape)
    #x = BatchNorm(name='bn_conv_21')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_22')(x)
    #x = BatchNorm(name='bn_conv_22')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    ot2 = KL.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    b2 = Model(inputs= input_shape, outputs=ot2, name='vgg16_b2')
    if not train_state[1]:
        b2.trainable = False
    else:
        b2.trainable = True
    #b2.summary()
    
    shape = int((shape)/2)
    input_shape = keras.Input((shape, shape, 128))
    
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_31')(input_shape)
    #x = BatchNorm(name='bn_conv_31')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_32')(x)
    #x = BatchNorm(name='bn_conv_32')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_33')(x)
    #x = BatchNorm(name='bn_conv_33')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    ot3 = KL.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    b3 = Model(inputs= input_shape, outputs=ot3, name='vgg16_b3')
    if not train_state[2]:
        b3.trainable = False
    else:
        b3.trainable = True
    #b3.summary()
    
    shape = int((shape)/2)
    input_shape = keras.Input((shape, shape, 256))
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_41')(input_shape)
    #x = BatchNorm(name='bn_conv_41')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_42')(x)
    #x = BatchNorm(name='bn_conv_42')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_43')(x)
    #x = BatchNorm(name='bn_conv_43')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    ot4 = KL.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    b4 = Model(inputs= input_shape, outputs=ot4, name='vgg16_b4')
    if not train_state[3]:
        b4.trainable = False
    else:
        b4.trainable = True
    #b4.summary()
    
    shape = int((shape)/2)
    input_shape = keras.Input((shape, shape, 512))
    
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_51')(input_shape)
    #x = BatchNorm(name='bn_conv_51')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_52')(x)
    #x = BatchNorm(name='bn_conv_52')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2d_53')(x)
    #x = BatchNorm(name='bn_conv_53')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    ot5 = KL.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    b5 = Model(inputs= input_shape, outputs=ot5, name='vgg16_b5')
    if not train_state[4]:
        b5.trainable = False
    else:
        b5.trainable = True
    #b5.summary()
    
    model=keras.Sequential()
    model.add(b1)
    model.add(b2)
    model.add(b3)
    model.add(b4)
    model.add(b5)
    
    return model
    #return x


def main():
    print('hello python!!')
    input_shape = keras.Input((224, 224, 3))
    #print(type(input_shape))
    #print(np.shape(input_shape)[1])
    model = vgg16_net(input_shape, True, [False, False, False, False, True])
    model.summary()
    conv_base=keras.applications.VGG16(weights='imagenet',include_top=False)
    weight = conv_base.get_weights()
    model.set_weights(weight)
    

if __name__ == '__main__':
    main()