# -*- coding: utf-8 -*-
# -*- vicky -*- 
"""
Inception V3 model for Keras: 
  - Input size in architecture (299x299 instead of 224x224)
  - implemented conv2d with batch norm

Requirements :
  - works in Tf1.1x and Tf2.x

Reference :
- [2016 CVPR] [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
"""

from __future__ import print_function, absolute_import
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
### GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = #GPU_ID from earlier

def conv2d_bn(x, filters_, kernel_size_, strides_=(1, 1), padding_='same',batch_norm=True, name=None):
    """
    - Utility function to apply conv + BN + activation.
    """  
    x = Conv2D(filters=filters_, 
               kernel_size=kernel_size_ , 
               strides=strides_ , 
               padding = padding_, 
               use_bias=False)(x)
    if batch_norm == True:
      x = BatchNormalization(axis=-1, scale=False)(x)  #    bn_axis
    x = Activation('relu')(x)
    return x

def InceptionA(input,f1,f2,f3,f4,f5,f6,f7):
  '''
  Factorization Into Smaller Convolutions to reduce the number of connections/parameters without decreasing the network efficiency.
  Two 3×3 convolutions replaces one 5×5 convolution
  '''
    # branch 1
  conv_a1 = conv2d_bn(input, f1, (1, 1))

  # branch 2
  conv_b1 = conv2d_bn(input, f2, (1, 1))
  conv_b2 = conv2d_bn(conv_b1, f3, (3, 3))    # (5,5)

  # branch 3
  conv_c1 = conv2d_bn(input, f4, (1, 1))
  conv_c2 = conv2d_bn(conv_c1, f5, (3, 3))
  conv_c3 = conv2d_bn(conv_c2, f6, (3, 3))

  # branch 4
  pool_d1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
  conv_d2 = conv2d_bn(pool_d1, f7, (1, 1))
   
  outputs = [conv_a1, conv_b2, conv_c3, conv_d2]
  concat = concatenate(outputs, axis=-1)  
  return concat

def InceptionB(input,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10):
  '''
  Factorization Into Asymmetric Convolutions (Inception B):
  One 3×1 convolution followed by one 1×3 convolution replaces one 3×3 convolution
  '''
  # branch1
  conv_a1 = conv2d_bn(input, f1, (1, 1))

  # branch2
  conv_b1 = conv2d_bn(input, f2, (1, 1))
  conv_b2 = conv2d_bn(conv_b1, f3, (1, 7))
  conv_b3 = conv2d_bn(conv_b2, f4, (7, 1))

  # branch3
  conv_c1 = conv2d_bn(input, f5, (1, 1))
  conv_c2 = conv2d_bn(conv_c1, f6, (7, 1))
  conv_c3 = conv2d_bn(conv_c2, f7, (1, 7))
  conv_c4 = conv2d_bn(conv_c3, f8, (7, 1))
  conv_c5 = conv2d_bn(conv_c4, f9, (1, 7))

  # branch4
  pool_d1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
  conv_d2 = conv2d_bn(pool_d1, f10, (1, 1))

  outputs = [conv_a1, conv_b3, conv_c5, conv_d2]
  concat = concatenate(outputs, axis=-1)
  return concat

def InceptionC(input,f1,f2,f3,f4,f5,f6,f7,f8,f9):
  '''
  Promotes high dimensional representations
  '''
  # branch1
  conv_a1 = conv2d_bn(input, f1, (1, 1))

  # branch2
  conv_b1 = conv2d_bn(input, f2, (1, 1))
  conv_b2 = conv2d_bn(conv_b1, f3, (1, 3))
  conv_b3 = conv2d_bn(conv_b1, f4, (3, 1))
  concat_b4 = concatenate([conv_b2 ,conv_b3], axis=-1) 

  # branch3
  conv_c1 = conv2d_bn(input, f5, (1, 1))
  conv_c2 = conv2d_bn(conv_c1, f6, (3, 3))
  conv_c3 = conv2d_bn(conv_c2, f7, (1, 3))
  conv_c4 = conv2d_bn(conv_c2, f8, (3, 1))
  concat_c5 = concatenate([conv_c3,conv_c4], axis=-1)

  # branch4
  pool_d1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
  conv_d2 = conv2d_bn(pool_d1, f9, (1, 1))

  outputs = [conv_a1, concat_b4, concat_c5, conv_d2]
  concat = concatenate(outputs, axis=-1)
  return concat

def ReductionA(input,f1,f2,f3,f4):
  conv_a1 = conv2d_bn(input, f1, (3, 3), strides_=(2, 2), padding_='valid')

  conv_b1 = conv2d_bn(input, f2, (1, 1))
  conv_b2 = conv2d_bn(conv_b1, f3, (3, 3))
  conv_b3 = conv2d_bn(conv_b2, f4, (3, 3), strides_=(2, 2), padding_='valid')

  pool_c1 = MaxPooling2D((3, 3), strides=(2, 2))(input)
 
  outputs = [conv_a1, conv_b3, pool_c1]
  concat = concatenate(outputs, axis=-1)
  return concat

def ReductionB(input,f1,f2,f3,f4,f5,f6):
  # branch1
  conv_a1 = conv2d_bn(input, f1, (1, 1))
  conv_a2 = conv2d_bn(conv_a1, f2, (3, 3), strides_=(2, 2), padding_='valid')

  # branch2
  conv_b1 = conv2d_bn(input, f3, (1, 1))
  conv_b2 = conv2d_bn(conv_b1, f4, (1, 7))
  conv_b3 = conv2d_bn(conv_b2, f5, (7, 1))
  conv_b4 = conv2d_bn(conv_b3, f6, (3, 3), strides_=(2, 2), padding_='valid')

  # branch3
  pool_c1 = MaxPooling2D((3, 3), strides=(2, 2))(input)
  
  outputs = [conv_a2, conv_b4, pool_c1]
  concat = concatenate(outputs, axis=-1)
  return concat

def aux_classifier(x,num_classes):
    conv1 = AveragePooling2D((5,5),strides=3,padding='valid')(x)
    conv2 = conv2d_bn(conv1, 128, (1,1), strides_=(1,1))
    conv3 = tf.keras.layers.Flatten()(conv2)
    conv3 = Dense(num_classes,activation='softmax')(conv3)
    return conv3

def InceptionV3(input_shape=(299,299,3), classes=10):
  NUM_CLASSES = classes
  img_input = Input(shape=input_shape)

  x = conv2d_bn(img_input, 32, (3, 3), strides_=(2, 2), padding_='valid')
  x = conv2d_bn(x, 32, (3, 3), padding_='valid')
  x = conv2d_bn(x, 64, (3, 3))
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv2d_bn(x, 80, (1, 1), padding_='valid') 
  x = conv2d_bn(x, 192, (3, 3), padding_='valid')
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  # mixed 0, 1, 2: 35 x 35 x 256   # InceptionA(input,f1,f2,f3,f4,f5,f6,f7)
  x = InceptionA(x, 64, 48, 64, 64, 96, 96, 64)
  x = InceptionA(x, 64, 48, 64, 64, 96, 96, 64)
  x = InceptionA(x, 64, 48, 64, 64, 96, 96, 64)
  
  x = ReductionA(x, 384, 64, 96, 96)

  # InceptionB(input,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10):
  x = InceptionB(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192)
  x = InceptionB(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192) #192, 160, 160, 192, 160, 160, 160, 160, 192, 192
  x = InceptionB(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192) #192, 160, 160, 192, 160, 160, 160, 160, 192, 192
  x = InceptionB(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192) #192, 192, 192, 192, 192, 192, 192, 192, 192

  aux = aux_classifier(x,NUM_CLASSES)

  x = ReductionB(x, 192, 320, 192, 192, 192, 192)

  x = InceptionC(x, 320, 384, 384, 384, 448, 384, 384, 384, 192)
  x = InceptionC(x, 320, 384, 384, 384, 448, 384, 384, 384, 192)

  x = GlobalAveragePooling2D()(x)
  # x = Dropout(rate=0.2)(x)
  # x = Flatten()(x)
  x = Dense(units=NUM_CLASSES,activation=tf.keras.activations.softmax)(x)

  model = tf.keras.Model(inputs=img_input, outputs=x)
  return model

if __name__ == "__main__":
  # image dimensions
  img_height = 299
  img_width = 299
  channels = 3

  # Model params
  input_shape = (img_height, img_width, channels)
  num_classes = 10

  # Model 
  model = InceptionV3(input_shape, num_classes)
  print(model.summary())
