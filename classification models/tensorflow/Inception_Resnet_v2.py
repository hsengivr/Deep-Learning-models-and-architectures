# -*- coding: utf-8 -*-
# -*- vicky -*-
"""
- Inception-ResNet-v2 -> Inception(wider than deeper) + Resnet(residuals) 
- activation scaling is done in paper . not implemented in code

- Implementation of network architectures described in:
    - (InceptionResNet)[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
    
- Requirements :
  - Python version - 3.6.3
  - Tensorflow(tf keras) - 1.1x / 2.x
"""

import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import add, AveragePooling2D, Dropout, Flatten, Dense

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
### GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = #GPU_ID from earlier



class Inception_Resnet_:
  def __init__(self,input_shape=(299,299,3), classes=10):
    self.NUM_CLASSES = classes
    self.input_shape = input_shape
  
  def BasicConv2D(self, x, filters, kernel_size, strides, padding ,act=True,name=None):
    '''
      - Function that performs Conv + BatchNorm + Activation
      - set use_bias=False in Conv2D, scale=False in BatchNormalization
    '''
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=False,
               name=name+'_conv2d')(x)   
    x = BatchNormalization(axis=-1,scale=False, name=name+'_bn')(x)
    if act:
      x = ReLU(name=name+'_act')(x)
    return x

  ### Stem Module :

  def stem_inception_resnet_v2(self,img_input):
    x = self.BasicConv2D(img_input, filters=32, kernel_size=(3, 3), strides=2, padding='valid', act=True, name='conv1')  
    x = self.BasicConv2D(x, filters=32, kernel_size=(3,3), strides=1, padding='valid', act=True, name='conv2')  
    x = self.BasicConv2D(x, filters=64, kernel_size=(3,3), strides=1, padding='same', act=True, name='conv3')   #padding='same '

    b1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='valid' ,name='b1_maxpool')(x)                           # stride=1 / 2 , padding='valid'
    b2 = self.BasicConv2D(x, filters=96, kernel_size=(3,3), strides=2, padding='valid', act=True, name='b2_conv')  # 64,3,1
    x = tf.concat(values=[b1, b2], axis=-1, name='stem_concat_1') 

    b3_conv1 = self.BasicConv2D(x, filters=64, kernel_size=(1, 1), strides=1, padding="same", act=True, name='b3_conv1')
    b3_conv2 = self.BasicConv2D(b3_conv1, filters=96, kernel_size=(3, 3), strides=1, padding="valid", act=True, name='b3_conv2')

    b4_conv1 = self.BasicConv2D(x, filters=64, kernel_size=(1, 1), strides=1, padding="same", act=True, name='b4_conv1')
    b4_conv2 = self.BasicConv2D(b4_conv1, filters=64, kernel_size=(7, 1), strides=1, padding="same", act=True, name='b4_conv2') #(7,1)
    b4_conv3 = self.BasicConv2D(b4_conv2, filters=64, kernel_size=(1, 7), strides=1, padding="same", act=True, name='b4_conv3') #(1,7)
    b4_conv4 = self.BasicConv2D(b4_conv3, filters=96, kernel_size=(3, 3), strides=1, padding="valid", act=True, name='b4_conv4')
    x = tf.concat(values=[b3_conv2, b4_conv4], axis=-1, name='stem_concat_2')

    b5_conv = self.BasicConv2D(x, filters=192, kernel_size=(3, 3), strides=2, padding="valid", act=True, name='b5_conv')   # stride= 1 doest work as mentioned ,2 only works 
    b6_maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid", name='b6_maxpool')(x) #stride==2
    x = tf.concat(values=[b5_conv, b6_maxpool], axis=-1, name='stem_concat_3')

    return x

  ### Inception-Resnet-A

  def inception_resnet_A(self, input, name=None):
    b1_conv = self.BasicConv2D(input, filters=32, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b1_conv')

    b2_conv1 = self.BasicConv2D(input, filters=32, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b2_conv1')
    b2_conv2 = self.BasicConv2D(b2_conv1, filters=32, kernel_size=(3, 3), strides=1, padding="same", name=name+'_b2_conv2')
  
    b3_conv1 = self.BasicConv2D(input, filters=32, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b3_conv1')
    b3_conv2 = self.BasicConv2D(b3_conv1, filters=48, kernel_size=(3, 3), strides=1, padding="same", name=name+'_b3_conv2')
    b3_conv3 = self.BasicConv2D(b3_conv2, filters=64, kernel_size=(3, 3), strides=1, padding="same", name=name+'_b3_conv3')

    x = tf.concat(values=[b1_conv, b2_conv2, b3_conv3], axis=-1, name=name+'_concat')
    x = self.BasicConv2D(x, filters=384, kernel_size=(1, 1), strides=1, padding="same", act=False ,name=name+'_conv_linear')

    output = tf.keras.layers.add([x, input],name=name+'_add')
    output = ReLU(name=name+'_activation_scaling')(output)   # perform activation scaling 
  
    return x

  ### Inception-ResNet-B

  def inception_resnet_B(self, input, name=None):
    b1_conv = self.BasicConv2D(input, filters=192, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b1_conv')
  
    b2_conv1 = self.BasicConv2D(input, filters=128, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b2_conv1')
    b2_conv2 = self.BasicConv2D(b2_conv1, filters=160, kernel_size=(1, 7), strides=1, padding="same", name=name+'_b2_conv2')
    b2_conv3 = self.BasicConv2D(b2_conv2, filters=192, kernel_size=(7, 1), strides=1, padding="same", name=name+'_b2_conv3')
 
    x = tf.concat(values=[b1_conv, b2_conv3], axis=-1, name=name+'_concat')
    x = self.BasicConv2D(x, filters=1152, kernel_size=(1, 1), strides=1, padding="same", act=False ,name=name+'_conv_linear') # filters =1152 instead of 1154
 
    output = add([x,input],name=name+'_add')
    output = ReLU(name=name+'_activation_scaling')(output)   # perform activation scaling  #tf.nn.relu(output,name=name+'_activation_scaling') 

    return output

  ### Inception-ResNet-C

  def inception_resnet_C(self, input, name=None):
    b1_conv = self.BasicConv2D(input, filters=192, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b1_conv')
  
    b2_conv1 = self.BasicConv2D(input, filters=192, kernel_size=(1, 1), strides=1, padding="same", name=name+'_b2_conv1')
    b2_conv2 = self.BasicConv2D(b2_conv1, filters=224, kernel_size=(1, 3), strides=1, padding="same", name=name+'_b2_conv2')
    b2_conv3 = self.BasicConv2D(b2_conv2, filters=256, kernel_size=(3, 1), strides=1, padding="same", name=name+'_b2_conv3')
  
    x = tf.concat(values=[b1_conv, b2_conv3], axis=-1, name=name+'_concat')
    x = self.BasicConv2D(x, filters=2144, kernel_size=(1, 1), strides=1, padding="same", act=False ,name=name+'_conv_linear') # filters =2144 instead of 2048 
  
    output = add([x, input],name=name+'_add')
    output = ReLU(name=name+'activation_scaling')(output)   # perform activation scaling 

    return output

  ### Reduction-A

  # 35 × 35 to 17 × 17 reduction module.
  def reduction_A(self, input, k, l, m, n):
    b1_conv = self.BasicConv2D(input, filters=n, kernel_size=(3, 3), strides=2, padding="valid", name='redA_b1_conv')

    b2_pool = MaxPooling2D(pool_size=(3,3), strides=2, padding='valid' ,name='redA_b2_maxpool')(input)                          
  
    b3_conv1 = self.BasicConv2D(input, filters=k, kernel_size=(1, 1), strides=1, padding="same", name='redA_b3_conv1')
    b3_conv2 = self.BasicConv2D(b3_conv1, filters=l, kernel_size=(3, 3), strides=1, padding="same", name='redA_b3_conv2')
    b3_conv3 = self.BasicConv2D(b3_conv2, filters=m, kernel_size=(3, 3), strides=2, padding="valid", name='redA_b3_conv3')
    x = tf.concat(values=[b1_conv, b2_pool, b3_conv3], axis=-1, name='redA_concat')
  
    return x

  ### Reduction-B

  #17 × 17 to 8 × 8 reduction module.
  def reduction_B(self, input):
    b1_pool = MaxPooling2D(pool_size=(3,3), strides=2, padding='valid' ,name='redB_b1_maxpool')(input)                          

    b2_conv1 = self.BasicConv2D(input, filters=256, kernel_size=(1, 1), strides=1, padding="same", name='redB_b2_conv1')
    b2_conv2 = self.BasicConv2D(b2_conv1, filters=384, kernel_size=(3, 3), strides=2, padding="valid", name='redB_b2_conv2')
  
    b3_conv1 = self.BasicConv2D(input, filters=256, kernel_size=(1, 1), strides=1, padding="same", name='redB_b3_conv1')
    b3_conv2 = self.BasicConv2D(b3_conv1, filters=288, kernel_size=(3, 3), strides=2, padding="valid", name='redB_b3_conv2')
  
    b4_conv1 = self.BasicConv2D(input, filters=256, kernel_size=(1, 1), strides=1, padding="same", name='redB_b4_conv1')
    b4_conv2 = self.BasicConv2D(b4_conv1, filters=288, kernel_size=(3, 3), strides=1, padding="same", name='redB_b4_conv2')
    b4_conv3 = self.BasicConv2D(b4_conv2, filters=320, kernel_size=(3, 3), strides=2, padding="valid", name='redB_b4_conv3')
    x = tf.concat(values=[b1_pool, b2_conv2, b3_conv2, b4_conv3], axis=-1, name='redB_concat')
    return x


  ### Inception-Resnet-v2 model

  def Inception_ResNet_V2(self, k, l, m, n):
    
    img_input = tf.keras.layers.Input(shape = self.input_shape) 
    
    # stem
    x = self.stem_inception_resnet_v2(img_input)     # shape=(None, 35, 35, 384)
  
    #Inception-ResNet-A modules
    x = self.inception_resnet_A(x, name='incresA_1')
    x = self.inception_resnet_A(x, name='incresA_2')
    x = self.inception_resnet_A(x, name='incresA_3')
    x = self.inception_resnet_A(x, name='incresA_4')
    x = self.inception_resnet_A(x, name='incresA_5')

    x = self.reduction_A(x,k,l,m,n)

    #Inception-ResNet-B modules
    x = self.inception_resnet_B(x, name='incresB_1')
    x = self.inception_resnet_B(x, name='incresB_2')
    x = self.inception_resnet_B(x, name='incresB_3')
    x = self.inception_resnet_B(x, name='incresB_4')
    x = self.inception_resnet_B(x, name='incresB_5')
    x = self.inception_resnet_B(x, name='incresB_6')
    x = self.inception_resnet_B(x, name='incresB_7')
    x = self.inception_resnet_B(x, name='incresB_8')
    x = self.inception_resnet_B(x, name='incresB_9')
    x = self.inception_resnet_B(x, name='incresB_10')

    x = self.reduction_B(x)

    #Inception-ResNet-C modules
    x = self.inception_resnet_C(x, name='incresC_1')
    x = self.inception_resnet_C(x, name='incresC_2')
    x = self.inception_resnet_C(x, name='incresC_3')
    x = self.inception_resnet_C(x, name='incresC_4')
    x = self.inception_resnet_C(x, name='incresC_5')

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=self.NUM_CLASSES,activation=tf.keras.activations.softmax)(x)

    model = tf.keras.models.Model(inputs=[img_input], outputs=[x])  
    return model

if __name__ == "__main__":

  # default input dim for Inception Resnext = (299,299,3)
  # image dimensions  
  img_height = 299
  img_width = 299
  img_channels = 3

  # model params
  input_shape = (img_height, img_width, img_channels)
  num_classes = 10
  (k,l,m,n) = (256, 256, 384, 384)

  # Model 
  network = Inception_Resnet_(input_shape, num_classes)
  model = network.Inception_ResNet_V2(k,l,m,n)    # Reduction_block_A (k,l,m,n) changes based on architecture 
  print(model.summary())