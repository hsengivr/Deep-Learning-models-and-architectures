# -*- coding: utf-8 -*-
# -*- vicky -*-

"""
- ResNext architecture is same as ResNet.Only difference is group convolution
- ResNext -> Inception(wider than deeper) + Resnet(residuals)  + AlexNet(group convolution)
- Implementation of network architectures described in:
    - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
    - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

- Requirements :
  - Python version - 3.6.3
  - Tensorflow(tf keras) - 1.1x / 2.x
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = #GPU_ID from earlier

class ResNext:
  def __init__(self, classes, cardinality, repeat_num_list, input_shape):
    # image dimensions
    self.NUM_CLASSES = classes
    self.cardinality = cardinality
    self.repeat_num_list = repeat_num_list
    self.input_shape = input_shape

  ################################################
    """### Group convolution :"""
  ################################################

  def grouped_convolution(self, y, nb_channels, _strides):
    '''
    y - input tensor from previous layer
    nb_channels - original input value to channel before splitting to groups.
    cardinality - number of groups the channel is to be split  

    ResNext group convolution follows 'Split + Transform + Merge' strategy
    '''
  
    if nb_channels % self.cardinality!= 0:
      raise ValueError("The value of input_channels must be divisible by the value of groups.")
  
    div_group = nb_channels // self.cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(self.cardinality):
      group = tf.keras.layers.Lambda(lambda z: z[:, :, :, j * div_group:j * div_group + div_group])(y)
      branch = tf.keras.layers.Conv2D(div_group, kernel_size=(3, 3), strides=_strides, padding='same')(group)
      groups.append(branch)
            
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = tf.keras.layers.concatenate(groups)
    return y

  ################################################
  """ResNext Residual Block implementation :"""
  ################################################

  def resnext_residual_block(self, y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology and are subject to two simple rules:
      - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
      - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.

    - ResNext architecture is same as ResNet architecture .Only difference is that 'group convolutions' have been employed 
      inside identity blocks and conv blocks of ResNet
    - ResNext residual blocks is combination of 'Convolutional + Identity' blocks
    - ResNext residual block always have first block as convolutional followed by identical blocks (Eg : 3 blocks = 1conv + 2 identical)
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = tf.keras.layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)   # tf.keras.layers.LeakyReLU()(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = self.grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = tf.keras.layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
    # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
    # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
      shortcut = tf.keras.layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
      shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    y = tf.keras.layers.Add()([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = tf.keras.layers.ReLU()(y)

    return y

  ################################################
  """### ResNext Model :"""
  ################################################

  def ResNext_model(self):
    '''
    ResNext50 - repeat_num_list=[3, 4, 6, 3]
    ResNext101 - repeat_num_list=[3, 4, 23, 3]
    '''
    img_input = tf.keras.layers.Input(shape = self.input_shape)

    # conv1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(img_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = add_common_layers(x)

    # conv2
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(self.repeat_num_list[0]):
      project_shortcut = True if i == 0 else False
      x = self.resnext_residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(self.repeat_num_list[1]):
      # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
      strides = (2, 2) if i == 0 else (1, 1)
      x = self.resnext_residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(self.repeat_num_list[2]): #6
      strides = (2, 2) if i == 0 else (1, 1)
      x = self.resnext_residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(self.repeat_num_list[3]): #3
      strides = (2, 2) if i == 0 else (1, 1)
      x = self.resnext_residual_block(x, 1024, 2048, _strides=strides)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=self.NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
    # x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[img_input], outputs=[x])
    return model

if __name__=="__main__":
  
  # image dimensions
  img_height = 224
  img_width = 224
  img_channels = 3
  
  # network params
  cardinality = 32
  repeat_num_list =[3,4,6,3]
  input_shape = (img_height, img_width, img_channels)
  num_classes = 10

  # create Model
  network = ResNext(num_classes, cardinality, repeat_num_list, input_shape)
  model = network.ResNext_model()   #ResNext_(x,repeat_num_list=[3, 4, 6, 3]) for ResNext50
  print(model.summary())
