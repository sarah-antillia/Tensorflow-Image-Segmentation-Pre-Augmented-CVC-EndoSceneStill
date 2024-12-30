# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowEfficientUNet.py
# 2024/03/30 to-arai

"""
This code is based on the following github web-site.
https://github.com/ahmed-470/Segmentation_EfficientNetB7_Unet/blob/main/efficientnetb7_Unet.py
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from TensorflowUNet import TensorflowUNet

print("TF Version: ", tf.__version__)


class TensorflowEfficientNetB7UNet(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)
    
    
  """Defining the Convolution Block"""
  def conv_block(self, input, num_filters):
      x = Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(input)
      x = BatchNormalization()(x)
      x = Activation("relu")(x)

      x = Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(x)
      x = BatchNormalization()(x)
      x = Activation("relu")(x)

      return x

  """Defining the Transpose Convolution Block"""
  def decoder_block(self, input, skip_features, num_filters):
      x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
      x = Concatenate()([x, skip_features])
      #x = Dropout(0.05)(x)
      x = self.conv_block(x, num_filters)
      return x

  """Building the EfficientNetB7_UNet"""
  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):            
      input_shape = (image_width, image_height, image_channels)
      
      
      """ Input """
      inputs = Input(shape=input_shape, name='input_image')
      #inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

      """ Pre-trained EfficientNetB7 Model """
      effNetB7 = tf.keras.applications.EfficientNetB7(input_tensor=inputs, include_top=False, 
                                                      weights="imagenet")

      for layer in effNetB7.layers[:-61]:
          layer.trainable = False
      for l in effNetB7.layers:
          print(l.name, l.trainable)

      """ Encoder """
      s1 = effNetB7.get_layer("input_image").output                   ## (512 x 512)
      s2 = effNetB7.get_layer("block1a_activation").output            ## (256 x 256)
      s3 = effNetB7.get_layer("block2a_activation").output            ## (128 x 128)
      s4 = effNetB7.get_layer("block3a_activation").output            ## (64 x 64)
      s5 = effNetB7.get_layer("block4a_activation").output            ## (32 x 32)

      """ Bridge """
      b1 = effNetB7.get_layer("block7a_activation").output  ## (16 x 16)

      """ Decoder """
      d1 = self.decoder_block(b1, s5, 512)                     ## (32 x 32)
      d2 = self.decoder_block(d1, s4, 256)                     ## (64 x 64)
      d3 = self.decoder_block(d2, s3, 128)                     ## (128 x 128)
      d4 = self.decoder_block(d3, s2, 64)                      ## (256 x 256)
      d5 = self.decoder_block(d4, s1, 32)                      ## (512 x 512)

      """ Output """
      activation = "softmax"
      if num_classes == 1:
        activation = "sigmoid"
        
      outputs = Conv2D(1, 1, padding="same", activation=activation)(d5)

      model = Model(inputs, outputs, name="EfficientNetB7_U-Net")
      return model

