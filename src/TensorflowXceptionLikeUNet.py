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

import os
import sys

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import shutil
import sys
import glob
import traceback
import random
import numpy as np
import cv2
import tensorflow as tf

#from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (add, Activation, Conv2D, Dropout, Conv2D, MaxPool2D, MaxPooling2D,
                                     SeparableConv2D, UpSampling2D, BatchNormalization)

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.losses import  BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ConfigParser import ConfigParser
#from EpochChangeCallback import EpochChangeCallback
#from GrayScaleImageWriter import GrayScaleImageWriter
#from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from TensorflowUNet import TensorflowUNet


class TensorflowXceptionLikeUNet(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)
    
  # The following create method is based on the following code in keras.io web-site.
  # https://keras.io/examples/vision/oxford_pets_image_segmentation/
  # This method is automatically called in the constructor of TensorflowUNet.
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("=== TensorflowXceptionLikeUNet.create ")
    print("--- base_filters {} ".format(base_filters))
    print("--- num_layers {} ".format(num_layers))
    dilation = self.config.get(ConfigParser.MODEL, "diation", dvalue=(1,1))

    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))

    filters_list = []
    for i in range(num_layers):
       num_filters = base_filters * (2**i)
       filters_list.append(num_filters )
    print("--- filters {}".format(filters_list))

    ### [First half of the network: downsampling inputs] ###
    # Entry block
    base_filters = filters_list[0]
    stride = 2

    x = Conv2D(base_filters, 3, strides=stride, dilation_rate=dilation, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    #for filters in [64, 128, 256]:
    for filters in filters_list[1:]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, dilation_rate=dilation, padding="same")(x)
        x = BatchNormalization()(x)
      
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, dilation_rate=dilation, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=stride, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=stride, dilation_rate=dilation, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    filters_list.reverse()
    #for filters in [256, 128, 64, 32]:
    print("--- filters {}".format(filters_list))
    for filters in filters_list:
           
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, dilation_rate=dilation, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    #outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    activation = "softmax"
    if num_classes == 1:
      activation = "sigmoid"
    outputs = Conv2D(num_classes, (3, 3), activation=activation, padding="same")(x)

    # Define the model
    model = Model(inputs=[inputs], outputs=[outputs], name="Xception-UNET")
    return model

    
if __name__ == "__main__":

  try:
    # Default config_file
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      cfile = sys.argv[1]
      if not os.path.exists(cfile):
         raise Exception("Not found " + cfile)
      else:
        config_file = cfile

    config   = ConfigParser(config_file)

    width    = config.get(ConfigParser.MODEL, "image_width")
    height   = config.get(ConfigParser.MODEL, "image_height")
    channels = config.get(ConfigParser.MODEL, "image_channels")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowXceptionLikeUNet(config_file)
 
  except:
    traceback.print_exc()
    
