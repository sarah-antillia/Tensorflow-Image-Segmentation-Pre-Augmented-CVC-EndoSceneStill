# Copyright 2023 antillia.com Toshiyuki Arai
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

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

# 2023/06/29 Updated create method to add BatchNormalization provied that 
#[model]
#normalization=True
# However, this True setting will not be recommended because this may have adverse effect
# on tiled_image_segmentation.

# 2023/07/01 Support Overlapped-Tiled-Image-Segmentation 
#[tiledinfer]
#overlapping=32
#Specify a pixel size to overlap-tiling.
#Specify 0 if you need no overlapping.

# 2023/11/01
# Remove set_seed method from TensorflowUNet class.

# 2023/11/01
# Added dropout_seed_fixing flag to [model] section
""" 
[model]
; 2023/11/01 Fixing a random-seed in Dropout layer
dropout_seed_fixing = True

if dropout_seed_fixing:
    u = Dropout(dropout_rate * f, seed=self.seed)(u)
"""

# 2023/11/01
# Added seedreset_callbacck flag to [train] section.
"""
; Experimental: Enable the random-seed-reset-callback if Ture.
; This will affect the behavior of Dropout layer of your CNN model.
seedreset_callback = True
"""

# 2023/11/01
# Added dataset_splitter flag to [train] section.
#; Enable splitting dataset into train and valid if True.
#dataset_splitter = True
"""
#; Enable splitting dataset into train and valid if True.
[train]
#dataset_splitter = True

# 2023/10/27
dataset_splitter = self.config.get(ConfigParser.TRAIN, "dataset_splitter", dvalue=False) 

"""

# 2024/03/28
"""
Added 'plot_line_graphs' method to <a href="./src/TensorflowUNet.py">TensorflowUNet</a> class 
to plot line_graphs for <i>train_eval.csv</i> and <i>train_losses.csv</i> generated through the training-process.</li>
"""

import os
import sys
import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 2023/10/20 "false" -> "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2023/10/13: Added the following lines.
SEED = 137
os.environ['PYTHONHASHSEED']         = "0"

#os.environ['TF_DETERMINISTIC_OPS']   = '1'
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print("=== os.environ['PYTHONHASHSEED']         {}".format(os.environ['PYTHONHASHSEED']))
# 2024/01/29
#print("=== os.environ['TF_DETERMINISTIC_OPS']   {}".format(os.environ['TF_DETERMINISTIC_OPS']))
#print("=== os.environ['TF_CUDNN_DETERMINISTIC'] {}".format(os.environ['TF_CUDNN_DETERMINISTIC']))

import shutil

import sys
import glob
import traceback
import random
import numpy as np
import cv2
from ConfigParser import ConfigParser

import tensorflow as tf
print("====== Tensorflow Version: {} ====== ".format(tf.version.VERSION))

tf.compat.v1.disable_eager_execution()

from PIL import Image, ImageFilter, ImageOps
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.losses import  BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 2023/10/20
from tensorflow.python.framework import random_seed
from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

from SeedResetCallback       import SeedResetCallback
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss,  bce_dice_loss

from mish import mish
from LineGraph import LineGraph

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("=== GPU Name:", gpu.name, "  Type:", gpu.device_type)

# 2023/10/31
# See https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 2023/10/23
random.seed    = SEED
print("=== random.seed {}".format(SEED))

np.random.seed = SEED
print("=== numpy.random.seed {}".format(SEED))
tf.random.set_seed(SEED)
print("=== tf.random.set_seed({})".format(SEED))

# See https://www.tensorflow.org/community/contribute/tests
# Always seed any source of stochasticity
random_seed.set_seed(SEED)
print("=== tensorflow.python.framework random_seed({})".format(SEED))

# Disable OpenCL and disable multi-threading.
#cv2.ocl.setUseOpenCL(False)
#cv2.setNumThreads(1)
cv2.setRNGSeed(SEED)
print("=== cv2.setRNGSeed ({})".format(SEED))


from TensorflowModel import TensorflowModel

class TensorflowUNet(TensorflowModel):

  def __init__(self, config_file):
    super().__init__(config_file)
  
  
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    print("=== create")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = tf.keras.layers.Input((image_height, image_width, image_channels))
    input_normalize = self.config.get(ConfigParser.MODEL, "input_normalize", dvalue=True)
    print("--- input_normalize {}".format(input_normalize))
    if input_normalize:
      s= tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    else:
      s = inputs

    # normalization is False on default.
    normalization = self.config.get(ConfigParser.MODEL, "normalization", dvalue=False)
    print("--- normalization {}".format(normalization))
    # fixing_dropout_seed is False on default.
    dropout_seed_fixing = self.config.get(ConfigParser.MODEL, "dropout_seed_fixing", dvalue=False)
    print("--- dropout_seed_fixing {}".format(dropout_seed_fixing))

    # Encoder
    dropout_rate = self.config.get(ConfigParser.MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3)
    pool_size   = (2, 2)
    dilation    = (2, 2)
    strides     = (1, 1)
    # [model] 
    # Specify a tuple of base kernel size of odd number something like this: 
    # base_kernels = (5,5)
    base_kernels   = self.config.get(ConfigParser.MODEL, "base_kernels", dvalue=(3,3))
    (k, k) = base_kernels
    kernel_sizes = []
    for n in range(num_layers):
      kernel_sizes += [(k, k)]
      k -= 2
      if k <3:
        k = 3
    rkernel_sizes =  kernel_sizes[::-1]
    rkernel_sizes = rkernel_sizes[1:] 
    
    # kernel_sizes will become a list [(7,7),(5,5), (3,3),(3,3)...] if base_kernels were (7,7)
    print("--- kernel_size   {}".format(kernel_sizes))
    print("--- rkernel_size  {}".format(rkernel_sizes))
    # </experiment>
    dilation = None
    try:
      dilation_ = self.config.get(ConfigParser.MODEL, "dilation", (1, 1))
      (d1, d2) = dilation_
      if d1 == d2:
        dilation = dilation_
    except:
      traceback.print_exc()

    dilations = []
    (d, d) = dilation
    for n in range(num_layers):
      dilations += [(d, d)]
      d -= 1
      if d <1:
        d = 1
    rdilations = dilations[::-1]
    rdilations = rdilations[1:]
    print("=== dilations  {}".format(dilations))
    print("=== rdilations {}".format(rdilations))

    for i in range(num_layers):
      filters = base_filters * (2**i)
      kernel_size = kernel_sizes[i] 
      dilation = dilations[i]
      print("--- kernel_size {}".format(kernel_size))
      print("--- dilation {}".format(dilation))
      
      c = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(s)
      if normalization:
        c = tf.keras.layers.BatchNormalization()(c) 
      if dropout_seed_fixing:
        c = tf.keras.layers.Dropout(dropout_rate * i, seed= self.seed)(c)
      else:
        c = tf.keras.layers.Dropout(dropout_rate * i)(c)
      c = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      if normalization:
        c = tf.keras.layers.BatchNormalization()(c) 
      if i < (num_layers-1):
        p = tf.keras.layers.MaxPool2D(pool_size=pool_size)(c)
        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()
    n = 0
    c = enc[n]
    
    # --- Decoder
    for i in range(num_layers-1):
      kernel_size = rkernel_sizes[i] 
      dilation = rdilations[i]
      print("+++ kernel_size {}".format(kernel_size))
      print("+++ dilation {}".format(dilation))

      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c)
      n += 1
      u = tf.keras.layers.concatenate([u, enc[n]])
      u = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      if normalization:
        u = tf.keras.layers.BatchNormalization()(u)
      if dropout_seed_fixing:
        u = tf.keras.layers.Dropout(dropout_rate * f, seed=self.seed)(u)
      else:
        u = tf.keras.layers.Dropout(dropout_rate * f)(u)

      u = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      if normalization:
        u = tf.keras.layers.BatchNormalization()(u) 
      c  = u

    # outouts
    # 2024/03/28
    final_activation = self.config.get(ConfigParser.MODEL, "final_activation", dvalue="sigmoid")
    activation ="sigmoid"
    if final_activation:
      activation = final_activation

    print("--- final activation {}".format(activation))
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation)(c)

    # create Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
  
if __name__ == "__main__":
  try:
    # Default config_file
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      config_file= sys.argv[1]
      if not os.path.exists(config_file):
         raise Exception("Not found " + config_file)
    config   = ConfigParser(config_file)
    width    = config.get(ConfigParser.MODEL, "image_width")
    height   = config.get(ConfigParser.MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowUNet(config_file)
 
  except:
    traceback.print_exc()
    
