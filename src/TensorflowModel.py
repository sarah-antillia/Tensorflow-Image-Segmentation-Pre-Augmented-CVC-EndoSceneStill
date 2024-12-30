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
import datetime
import json

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# "false" -> "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Added the following lines.
SEED = 137
os.environ['PYTHONHASHSEED']         = "0"

#os.environ['TF_DETERMINISTIC_OPS']   = '1'
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print("=== os.environ['PYTHONHASHSEED']         {}".format(os.environ['PYTHONHASHSEED']))
# 
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

from tensorflow.python.framework import random_seed
#from EpochChangeCallback import EpochChangeCallback
#from GrayScaleImageWriter import GrayScaleImageWriter
from ImageMaskDataset import ImageMaskDataset
from BaseImageMaskDataset import BaseImageMaskDataset
from NormalizedImageMaskDataset import NormalizedImageMaskDataset

from ImageMaskDatasetGenerator import ImageMaskDatasetGenerator
from RGB2GrayscaleImageMaskDataset import RGB2GrayscaleImageMaskDataset


#from TensorflowImageMaskDataset import TensorflowImageMaskDataset

from SeedResetCallback       import SeedResetCallback
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss,  bce_dice_loss

from mish import mish
from LineGraph import LineGraph

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("=== GPU Name:", gpu.name, "  Type:", gpu.device_type)

#
# See https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 
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


class TensorflowModel:
  BEST_MODEL_FILE = "best_model.h5"
  HISTORY_JSON    = "history.json"

  def __init__(self, config_file):
    #self.set_seed()
    self.seed        = SEED
    self.config_file = config_file
    self.config    = ConfigParser(config_file)
    self.config.dump_all()

    image_height   = self.config.get(ConfigParser.MODEL, "image_height")
    image_width    = self.config.get(ConfigParser.MODEL, "image_width")
    image_channels = self.config.get(ConfigParser.MODEL, "image_channels")

    num_classes    = self.config.get(ConfigParser.MODEL, "num_classes")
   
    self.num_classes = num_classes
    self.tiledinfer_binarize =self.config.get(ConfigParser.TILEDINFER,   "binarize", dvalue=True) 
    print("--- tiledinfer binarize {}".format(self.tiledinfer_binarize))
    self.tiledinfer_threshold = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=60)

    base_filters   = self.config.get(ConfigParser.MODEL, "base_filters")
    num_layers     = self.config.get(ConfigParser.MODEL, "num_layers")
      
    activatation    = self.config.get(ConfigParser.MODEL, "activation", dvalue="relu")
    self.activation = eval(activatation)
    print("=== activation {}".format(activatation))
    
    # 2024/04/18
    self.mask_colors= self.config.get(ConfigParser.MASK, "mask_colors", dvalue=[])
    self.grayscaling = self.config.get(ConfigParser.MASK, "grayscaling", dvalue=None)
    
    self.create_gray_map()

    self.model     = self.create(num_classes, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)  
    learning_rate  = self.config.get(ConfigParser.MODEL, "learning_rate")
    clipvalue      = self.config.get(ConfigParser.MODEL, "clipvalue", 0.2)
    print("--- clipvalue {}".format(clipvalue))
  
    optimizer = self.config.get(ConfigParser.MODEL, "optimizer", dvalue="Adam")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
         beta_1=0.9, 
         beta_2=0.999, 
         clipvalue=clipvalue, 
         amsgrad=False)
      print("=== Optimizer Adam learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
    
    elif optimizer == "AdamW":
      self.optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate,
         clipvalue=clipvalue,
         )
      print("=== Optimizer AdamW learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
            
    self.model_loaded = False

    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy
  
    """ 
    Default loss and metrics functions if num_classes == 1
    """
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]


    # Read a loss function name from our config file, and eval it.
    self.loss  = eval(self.config.get(ConfigParser.MODEL, "loss"))
    # Read a list of metrics function names, and eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(ConfigParser.MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      try:
        m = eval(metric)
        self.metrics.append(m)    
      except:
        traceback.print_exc()

    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
       
    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
   
    show_summary = self.config.get(ConfigParser.MODEL, "show_summary")
    if show_summary:
      self.model.summary()
    self.show_history = self.config.get(ConfigParser.TRAIN, "show_history", dvalue=False)

  def create_gray_map(self,):
     self.gray_map = []
     if self.grayscaling !=None and len(self.mask_colors)>0:
       (IR, IG, IB) = self.grayscaling
       for color in self.mask_colors:
         (b, g, r) = color
         gray = int(IR* r + IG * g + IB * b)
         self.gray_map += [gray]

  def get_gray_map(self):
    return self.gray_map
  
  
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
  
    print("=== create")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    raise Exception("Please define your own Model class which inherits this TensorflowModel class")
  


  def inspect(self, image_file='./model.png', summary_file="./summary.txt"):
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    tf.keras.utils.plot_model(self.model, to_file=image_file, show_shapes=True)
    print("=== Saved model graph as an image_file {}".format(image_file))
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(summary_file, 'w') as f:
      # Pass the file handle in as a lambda function to make it callable
      self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("=== Saved model summary as a text_file {}".format(summary_file))

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
    
