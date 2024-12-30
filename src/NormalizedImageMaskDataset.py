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

# TensorflowImageMaskDataset.py

# This code is based on the python script in the following web-site
# https://github.com/nikhilroxtomar/Human-Face-Segmentation-in-TensorFlow
# https://kiansoon.medium.com/semantic-segmentation-is-the-task-of-partitioning-an-image-into-multiple-segments-based-on-the-356a5582370e
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
# https://www.tensorflow.org/tutorials/images/segmentation
# https://keras.io/examples/vision/deeplabv3_plus/


import os
import cv2
import glob
import numpy as np
import tensorflow as tf

import traceback
from ConfigParser import ConfigParser
from BaseImageMaskDataset import BaseImageMaskDataset
from PIL import Image

class NormalizedImageMaskDataset(BaseImageMaskDataset):
 
  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== {} constructor".format(self.__class__))


    self.mask_colors = self.config.get(ConfigParser.MASK, "mask_colors")
    print("--- mask_colors {}".format(self.mask_colors))
    self.normaized_threshold = self.config.get(ConfigParser.MASK, "normalized_threshold", dvalue=0.15)
     
  def read_image_file(self, x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (self.image_width, self.image_height))
    if np.max(x) >0:
      x = x/255.0
      x = x.astype(np.float32)
    return x

  def read_mask_file(self, y):
    y = cv2.imread(y, cv2.IMREAD_COLOR)
    y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

    y = cv2.resize(y, (self.image_width, self.image_height))
    if np.max(y) > 1:
      y = y/255
      y[y >= self.normaized_threshold ] = 1 
      y[y <  self.normaized_threshold ] = 0
   
    #y = y.astype(np.int32)
    if len(y.shape) == 2:
      y = np.expand_dims(y, axis=-1)
  
    return y

 

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = NormalizedImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=ConfigParser.TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=ConfigParser.EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

