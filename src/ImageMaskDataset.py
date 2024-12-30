# Copyright 2023-2024 antillia.com Toshiyuki Arai
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

# ImageMaskDataset.py
# 2023/05/31 to-arai Modified to use config_file
# 2023/10/02 Updated to call self.read_image_file, and self.read_mask_file in create nethod.

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback
import tensorflow as tf

from ConfigParser import ConfigParser
from BaseImageMaskDataset import BaseImageMaskDataset

class ImageMaskDataset(BaseImageMaskDataset):

  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== ImageMaskDataset.constructor")

    self.resize_interpolation = eval(self.config.get(ConfigParser.DATASET, "resize_interpolation", dvalue="cv2.INTER_NEAREST"))
    print("--- self.resize_interpolation {}".format(self.resize_interpolation))

  def read_image_file(self, image_file):
    image = cv2.imread(image_file) 
    if self.gamma >0:
      #print("---- image gamma_correction{}".format(self.gamma))
      image = self.gamma_correction(image, self.gamma)

    if self.sharpening_k >0:
      #print("---- image sharpening {}".format(self.sharpening_k))
      image = self.sharpen(image, self.sharpening_k)

    if self.color_converter !=None:
      #print("---- image color_convter {}".format(self.color_converter))
      image = cv2.cvtColor(image, self.color_converter)

    image = cv2.resize(image, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)
    if self.image_normalize:
      image = image / 255.0
      image = image.astype(np.float32)
    return image


  def create_one_class_mask(self, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if self.binarize:
      if  self.algorithm == cv2.THRESH_TRIANGLE or self.algorithm == cv2.THRESH_OTSU: 
        _, mask = cv2.threshold(mask, 0, 255, self.algorithm)
      if  self.algorithm == cv2.THRESH_BINARY or self.algorithm ==  cv2.THRESH_TRUNC: 
        #_, mask = cv2.threshold(mask, 127, 255, self.algorithm)
        _, mask = cv2.threshold(mask, self.threshold, 255, self.algorithm)
      elif self.algorithm == None:
        mask[mask< self.threshold] =   0
        mask[mask>=self.threshold] = 255
    # Blur mask 
    if self.blur_mask:
      mask = cv2.blur(mask, self.blur_size)

    if mask.ndim == 2:
       mask  = np.expand_dims(mask, axis=-1)
    return mask
    
  def read_mask_file(self, mask_file):
    mask = cv2.imread(mask_file) 
    mask = cv2.resize(mask, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)
    
    if self.num_classes == 1:
      return self.create_one_class_mask(mask)
    else:
      return self.create_multi_class_mask(mask)
  
  def create_multi_class_mask(self, mask):
    l =  len(self.mask_colors)
    if l == 0:
      raise Exception("Invali mask_colors parameter in train_eval_inf.config") 
    categorized_rgb_masks = []
    for color in self.mask_colors:
      rgb_mask = self.create_categorized_rgb_mask(mask, color)
      categorized_rgb_masks +=[rgb_mask]
    categorized_rgb_masks = np.array(categorized_rgb_masks)
    #print("--- categorized_rgb_masks {}".format(categorized_rgb_masks.shape))
    return categorized_rgb_masks
  
  def create_categorized_rgb_mask(self, mask, color):
    (h, w, c) = (0, 0, 0)
   
    if len(mask.shape) == 3:
      h, w, c = mask.shape[:3]
    if c >1:
      ch = 3
    # create RGB 3 channel black background 
    back = np.zeros((w, h, ch), np.uint8)
    # mask_format = "bgr"
    (b, g, r) = color
    if self.mask_format == "rgb":
      (r, b, g) = color
    condition = (mask[..., 0] == b) & (mask[..., 1] == g) & (mask[..., 2] == r)
    back[condition] = [b, g, r]
    return back

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = ImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=ConfigParser.TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=ConfigParser.EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

