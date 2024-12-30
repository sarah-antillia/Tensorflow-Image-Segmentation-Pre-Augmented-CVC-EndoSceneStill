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

# RGB2GrayscaleImageMaskDataset.py

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

class RGB2GrayscaleImageMaskDataset(BaseImageMaskDataset):

  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== RGB2GrayscaleImageMaskDataset.constructor")

    self.resize_interpolation = eval(self.config.get(ConfigParser.DATASET, "resize_interpolation", dvalue="cv2.INTER_NEAREST"))
    print("--- self.resize_interpolation {}".format(self.resize_interpolation))
    self.mask_colors_order = self.config.get(ConfigParser.MASK, "color_order", dvalue="rgb")
    self.mask_colors = self.config.get(ConfigParser.MASK, "mask_colors")
    # mask_colors_order = (b, g, r)
    # mask_colors = [(0, 0, 0), ( 0, 255, 0), (255, 0, 0), ( 0,  0, 255), (255, 255, 0), ( 0, 255, 255),]

    self.classes     = self.config.get(ConfigParser.MASK, "classes")
    self.grayscaling = self.config.get(ConfigParser.MASK, "grayscaling")
    print(self.mask_colors)
    print("--- grayscaling parameter {}".format(self.grayscaling))
    self.mask_len = len(self.mask_colors)
    self.sharpening = self.config.get(ConfigParser.DATASET, "sharpening", dvalue=True)

  def read_image_file(self, image_file):
    #print("=== read_image_file {}".format(image_file))
    self.read_mask_file(image_file)
    
    image = cv2.imread(image_file) 
    # image is bgr color-order
    if self.color_order == "rgb":
      # convert (B,G,R) -> (R,G,B) color-order
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)
    if self.image_normalize:
      image = image / 255.0
      image = image.astype(np.float32)
    
    """
    if self.image_format== "gray":
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if self.sharpening :
      klist = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
      
      skernel = np.array(klist, np.float32)

      image = cv2.filter2D(image,-1, skernel)
      image = image.astype(np.uint8)
    """
    return image
    

  def read_mask_file(self, mask_file):
    mask = cv2.imread(mask_file)
    if self.color_order == "rgb":
      # convert (B,G,R) -> (R,G,B) color-order
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask = cv2.resize(mask, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)
    # mask color_order is BGR
    h, w  = mask.shape[:2]
    # create blank mask of depth 1
    merged_grayscale_mask = np.zeros((w, h, 1), np.uint8)

    for color in self.mask_colors:
      grayscale_mask = self.create_categorized_grayscale_mask(mask, color)
      merged_grayscale_mask += grayscale_mask
    # Blur mask 
    if self.blur_mask:
      merged_grayscale_mask = cv2.blur(merged_grayscale_mask, self.blur_size)
      
    if  merged_grayscale_mask.ndim == 2:
        merged_grayscale_mask  = np.expand_dims( merged_grayscale_mask, axis=-1)
    return merged_grayscale_mask

  def create_categorized_grayscale_mask(self, mask, color):
    h, w = mask.shape[:2]
    # create a grayscale 1 channel black background 
    grayscale_mask = np.zeros((w, h, 1), np.uint8)

    # CCIR 601
    (IR, IG, IB) = self.grayscaling
    
    # bgr color-order
    (b, g, r) = color
    if self.mask_colors_order == "rgb":
      (r, b, g) = color
     
    condition = (mask[..., 0] == b) & (mask[..., 1] == g) & (mask[..., 2] == r)
    if self.color_order == "rgb":
      condition = (mask[..., 0] == r) & (mask[..., 1] == g) & (mask[..., 2] == b)

    gray = int(IR * r + IG * g + IB * b)


    # https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    # See also: 
    # https://en.wikipedia.org/wiki/Luma_(video)
    """
    For digital formats following CCIR 601 (i.e. most digital standard definition formats), 
    luma is calculated with this formula:
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    Formats following ITU-R Recommendation BT. 709 (i.e. most digital high definition formats)  
    use a different formula:
    gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    """
    # CCIR 601
    #(IR, IG, IB) = self.grayscaling
    # BT. 709
    #gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    grayscale_mask[condition] = [gray]
        
    return grayscale_mask


if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = RGB2GrayscaleImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=ConfigParser.TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=ConfigParser.EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

