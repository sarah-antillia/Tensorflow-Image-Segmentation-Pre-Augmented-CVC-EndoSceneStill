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

# 2024/05/04
# MaskColorizedWriter.py


import os
import cv2
import shutil
import numpy as np
import traceback
from ConfigParser import ConfigParser 

from PIL import Image, ImageOps

class MaskColorizedWriter:

  def __init__(self, config, verbose=True):
    print("=== MaskColorizedWriter ")

    self.config = config 
    self.verbose = verbose
    #ConfigParser(config_file)
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")

    self.mask_colors     = self.config.get(ConfigParser.MASK, "mask_colors")
    self.grayscaling     = self.config.get(ConfigParser.MASK, "grayscaling", dvalue=None)
    self.sharpening      = self.config.get(ConfigParser.MASK, "sharpening",  dvalue=False)
    if self.verbose:
      print("--- self.grayscaling {}".format(self.grayscaling))

    self.mask_channels   = self.config.get(ConfigParser.MASK, "mask_channels")
    self.masks_colors_order   = self.config.get(ConfigParser.MASK, "color_order")
    self.mask_colors     = self.config.get(ConfigParser.MASK, "mask_colors")
    self.colorized_output_format = self.config.get(ConfigParser.INFER, 
                                                   "colorized_output_format",
                                                   dvalue="rgb")    
    self.mask_colorize = self.config.get(ConfigParser.INFER, "mask_colorize", dvalue=False)

    self.colorized_dir = self.config.get(ConfigParser.INFER, "colorized_dir", dvalue=None)
    if self.mask_colorize and self.colorized_dir !=None:
      if os.path.exists(self.colorized_dir):
        shutil.rmtree(self.colorized_dir)
      if not os.path.exists(self.colorized_dir):
        os.makedirs(self.colorized_dir)
        
  def create_gray_map(self,):
     self.gray_map = []
     if self.verbose:
       print("---- create_gray_map {}".format(self.grayscaling))
        
     if self.grayscaling !=None and self.mask_colorize: 
       (IR, IG, IB) = self.grayscaling
       for color in self.mask_colors:
         (b, g, r) = color
         gray = int(IR* r + IG * g + IB * b)
         self.gray_map += [gray]
 
  def save_mask(self, image, size, basename, output_filepath):
    (w, h) = size
    self.mask_colorize = self.config.get(ConfigParser.SEGMENTATION, "colorize", dvalue=False)
    if self.mask_colorize:
      self.create_gray_map()

    # You will have to resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
    mask = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = mask*255
    mask = mask.astype(np.uint8) 

    gray_mask = mask    # This is used for merging input image with.
    if self.num_classes ==1:
        if self.sharpening:
          mask = self.sharpen(mask)
          cv2.imwrite(output_filepath, mask)
        else:
          if self.verbose:
            print("=== Inference for a single classes {} ".format(self.num_classes))
          cv2.imwrite(output_filepath, mask)
          if self.verbose:
            print("--- Saved {}".format(output_filepath))

        if self.mask_colorize:# and os.path.exists(self.colorized_dir):
          if self.verbose:
            print("--- colorizing the inferred mask ")
          mask = self.colorize_mask_one(mask, w, h)
          colorized_filepath = os.path.join(self.colorized_dir, basename)
          #2024/04/20 Experimental
          #mask = cv2.medianBlur(mask, 3)
          # colorrized_output_format = "bgr"

          if self.colorized_output_format == "bgr":
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
          cv2.imwrite(colorized_filepath, mask)
          if self.verbose:
            print("--- Saved {}".format(colorized_filepath))
    else:
        print("=== Inference in multi classes {} ".format(self.num_classes))
        print("----infered mask shape {}".format(image.shape))
        # The mask used in traiing     
    return gray_mask


  def colorize_mask_one(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    #condition = (mask[...] == gray) 
    condition = (mask[...] >= gray-10) & (mask[...] <= gray+10)   
    rgb_mask[condition] = [color]  
    return rgb_mask   
    
  def colorize_mask(self, img, w, h,):
      rgb_background = np.zeros((w, h, 3), np.uint8)
      for i in range(len(self.mask_colors)):
        color  = self.mask_colors[i]
        gray  = self.gray_map[i]
        rgb_mask = self.colorize_mask_one(img, color=color, gray=gray)
        rgb_background += rgb_mask
      rgb_background = cv2.resize(rgb_background, (w, h), interpolation=cv2.INTER_NEAREST)
      return rgb_background

  def sharpen(self, image):
    klist  =   [ [-1, -1, -1],[-1, 9, -1], [-1, -1, -1] ]
    kernel = np.array(klist, np.float32)
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

