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
# TiledInferencer.py
# 2024/06/01:  

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import numpy as np

import shutil
import sys
import cv2
import glob
import traceback
import tensorflow as tf
from GrayScaleImageWriter import GrayScaleImageWriter
from MaskColorizedWriter import MaskColorizedWriter

from PIL import Image, ImageOps
from ConfigParser import ConfigParser
from Inferencer import Inferencer

class TiledInferencer(Inferencer):

  def __init__(self, model, config_file, on_epoch_change=False):
    self.model = model
    self.on_epoch_change = on_epoch_change

    print("=== TiledInferencer.__init__ config?file {}".format(config_file))
    self.config = ConfigParser(config_file)
    self.num_classes = self.config.get(ConfigParser.MODEL, "num_classes")

    self.images_dir = self.config.get(ConfigParser.TILEDINFER, "images_dir")
    self.output_dir = self.config.get(ConfigParser.TILEDINFER, "output_dir")
    self.merged_dir = self.config.get(ConfigParser.TILEDINFER, "merged_dir")
    self.algorithm  = self.config.get(ConfigParser.TILEDINFER, "algorithm", dvalue=None)
    self.threshold  = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=127)

    # 2024/07/19
    self.color_converter  = self.config.get(ConfigParser.IMAGE, "color_converter", dvalue=None)
    if self.color_converter != None:
      self.color_converter  = eval(self.color_converter)
    # 2024/08/20 
    self.gamma  = self.config.get(ConfigParser.IMAGE, "gamma", dvalue=0)
    self.sharpen_k  = self.config.get(ConfigParser.IMAGE, "sharpening", dvalue=0)
    
    if self.on_epoch_change:
       self.output_dir  = self.config.get(ConfigParser.TRAIN, "epoch_change_tiledinfer_dir", 
                                          dvalue="./epoch_change_tiledinfer")
 
    self.num_infer_images = self.config.get(ConfigParser.TRAIN, "num_infer_images", dvalue=1)


    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)

    self.colorize = self.config.get(ConfigParser.SEGMENTATION, "colorize", dvalue=False)
    self.black    = self.config.get(ConfigParser.SEGMENTATION, "black",    dvalue="black")
    self.white    = self.config.get(ConfigParser.SEGMENTATION, "white",    dvalue="white")
    self.blursize = self.config.get(ConfigParser.SEGMENTATION, "blursize", dvalue=None)
    verbose       = not  self.on_epoch_change 
    self.writer   = GrayScaleImageWriter(colorize=self.colorize, 
                                         black=self.black, white=self.white, verbose=verbose)
    verbose = True
    self.maskcolorizer = MaskColorizedWriter(self.config, verbose=verbose)
    # 2024/06/15 INFER ->TILEDINFER
    #self.mask_colorize = self.config.get(ConfigParser.TILEDINFER, "mask_colorize", dvalue=False)

    self.MARGIN       = self.config.get(ConfigParser.TILEDINFER, "overlapping", dvalue=0)
    self.bitwise_blending    = self.config.get(ConfigParser.TILEDINFER, "bitwise_blending", dvalue=True)
    self.tiledinfer_binarize  = self.config.get(ConfigParser.TILEDINFER, "binarize", dvalue=False)
    self.tiledinfer_threshold = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=127)

    self.bgcolor = self.config.get(ConfigParser.TILEDINFER, "background", dvalue=0)  

    self.color_order = self.config.get(ConfigParser.DATASET,   "color_order", dvalue="rgb")
   
    self.image_files  = glob.glob(self.images_dir + "/*.png")
    self.image_files += glob.glob(self.images_dir + "/*.jpg")
    self.image_files += glob.glob(self.images_dir + "/*.tif")
    self.image_files += glob.glob(self.images_dir + "/*.bmp")
    self.width        = self.config.get(ConfigParser.MODEL, "image_width")
    self.height       = self.config.get(ConfigParser.MODEL, "image_height")
    self.split_size   = self.config.get(ConfigParser.TILEDINFER, "split_size", dvalue=self.width)
    #print("---split_size {}".format(self.split_size))
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")

    if on_epoch_change:
      num_images = len(self.image_files)
      if self.num_infer_images > num_images:
        self.num_infer_images =  num_images
      if self.num_infer_images < 1:
        self.num_infer_images =  1
      self.image_files = self.image_files[:self.num_infer_images]

    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    #2024/06/20
    if self.merged_dir !=None:
      if os.path.exists(self.merged_dir):
        shutil.rmtree(self.merged_dir)
      if not os.path.exists(self.merged_dir):
        os.makedirs(self.merged_dir)

  def infer(self, epoch =None):
    if self.on_epoch_change == False:
      print("=== TiledInferencer.infer ")
            
    output_dir = self.output_dir
    expand     = True

    for image_file in self.image_files:
      image   = Image.open(image_file)
      #PIL image color_order = "rgb"
      w, h    = image.size

      # Resize the image to the input size (width, height) of our UNet model.      
      resized = image.resize((self.width, self.height))

      # Make a prediction to the whole image not tiled image of the image_file 
      cv_image= self.pil2cv(resized)
      predictions = self.predict([cv_image], expand=expand)
          
      prediction  = predictions[0]
      whole_mask  = prediction[0]    

      #whole_mask_pil = self.mask_to_image(whole_mask)
      #whole_mask  = self.pil2cv(whole_mask_pil)
      whole_mask  = self.normalize_mask(whole_mask)
      # 2024/03/30
      whole_mask  = self.binarize(whole_mask)
      whole_mask  = cv2.resize(whole_mask, (w, h), interpolation=cv2.INTER_NEAREST)     
      basename = os.path.basename(image_file)
      w, h  = image.size

      vert_split_num  = h // self.split_size
      if h % self.split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // self.split_size
      if w % self.split_size != 0:
        horiz_split_num += 1
      background = Image.new("L", (w, h), self.bgcolor)

      # Tiled image segmentation
      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = self.split_size * i
          upper = self.split_size * j
          right = left  + self.split_size
          lower = upper + self.split_size

          if left >=w or upper >=h:
            continue 
      
          left_margin  = self.MARGIN
          upper_margin = self.MARGIN
          if left-self.MARGIN <0:
            left_margin = 0
          if upper-self.MARGIN <0:
            upper_margin = 0

          right_margin = self.MARGIN
          lower_margin = self.MARGIN 
          if right + right_margin > w:
            right_margin = 0
          if lower + lower_margin > h:
            lower_margin = 0

          cropbox = (left  - left_margin,  upper - upper_margin, 
                     right + right_margin, lower + lower_margin )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          # Get the size of the cropped image.
          cw, ch  = cropped.size

          # Resize the cropped image to the model image size (width, height) for a prediction.
          cropped = cropped.resize((self.width, self.height))

          cvimage  = self.pil2cv(cropped)
          # 2024/08/20
          if self.gamma > 0:
            cvimage = self.gamma_correction(cvimage, self.gamma)
         # 2024/09/20
          if self.sharpen_k > 0:
            cvimage = self.sharpen(cvimage, self.sharpen_k)
          # 2024/07/18
          if self.color_converter:
            cvimage = cv2.cvtColor(cvimage, self.color_converter) # cv2.COLOR_BGR2HLS)

          predictions = self.predict([cvimage], expand=expand)
          
          prediction  = predictions[0]
          mask        = prediction[0]    
          mask        = self.mask_to_image(mask)
          # Resize the mask to the same size of the corresponding the cropped_size (cw, ch)
          mask        = mask.resize((cw, ch))

          right_position = left_margin + self.width
          if right_position > cw:
             right_position = cw

          bottom_position = upper_margin + self.height
          if bottom_position > ch:
             bottom_position = ch

          # Excluding margins of left, upper, right and bottom from the mask. 
          mask         = mask.crop((left_margin, upper_margin, 
                                  right_position, bottom_position)) 
          #iw, ih = mask.size
          background.paste(mask, (left, upper))

      basename = os.path.basename(image_file)
      #name, ext = os.path.splitext(basename)
      filename = basename

      if self.on_epoch_change:
        filename = "Epoch_" + str(epoch+1) + "_" + basename
      output_filepath = os.path.join(output_dir, filename)
      cv_background = self.pil2cv(background)
         
      if self.bitwise_blending:
        #print("--- TiledInferencer bitwise_blending")
        # Blend the non-tiled whole_mask and the tiled-background
        #cv_background = cv2.cvtColor(cv_background, cv2.COLOR_BGR2GRAY)
        cv_background = cv2.bitwise_and(whole_mask, cv_background)

        cv_background = self.binarize(cv_background)
      if self.tiledinfer_binarize:
        #print("---non blending save binarize ")
        cv_background = self.binarize(cv_background)

      if self.colorize:
        #print("--- TiledInferencer colorrize")  
        background = self.cv2pil(cv_background)
        if background.mode != "L":
          background = image.convert("L")
        background = ImageOps.colorize(background, black=self.black, white=self.white)
        cv_background = self.pil2cv(background)
  
      cv2.imwrite(output_filepath, cv_background)

      if self.on_epoch_change == False:
        print("=== Saved outputfile {}".format(output_filepath))

      #if self.merged_dir !=None:
      # 2024/06/20
      # Don't save merged_image if self.on_epoch_change==True
      if self.merged_dir !=None and self.on_epoch_change ==False:

        img   = np.array(image)
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #2024/03/10
        mask  = cv_background
        if mask.ndim ==2:  
          mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img += mask
        merged_file = os.path.join(self.merged_dir, basename)
        if self.on_epoch_change == False:
          print("=== Saved outputfile {}".format(merged_file))
        cv2.imwrite(merged_file, img)     
