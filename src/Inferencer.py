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
# Inferencer.py
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
from PIL import Image
import tensorflow as tf
from GrayScaleImageWriter import GrayScaleImageWriter
from MaskColorizedWriter import MaskColorizedWriter

from ConfigParser import ConfigParser

class Inferencer:

  def __init__(self, model, config_file, on_epoch_change=False):
    self.model = model
    self.on_epoch_change = on_epoch_change

    print("=== Inferencer.__init__ config_file {}".format(config_file))
    self.config = ConfigParser(config_file)
    self.num_classes = self.config.get(ConfigParser.MODEL, "num_classes")

    self.images_dir = self.config.get(ConfigParser.INFER, "images_dir")
    self.output_dir = self.config.get(ConfigParser.INFER, "output_dir")
    self.merged_dir = self.config.get(ConfigParser.INFER, "merged_dir")
    self.algorithm  = self.config.get(ConfigParser.INFER, "algorithm", dvalue=None)
    self.threshold  = self.config.get(ConfigParser.INFER, "threshold", dvalue=127)
    self.blur       = self.config.get(ConfigParser.INFER, "blur",      dvalue=False)
    self.ksize      = self.config.get(ConfigParser.INFER, "ksize",     dvalue= (5,5))
    # 2024/07/19
    self.color_converter  = self.config.get(ConfigParser.IMAGE, "color_converter", dvalue=None)
    if self.color_converter != None:
      self.color_converter  = eval(self.color_converter)
    #self.gamma_correction     =
    self.gamma  = self.config.get(ConfigParser.IMAGE, "gamma", dvalue=0)
    self.sharpen_k  = self.config.get(ConfigParser.IMAGE, "sharpening", dvalue=0)

    if self.on_epoch_change:
      self.output_dir    = self.config.get(ConfigParser.TRAIN, "epoch_change_infer_dir", dvalue="./epoch_change_infer")
    self.num_infer_images = self.config.get(ConfigParser.TRAIN, "num_infer_images", dvalue=1)

    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)

    self.colorize = self.config.get(ConfigParser.SEGMENTATION, "colorize", dvalue=False)
    self.black    = self.config.get(ConfigParser.SEGMENTATION, "black",    dvalue="black")
    self.white    = self.config.get(ConfigParser.SEGMENTATION, "white",    dvalue="white")
    self.blursize = self.config.get(ConfigParser.SEGMENTATION, "blursize", dvalue=None)
    self.bgr2hls   = True

    verbose       = not self.on_epoch_change
    self.writer   = GrayScaleImageWriter(colorize=self.colorize, black=self.black, 
                                         white=self.white, verbose=verbose)

    self.maskcolorizer = MaskColorizedWriter(self.config, verbose=verbose)
    self.mask_colorize = self.config.get(ConfigParser.INFER, "mask_colorize", dvalue=False)

    self.color_order = self.config.get(ConfigParser.DATASET,   "color_order", dvalue="rgb")
   
    self.image_files  = glob.glob(self.images_dir + "/*.png")
    self.image_files += glob.glob(self.images_dir + "/*.jpg")
    self.image_files += glob.glob(self.images_dir + "/*.tif")
    self.image_files += glob.glob(self.images_dir + "/*.bmp")
    self.width        = self.config.get(ConfigParser.MODEL, "image_width")
    self.height       = self.config.get(ConfigParser.MODEL, "image_height")
  
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")
  
    if self.on_epoch_change:
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
    #2024/06/27 Added the following line.
    if self.merged_dir !=None:
      if self.merged_dir and os.path.exists(self.merged_dir):
        shutil.rmtree(self.merged_dir)
      if not os.path.exists(self.merged_dir):
        os.makedirs(self.merged_dir)

  def gamma_correction(self, img, gamma):
    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

 # 2024/09/20
  def sharpen(self, img, k):
    if k > 0:
      kernel = np.array([[-k, -k, -k], 
                       [-k, 1+8*k, -k], 
                       [-k, -k, -k]])
      img = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img
  
  def infer(self, epoch=None):
    if self.on_epoch_change == False:
      print("=== Inferencer.infer start")
    expand     = True

    for image_file in self.image_files:

      basename = os.path.basename(image_file)
      # 2024/06/13 Modified to use os.path.splitext
      #name     = basename.split(".")[0]    
      name, ext = os.path.splitext(basename)
      img      = cv2.imread(image_file)
      # convert (B,G,R) -> (R,G,B) color-order
      # 2024/04/20
      if self.color_order == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
      # 2024/07/18
      if self.gamma > 0:
        img = self.gamma_correction(img, self.gamma)

      # 2024/09/20
      if self.sharpen_k > 0:
        img = self.sharpen(img, self.sharpen_k)
      if self.color_converter:
        img = cv2.cvtColor(img, self.color_converter) # cv2.COLOR_BGR2HLS)

      h, w = img.shape[:2]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
       
      # filename take the following format, which takes a prefix as an epoch number
      # filename without file extension 
      filename = name
      if self.on_epoch_change:
        filename = "Epoch_" +str(epoch+1) + "_" + name
      output_filepath = os.path.join(self.output_dir, filename)
      # 2024/06/22
      if self.blur:
          image = cv2.GaussianBlur(image, ksize=self.ksize, sigmaX=0)

      if self.mask_colorize:
        # MaskColorizer
        #print("==== using MaskColorizeWriter")
        gray_mask = self.maskcolorizer.save_mask(image, (w, h), filename, output_filepath)
      else:
        #Use GrayScaleWriter
        #print("---- using GrayScaleImageWrite")
        gray_mask = self.writer.save_resized(image, (w, h), self.output_dir, filename)

      #if self.merged_dir !=None:
      # 2024/06/20
      # Don't save merged_image if self.on_epoch_change==True
      if self.merged_dir !=None and self.on_epoch_change ==False:
  
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img   = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        #if blursize:
        #  img   = cv2.blur(img, blursize)
        #img = cv2.medianBlur(img, 3)
        if gray_mask.ndim ==2:
          gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        img += gray_mask
        merged_file = os.path.join(self.merged_dir, basename)
        if self.on_epoch_change == False:
          print("=== Saved {}".format(merged_file))
        cv2.imwrite(merged_file, img)

  def predict(self, images, expand=True):
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    

  def cv2pil(self, image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
  
  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

  """
  def sharpen(self, image):
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel) 
    return sharpened
  """
  
  def mask_to_image(self, data, factor=255.0, format="RGB"):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    image = image.convert(format)
    return image
  
  def normalize_mask(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    return data

  def binarize(self, mask):
    if self.num_classes == 1:
 
      if self.tiledinfer_binarize:

        if  self.algorithm == cv2.THRESH_TRIANGLE or self.algorithm == cv2.THRESH_OTSU: 
          _, mask = cv2.threshold(mask, 0, 255, self.algorithm)
        if  self.algorithm == cv2.THRESH_BINARY or self.algorithm ==  cv2.THRESH_TRUNC: 
          #_, mask = cv2.threshold(mask, 127, 255, self.algorithm)
          _, mask = cv2.threshold(mask, self.threshold, 255, self.algorithm)
        elif self.algorithm == None:
          mask[mask< self.threshold] =   0
          mask[mask>=self.threshold] = 255    
      pass
    return mask     



