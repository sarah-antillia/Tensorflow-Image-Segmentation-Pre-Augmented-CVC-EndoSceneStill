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

# ImageMaskDataset.py
# 2023/05/31 to-arai Modified to use config_file
# 2023/10/02 Updated to call self.read_image_file, and self.read_mask_file in create nethod.

import os
import numpy as np
import cv2
import shutil

from tensorflow.python.keras.utils import np_utils
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label

from skimage.io import imread
import traceback
from ConfigParser import ConfigParser

"""
MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"
TEST   = "test"
MASK   = "mask"
IMAGE  = "image"
"""

class BaseImageMaskDataset:

  def __init__(self, config_file):
    print("=== BaseImageMaskDataset.constructor")

    self.config         = ConfigParser(config_file)
    self.image_width    = self.config.get(ConfigParser.MODEL, "image_width")
    self.image_height   = self.config.get(ConfigParser.MODEL, "image_height")
    self.image_channels = self.config.get(ConfigParser.MODEL, "image_channels")
    self.num_classes    = self.config.get(ConfigParser.MODEL, "num_classes")

    self.algorithm      = self.config.get(ConfigParser.MASK,  "algorithm", dvalue=None)
    self.batch_size     = self.config.get(ConfigParser.TRAIN, "batch_size", dvalue=2)
    self.threshold      = self.config.get(ConfigParser.MASK, "threshold")
    self.blur_mask      = self.config.get(ConfigParser.MASK, "blur")
  
    self.blur_size      = self.config.get(ConfigParser.MASK,  "blur_size", dvalue=(3,3))

    self.color_converter= self.config.get(ConfigParser.IMAGE,  "color_converter", dvalue=None)
    if self.color_converter != None:
      self.color_converter  = eval(self.color_converter)
    # image_format may take "rgb" which is default, or "grayscale" 
    self.gamma= self.config.get(ConfigParser.IMAGE, "gamma", dvalue=0)

    self.image_format   = self.config.get(ConfigParser.DATASET, "image_format", dvalue="rgb")

    self.input_normalize= self.config.get(ConfigParser.DATASET, "input_normalize", dvalue=True)
    self.debug          = self.config.get(ConfigParser.DATASET, "debug", dvalue=True)
    self.rgb_mask       = self.config.get(ConfigParser.DATASET, "rgb_mask", dvalue=False)
    # 2024/04/20
    self.color_order    = self.config.get(ConfigParser.DATASET, "color_order", dvalue="bgr")
    #self.sharpening_k   = self.config.get(ConfigParser.DATASET, "sharpening",   dvalue=False)
    # 2024/09/19 Added the following line
    self.sharpening_k  = self.config.get(ConfigParser.IMAGE, "sharpening", dvalue=0)
  
    self.debug_images_dir = "./dataset_images/"
    self.debug_masks_dir  = "./dataset_masks/"
    self.mask_format      = self.config.get(ConfigParser.DATASET, "mask_format", dvalue="gray")

    if os.path.exists(self.debug_images_dir):
      shutil.rmtree(self.debug_images_dir)
    if not os.path.exists(self.debug_images_dir):
      os.makedirs(self.debug_images_dir)

    if os.path.exists(self.debug_masks_dir):
      shutil.rmtree(self.debug_masks_dir)
    if not os.path.exists(self.debug_masks_dir):
      os.makedirs(self.debug_masks_dir)
  
    # Convert white-black mask from gray-scale mask if binarize is True
    self.binarize       = self.config.get(ConfigParser.MASK,  "binarize", dvalue=False)
    
    # Convert gray-scale mask from rgb-mask if grayscaling is True

    self.grayscaling    = self.config.get(ConfigParser.MASK,  "grayscaling", dvalue=True)

    self.image_normalize= self.config.get(ConfigParser.DATASET, "image_normalize", dvalue=False)
    self.debug          = self.config.get(ConfigParser.DATASET, "debug",     dvalue=False)

    self.image_dtype    = np.int8
    self.mask_dtype     = bool
    #
    if self.image_normalize:
      # Color Image (R,G,B) in range 0~255 to be converted in range 0~1.0
      self.image_dtype = np.float32

  
    self.mask_dtype  = np.int32   

    self.mask_colors   = self.config.get(ConfigParser.MASK,  "mask_colors", dvalue=None)
    print("--- mask_colors {}".format(self.mask_colors))
    print("--- num_classes {}".format(self.num_classes))

    print("--- image_normalize {}".format(self.image_normalize))
    print("--- binarize algorithm {}".format(self.algorithm))

    if self.algorithm !=None:
      self.algorithm = eval(self.algorithm)

 
  # If needed, please override this method in a subclass derived from this class.
  def create(self, dataset = ConfigParser.TRAIN,  debug=False):
    if not dataset in [ConfigParser.TRAIN, ConfigParser.EVAL, ConfigParser.TEST]:
      raise Exception("Invalid dataset")
    print("=== BaseImagMaskDataset.create dataset {}".format(dataset))
    image_datapath = self.config.get(dataset, "image_datapath")
    mask_datapath  = self.config.get(dataset, "mask_datapath")
    print("=== create  {} {}".format(image_datapath, mask_datapath))

    if not os.path.exists(image_datapath) or not os.path.exists(mask_datapath):
       print("=== Not found datapath for dataset:{}".format(dataset))
       return [], []
    
    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)
    mask_channels  = self.config.get(ConfigParser.MASK, "mask_channels", dvalue=1)
   
    mask_files   = None
    if os.path.exists(mask_datapath):
      mask_files  = glob.glob(mask_datapath + "/*.jpg")
      mask_files += glob.glob(mask_datapath + "/*.png")
      mask_files += glob.glob(mask_datapath + "/*.bmp")
      mask_files += glob.glob(mask_datapath + "/*.tif")
      mask_files  = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")

    # image datatype    
    # 2024/04/15
    self.image_dtype = np.uint8
    if self.image_normalize:
      self.image_dtype = np.float32
    
    print("--- num_classes {} image data_type {}".format(self.num_classes, self.image_dtype))
    print("num_images {} {} {}".format(num_images, self.image_height, self.image_width, self.image_channels))

    X = np.zeros((num_images, self.image_height, self.image_width, self.image_channels),
                 dtype=self.image_dtype)

    # mask datatype
    # 2024/04/15
    self.mask_dtype = bool
    if self.num_classes >1:
      self.mask_dtype = np.int8
      print("--- num_classes {} mask data_type  {}".format(self.num_classes, self.mask_dtype))
    Y = np.zeros((num_images, self.image_height, self.image_width, 1, ), 
                 dtype=self.mask_dtype)


    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
      X[n]  = self.read_image_file(image_file)
      Y[n]  = self.read_mask_file(mask_files[n])

      if self.debug:
        basename    = os.path.basename(image_file)
        output_file = os.path.join(self.debug_images_dir, basename)
        cv2.imwrite(output_file, X[n])

        basename    = os.path.basename(mask_files[n])
        output_file = os.path.join(self.debug_masks_dir, basename)

        print("Y shape {}".format(Y[n].shape))
        M = Y[n]
        back = np.zeros((self.image_width, self.image_height, 1))
        for i in range(self.num_classes):
          mask = M[:,:, i]
          mask = np.expand_dims(mask, axis=-1)
          print("---- mask shape {}".format(mask.shape))
          back += mask
          
        cv2.imwrite(output_file, back)

      if self.num_classes >1:
        pass
        #print("=== call to_categorical ")
        #Y = np_utils.to_categorical(Y, self.num_classes)
    print("=== X: shape {} type {}".format(X.shape, X.dtype))
    print("=== Y: shape {} type {}".format(Y.shape, Y.dtype))

    print("-----Create X-len: {}  Y-len {}".format(len(X), len(Y)))
    return X, Y

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
  

  def read_image_file(self, image_file):
    image = imread(image_file)
  
    image = resize(image, (self.image_height, self.image_width, self.image_channels), 
                     mode='constant', 
                     preserve_range=True)
 
    # 2024/04/15
    if self.image_normalize:
      image = image/255.0
    image = image.astype(self.image_dtype)
    return image
  
  def read_mask_file(self, mask_file):
    mask = imread(mask_file)
    channel = 1
    if self.rgb_mask:
      channel = 3

    mask = resize(mask, (self.image_height, self.image_width, channel),  
                    mode='constant', 
                    preserve_range=False, 
                    anti_aliasing=False) 

    if channel == 3 and self.grayscaling:
      mask = self.to_one_hot_mask(mask)
      
    mask = mask.astype(self.mask_dtype)
    
    return mask

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = BaseImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset= ConfigParser.TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=ConfigParser.EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

