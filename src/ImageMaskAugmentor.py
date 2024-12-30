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

# ImageMaskAugmentor.py
# 2023/08/20 to-arai
# 2023/08/25 Fixed bugs on some wrong section name settings.
# 2023/08/26 Added shrink method to augment images and masks.
# 2023/08/27 Added shear method to augment images and masks.
# 2023/08/28 Added elastic_transorm method to augment images and masks.
# 2024/02/12 Modified shear method to check self.hflip and self.vflip flags
# 2024/04/06 Added distort method.
# 2024/04/08 Modified to use [deformation] section.
# [deformation]
# alpah   = 1300
# sigmoid = 8
# Modified method name 'elastic_transform' to 'deform' 

# 2024/05/10 

# 2024/06/18 Addded barrel_distort method has been take from the following
# code in stackoverflow.com
# # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion

# 2024/09/01 Added pincushion_distortion to train_eval_infer.config
# amount should be negative 
"""
[pincdistortion]
radius = 0.3
amount = -0.3
centers =  [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
"""

# 2024/06/18 Added Barrel_distortion to train_eval_infer.config
# amount should be positive
"""
[barrdistortion]
radius = 0.3
amount = 0.3
centers =  [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
"""

# 2024/06/20 Modified to use sigmoids list instead of sigmoid in [deformation] section.
# [deformation]
#sigmoids  = [8.0 10.0]
# Modified method name 'elastic_transform' to 'deform' 

import os
import sys
import glob
import shutil
import math
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from ConfigParser import ConfigParser
import traceback

class ImageMaskAugmentor:
  
  def __init__(self, config_file):
    self.seed     = 137
    self.config   = ConfigParser(config_file)
    #self.config.dump_all()

    self.debug    = self.config.get(ConfigParser.GENERATOR, "debug",   dvalue=True)
    self.verbose =  self.config.get(ConfigParser.GENERATOR, "verbose", dvalue=False)
    self.W        = self.config.get(ConfigParser.MODEL,     "image_width")
    self.H        = self.config.get(ConfigParser.MODEL,     "image_height")
    
    self.rotation = self.config.get(ConfigParser.AUGMENTOR, "rotation", dvalue=True)
    self.SHRINKS  = self.config.get(ConfigParser.AUGMENTOR, "shrinks",  dvalue=[0.8])
    self.ANGLES   = self.config.get(ConfigParser.AUGMENTOR, "angles",   dvalue=[90, 180, 270])

    self.SHEARS   = self.config.get(ConfigParser.AUGMENTOR, "shears",   dvalue=[])

    self.hflip    = self.config.get(ConfigParser.AUGMENTOR, "hflip", dvalue=True)
    self.vflip    = self.config.get(ConfigParser.AUGMENTOR, "vflip", dvalue=True)
  
    self.deformation = self.config.get(ConfigParser.AUGMENTOR, "deformation", dvalue=False)
    if self.deformation:
      self.alpha    = self.config.get(ConfigParser.DEFORMATION, "alpah", dvalue=1300)
      sigmoid = self.config.get(ConfigParser.DEFORMATION, "sigmoid", dvalue=8)
      #2024/06/16 You may the simgoids parameter as a list somethinkg like [8,10]
      sigmoids = self.config.get(ConfigParser.DEFORMATION, "sigmoids", dvalue=[8])
      if len(sigmoids) !=0:
        self.sigmoids = sigmoids
      else:
        self.sigmoids = [sigmoid]
      # 2024/09/01
      print("--- simgoids {}".format(self.sigmoids))

    self.distortion = self.config.get(ConfigParser.AUGMENTOR, "distortion", dvalue=False)
    # Distortion
    if self.distortion:
      self.gaussina_filer_rsigma = self.config.get(ConfigParser.DISTORTION, "gaussian_filter_rsigma", dvalue=40)
      self.gaussina_filer_sigma  = self.config.get(ConfigParser.DISTORTION, "gaussian_filter_sigma",  dvalue=0.5)
      self.distortions           = self.config.get(ConfigParser.DISTORTION, "distortions",  dvalue=[0.02])
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)

    self.sharpening = self.config.get(ConfigParser.AUGMENTOR, "sharpening", dvalue=False)
    # Sharpening
    if self.sharpening:
      self.sharpen_k = self.config.get(ConfigParser.SHARPENING, "k", dvalue=1.0)

    self.brightening = self.config.get(ConfigParser.AUGMENTOR, "brightening", dvalue=False)
    # Brightening
    if self.brightening:
      self.alpha = self.config.get(ConfigParser.BRIGHTENING, "alpha", dvalue=1.2)
      self.beta  = self.config.get(ConfigParser.BRIGHTENING, "beta", dvalue=10)

    self.barrdistortion = self.config.get(ConfigParser.AUGMENTOR, "barrdistortion", dvalue=False)
    # Barrel_distortion
    if self.barrdistortion:
      self.radius  = self.config.get(ConfigParser.BARRDISTORTION, "radius",  dvalue=0.5)
      self.amount  = self.config.get(ConfigParser.BARRDISTORTION, "amount",  dvalue=0.5)
      self.centers = self.config.get(ConfigParser.BARRDISTORTION, "centers", dvalue=[(0.5, 0.5)])

    # 2024/09/01
    self.pincdistortion = self.config.get(ConfigParser.AUGMENTOR, "pincdistortion", dvalue=False)
    if self.pincdistortion:
      self.pincradius  = self.config.get(ConfigParser.PINCDISTORTION, "radius",  dvalue=0.5)
      self.pincamount  = self.config.get(ConfigParser.PINCDISTORTION, "amount",  dvalue=-0.5)
      self.pinccenters = self.config.get(ConfigParser.PINCDISTORTION, "centers", dvalue=[(0.5, 0.5)])

  # It applies  horizotanl and vertical flipping operations to image and mask repectively.
  def augment(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    if self.hflip:
      hflip_image = self.horizontal_flip(image) 
      hflip_mask  = self.horizontal_flip(mask) 
      IMAGES.append(hflip_image )
      MASKS.append( hflip_mask  )
      if self.debug:
        filepath = os.path.join(generated_images_dir, "hfliped_" + image_basename)
        cv2.imwrite(filepath, hflip_image)
        if self.verbose:
          print("=== Saved {}".format(filepath))
        filepath = os.path.join(generated_masks_dir,  "hfliped_" + mask_basename)
        cv2.imwrite(filepath, hflip_mask)
        if self.verbose:
          print("=== Saved {}".format(filepath))

    if self.vflip:
      vflip_image = self.vertical_flip(image)
      vflip_mask  = self.vertical_flip(mask)
      IMAGES.append(vflip_image )
      MASKS.append( vflip_mask  )

      if self.debug:
        filepath = os.path.join(generated_images_dir, "vfliped_" + image_basename)
        cv2.imwrite(filepath, vflip_image)
        if self.verbose:
          print("=== Saved {}".format(filepath))

        filepath = os.path.join(generated_masks_dir,  "vfliped_" + mask_basename)
        cv2.imwrite(filepath, vflip_mask)
        if self.verbose:
          print("=== Saved {}".format(filepath))

    if self.rotation:
       self.rotate(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
       
    if type(self.SHRINKS) is list and len(self.SHRINKS)>0:
       self.shrink(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if type(self.SHEARS) is list and len(self.SHEARS)>0:
       self.shear(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if self.deformation:
      self.deform(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
      
    if self.distortion:
      self.distort(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if self.barrdistortion:
      self.barrel_distort(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
    # 2024/09/01
    if self.pincdistortion:
      self.pincushion_distort(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if self.sharpening:
      self.sharpen(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if self.brightening:
      self.brighten(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )


  def horizontal_flip(self, image): 
    image = image[:, ::-1, :]
    return image

  def vertical_flip(self, image):
    image = image[::-1, :, :]
    return image
  
  def rotate(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    for angle in self.ANGLES:      

      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
      color  = image[2][2].tolist()
      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=color)
      color  = mask[2][2].tolist()
      rotated_mask  = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(self.W, self.H), borderValue=color)
      #rotated_mask  = np.expand_dims(rotated_mask, axis=-1) 

      if self.debug:
        filepath = os.path.join(generated_images_dir, "rotated_" + str(angle) + "_" + image_basename)
        cv2.imwrite(filepath, rotated_image)
        if self.verbose:
          print("=== Saved {}".format(filepath))
        filepath = os.path.join(generated_masks_dir,  "rotated_" + str(angle) + "_" + mask_basename)
        cv2.imwrite(filepath, rotated_mask)
        if self.verbose:
          print("=== Saved {}".format(filepath))
  

      if rotated_mask.ndim==2:
        rotated_mask  = np.expand_dims(rotated_mask, axis=-1) 

      IMAGES.append(rotated_image)
      MASKS.append(rotated_mask)


  def shrink(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):

    h, w = image.shape[:2]
  
    for shrink in self.SHRINKS:
      rw = int (w * shrink)
      rh = int (h * shrink)
      resized_image = cv2.resize(image, dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      resized_mask  = cv2.resize(mask,  dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      
      squared_image = self.paste(resized_image, mask=False)
      squared_mask  = self.paste(resized_mask,  mask=True)

      if self.debug:
        ratio   = str(shrink).replace(".", "_")
        image_filename = "shrinked_" + ratio + "_" + image_basename
        filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(filepath, squared_image)
        if self.verbose:
          print("=== Saved {}".format(filepath))
    
        mask_filename = "shrinked_" + ratio + "_" + mask_basename
        filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(filepath, squared_mask)
        if self.verbose:
          print("=== Saved {}".format(filepath))

      if squared_mask.ndim==2:
        squared_mask  = np.expand_dims(squared_mask, axis=-1) 

      IMAGES.append(squared_image)
      MASKS.append(squared_mask)


  def paste(self, image, mask=False):
    l = len(image.shape)
   
    h, w,  = image.shape[:2]
    if l==3:
      back_color = image[2][2]
      background = np.ones((self.H, self.W, 3), dtype=np.uint8)
      background = background * back_color

      #background = np.zeros((self.H, self.W, 3), dtype=np.uint8)

      (b, g, r) = image[h-10][w-10] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background += [b, g, r][::-1]
    else:
      v =  image[h-10][w-10] 
      image  = np.expand_dims(image, axis=-1) 
      background = np.zeros((self.H, self.W, 1), dtype=np.uint8)
      background[background !=v] = v
    x = (self.W - w)//2
    y = (self.H - h)//2
    background[y:y+h, x:x+w] = image
    return background
  

  # This method has been taken from the following code in stackoverflow.
  # https://stackoverflow.com/questions/57881430/how-could-i-implement-a-centered-shear-an-image-with-opencv
  # Do shear, hflip and vflip the image and mask and save them 
  def shear(self, IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename ):

    if self.SHEARS == None or len(self.SHEARS) == 0:
      return
   
    H, W = image.shape[:2]
    for shear in self.SHEARS:
      ratio = str(shear).replace(".", "_")
      M2 = np.float32([[1, 0, 0], [shear, 1,0]])
      M2[0,2] = -M2[0,1] * H/2 
      M2[1,2] = -M2[1,0] * W/2 

      # 1 shear and image
      color  = image[2][2].tolist()
      sheared_image = cv2.warpAffine(image, M2, (W, H), borderValue=color)
    
      color  = mask[2][2].tolist()
      sheared_mask  = cv2.warpAffine(mask,  M2, (W, H), borderValue=color)

      if self.debug:
        # 2 save sheared image and mask 
        filepath = os.path.join(generated_images_dir, "sheared_" + ratio + "_" + image_basename)
        cv2.imwrite(filepath, sheared_image)
        if self.verbose:
          print("=== Saved {}".format(filepath))

        filepath = os.path.join(generated_masks_dir,  "sheared_" + ratio + "_" + mask_basename)
        cv2.imwrite(filepath, sheared_mask)
        if self.verbose:
          print("=== Saved {}".format(filepath))

      if sheared_mask.ndim == 2:
        sheared_mask  = np.expand_dims(sheared_mask, axis=-1) 
      IMAGES.append(sheared_image)
      MASKS.append(sheared_mask)

      if self.hflip:
        # hflipp sheared image and mask
        hflipped_sheared_image = self.horizontal_flip(sheared_image)
        hflipped_sheared_mask  = self.horizontal_flip(sheared_mask)

        if self.debug:
          filepath = os.path.join(generated_images_dir, "hflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, hflipped_sheared_image)
          if self.verbose:
            print("=== Saved {}".format(filepath))

          filepath = os.path.join(generated_masks_dir,  "hflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, hflipped_sheared_mask)
          if self.verbose:
            print("=== Saved {}".format(filepath))

        if hflipped_sheared_mask.ndim == 2:
          hflipped_sheared_mask  = np.expand_dims(hflipped_sheared_mask, axis=-1) 
        IMAGES.append(hflipped_sheared_image)
        MASKS.append(hflipped_sheared_mask)

      if self.vflip:

        vflipped_sheared_image  = self.vertical_flip(sheared_image)
        vflipped_sheared_mask   = self.vertical_flip(sheared_mask)

        if self.debug:
          filepath = os.path.join(generated_images_dir, "vflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, vflipped_sheared_image)
          if self.verbose:
            print("=== Saved {}".format(filepath))
          filepath = os.path.join(generated_masks_dir,  "vflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, vflipped_sheared_mask)
          if self.verbose:
            print("=== Saved {}".format(filepath))

        if vflipped_sheared_mask.ndim == 2:
          vflipped_sheared_mask  = np.expand_dims(vflipped_sheared_mask, axis=-1) 
        IMAGES.append(vflipped_sheared_image)
        MASKS.append(vflipped_sheared_mask)
        
      if self.hflip and self.vflip:
        hvflipped_sheared_image = self.vertical_flip(hflipped_sheared_image)
        hvflipped_sheared_mask  = self.vertical_flip(hflipped_sheared_mask)

        if self.debug:
          filepath = os.path.join(generated_images_dir, "hvflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, hvflipped_sheared_image)
          if self.verbose:
            print("=== Saved {}".format(filepath))
          filepath = os.path.join(generated_masks_dir,  "hvflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, hvflipped_sheared_mask)
          if self.verbose:
            print("=== Saved {}".format(filepath))

        if hvflipped_sheared_mask.ndim == 2:
          hvflipped_sheared_mask  = np.expand_dims(hvflipped_sheared_mask, axis=-1) 

        IMAGES.append(hvflipped_sheared_image)
        MASKS.append(hvflipped_sheared_mask)


  # This method has been taken from the following code.
  # https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
  #
  # https://cognitivemedium.com/assets/rmnist/Simard.pdf
  #
  # See also
  # https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook
  # 
  def deform(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    # 2024/06/16 Modified to use self.sigmoids
    for sigmoid in self.sigmoids:
      sigmoid = int(sigmoid)
      shape = image.shape

      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      shape = mask.shape
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
      deformed_mask = map_coordinates(mask, indices, order=1, mode='nearest')  
      deformed_mask = deformed_mask.reshape(mask.shape)
    
      IMAGES.append(deformed_image)
      MASKS.append(deformed_mask)

      if self.debug:
        image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + image_basename
        image_filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, deformed_image)
        if self.verbose:
          print("=== Saved {}".format(image_filepath))
    
        mask_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, deformed_mask)
        if self.verbose:
          print("=== Saved {}".format(mask_filepath))

  # The code used here is based on the following stakoverflow web-site
  #https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420

  def distort(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename):
    for size in self.distortions:
      distorted_image = self.distort_one(image, size)  
      distorted_image = distorted_image.reshape(image.shape)
      distorted_mask  = self.distort_one(mask, size)
      distorted_mask  = distorted_mask.reshape(mask.shape)

      IMAGES.append(distorted_image)
      MASKS.append(distorted_mask)

      if self.debug:
        image_filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + image_basename

        image_filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, distorted_image)
        if self.verbose:
          print("=== Saved {}".format(image_filepath))
    
        mask_filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, distorted_mask)
        if self.verbose:
          print("=== Saved {}".format(mask_filepath))

  def distort_one(self, image, size):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)

    dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
    dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
    sizex = int(xsize * size)
    sizey = int(xsize * size)
    dx *= sizex/dx.max()  
    dy *= sizey/dy.max()

    img = gaussian_filter(image, self.gaussina_filer_sigma)

    yy, xx = np.indices(shape)
    xmap = (xx-dx).astype(np.float32)
    ymap = (yy-dy).astype(np.float32)

    distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
    distorted = cv2.resize(distorted, (w, h))
    return distorted

  # 2024/06/16 
  def sharpen(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):

    k = self.sharpen_k
    kernel = np.array([[-k, -k, -k], 
                       [-k, 1+8*k, -k], 
                       [-k, -k, -k]])
    sharpened_image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    sharpened_mask  = cv2.filter2D(mask,  ddepth=-1, kernel=kernel)
    if sharpened_mask.ndim == 2:
        sharpened_mask  = np.expand_dims(sharpened_mask, axis=-1) 
    IMAGES.append(sharpened_image)
    MASKS.append(sharpened_mask)

    if self.debug:
        image_filename = "sharpened_" + str(k) + "_" + image_basename
        image_filepath = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, sharpened_image)
    
        mask_filename  = "sharpened_" + str(k) + "_" +  mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, sharpened_mask)

  # 2024/06/16 
  def brighten(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    
    adjusted_image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
    adjusted_mask  = cv2.convertScaleAbs(mask,  alpha=self.alpha, beta=self.beta)
    if adjusted_mask.ndim == 2:
        adjusted_mask  = np.expand_dims(adjusted_mask, axis=-1) 
    IMAGES.append(adjusted_image)
    MASKS.append(adjusted_mask)
    if self.debug:
        image_filename = "brightened_" + str(self.alpha) + "_" + str(self.beta) +  image_basename
        image_filepath = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, adjusted_image)
    
        mask_filename  = "brightened_" + str(self.alpha) + "_" + str(self.beta) +  mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, adjusted_mask)

  # The following barrel_distort method has been take from the following
  # code in stackoverflow.com
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion

  def barrel_distort(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename):
    distorted_image  = image
    distorted_mask   = mask
    (h, w, _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.centers:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.radius
      amount = self.amount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      distorted_mask  = cv2.remap(distorted_mask,  map_x, map_y, cv2.INTER_LINEAR)
      if distorted_mask.ndim == 2:
        distorted_mask  = np.expand_dims(distorted_mask, axis=-1) 
      IMAGES.append(distorted_image)
      MASKS.append(distorted_mask)

 
      if self.debug:
        image_filename = "barrdistorted_" + str(index) + "_" + self.radius + "_" + self.amount + "_" + image_basename

        image_filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, distorted_image)
        if self.verbose:
          print("=== Saved {}".format(image_filepath))
    
        mask_filename = "barrdistorted_" + str(index) + "_" + self.radius + "_" + self.amount + "_" + mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, distorted_mask)
        if self.verbose:
          print("=== Saved {}".format(mask_filepath))

  # The following barrel_distort method has been take from the following
  # code in stackoverflow.com
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  # 2024/09/01
  def pincushion_distort(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename):
    distorted_image  = image
    distorted_mask   = mask
    (h, w, _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.pinccenters:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.pincradius
      amount = self.pincamount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      distorted_mask  = cv2.remap(distorted_mask,  map_x, map_y, cv2.INTER_LINEAR)
      if distorted_mask.ndim == 2:
        distorted_mask  = np.expand_dims(distorted_mask, axis=-1) 
      IMAGES.append(distorted_image)
      MASKS.append(distorted_mask)

      if self.debug:
        image_filename = "pincdistorted_" + str(index) + "_" + self.radius + "_" + self.amount + "_" + image_basename

        image_filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, distorted_image)
        if self.verbose:
          print("=== Saved {}".format(image_filepath))
    
        mask_filename = "pincdistorted_" + str(index) + "_" + self.radius + "_" + self.amount + "_" + mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, distorted_mask)
        if self.verbose:
          print("=== Saved {}".format(mask_filepath))