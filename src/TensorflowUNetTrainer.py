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
# TensorflowUNetTrainer.py
# 2023/05/30 to-arai
# 2024/04/22: Moved train method in TensorflowModel to this class 

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2024/06/01 Added two callbacks: EpochChangeInferencer and EpochChangeTiledInferencer
#
import os
import sys
import datetime
import json
import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import sys
import traceback

import tensorflow
import tensorflow as tf

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from BaseImageMaskDataset import BaseImageMaskDataset
from NormalizedImageMaskDataset import NormalizedImageMaskDataset
from RGB2GrayscaleImageMaskDataset import RGB2GrayscaleImageMaskDataset
from ImageMaskDatasetGenerator import ImageMaskDatasetGenerator
from TensorflowUNet import TensorflowUNet
from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet
from TensorflowTransUNet import TensorflowTransUNet
from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus
from TensorflowEfficientNetB7UNet import TensorflowEfficientNetB7UNet
#from TensorflowXceptionLikeUNet import TensorflowXceptionLikeUNet

from tensorflow.python.framework import random_seed

from SeedResetCallback       import SeedResetCallback
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss,  bce_dice_loss

from EpochChangeCallback import EpochChangeCallback
#from GrayScaleImageWriter import GrayScaleImageWriter

from EpochChangeInferencer import EpochChangeInferencer
from EpochChangeTiledInferencer import EpochChangeTiledInferencer

#from SeedResetCallback       import SeedResetCallback
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss,  bce_dice_loss

from mish import mish
from LineGraph import LineGraph
from mish import mish
from LineGraph import LineGraph

class TensorflowUNetTrainer:
  BEST_MODEL_FILE = "best_model.h5"
  HISTORY_JSON    = "history.json"

  def __init__(self, config_file):
    self.config_file = config_file

    self.config = ConfigParser(config_file)
    # Create a UNetModel and compile
    ModelClass = eval(config.get(ConfigParser.MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))
    # 1 Create model by call the constructor of ModelClass
    self.model= ModelClass(config_file).model
    
  def create_dirs(self, eval_dir, model_dir ):
    dt_now = str(datetime.datetime.now())
    dt_now = dt_now.replace(":", "_").replace(" ", "_")
    create_backup = self.config.get(ConfigParser.TRAIN, "create_backup", False)
    if os.path.exists(eval_dir):
      # if create_backup flag is True, move previous eval_dir to *_bak  
      if create_backup:
        moved_dir = eval_dir +"_" + dt_now + "_bak"
        shutil.move(eval_dir, moved_dir)
        print("--- Moved to {}".format(moved_dir))
      else:
        shutil.rmtree(eval_dir)

    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)

    if os.path.exists(model_dir):
      # if create_backup flag is True, move previous model_dir to *_bak  
      if create_backup:
        moved_dir = model_dir +"_" + dt_now + "_bak"
        shutil.move(model_dir, moved_dir)
        print("--- Moved to {}".format(moved_dir))      
      else:
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

  def count_files(self, dir):
     count = 0
     if os.path.exists(dir):
       count = sum(len(files) for _, _, files in os.walk(dir))
     return count

  def create_callbacks(self):
    # 2024/06/09 added dvalue=10
    patience   = self.config.get(ConfigParser.TRAIN, "patience", dvalue=10)
    weight_filepath   = os.path.join(self.model_dir, self.BEST_MODEL_FILE)
    #Modified to correct "save_weights_only" name
    save_weights_only = self.config.get(ConfigParser.TRAIN, "save_weights_only", dvalue=False)
    dmetrics    = ["accuracy", "val_accuracy"]
    metrics    = self.config.get(ConfigParser.TRAIN, "metrics", dvalue=dmetrics)

    reducer  = None
    lr_reducer = self.config.get(ConfigParser.TRAIN, "learning_rate_reducer", dvalue=False )
    if lr_reducer:
      lr_patience = int(patience/2)
      if lr_patience == 0:
        lr_patience = 5
      lr_patience = self.config.get(ConfigParser.TRAIN, "reducer_patience", dvalue= lr_patience)
      # 2024/05/30
      reducer_factor = self.config.get(ConfigParser.TRAIN, "reducer_factor", dvalue= 0.1)

      reducer = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor = 'val_loss',
                        factor  = reducer_factor, 
                        #factor  = 0.1,
                        patience= lr_patience,
                        min_lr  = 0.0)

    print("=== Created callback: EarlyStopping ")
    check_point    = tf.keras.callbacks.ModelCheckpoint(weight_filepath, verbose=1, 
                                     save_best_only    = True,
                                     save_weights_only = save_weights_only)
    print("=== Created callback: ModelCheckpoint ")

    self.epoch_change   = EpochChangeCallback(self.eval_dir, metrics)
    print("=== Created callback: EpochChangeCallback ")
    
    if reducer:
      callbacks = [check_point, self.epoch_change, reducer]
    else:
      callbacks = [check_point, self.epoch_change]
   
    seedreset_callback = self.config.get(ConfigParser.TRAIN, "seedreset_callback", dvalue=False) 
    if seedreset_callback:
      print("=== Created callback: SeedResetCallback")
      seedercb = SeedResetCallback(seed=self.seed)
      callbacks += [seedercb]

    epoch_change_infer   = self.config.get(ConfigParser.TRAIN, "epoch_change_infer", dvalue=False)

    if epoch_change_infer:
      print("=== Created callback: EpochChangeInferecer")
      inference_callback = EpochChangeInferencer(self.model, self.config_file)
      callbacks += [inference_callback]

    epoch_change_tiledinfer   = self.config.get(ConfigParser.TRAIN, "epoch_change_tiledinfer", dvalue=False)

    if epoch_change_tiledinfer:
      print("=== Created callback: EpochChangeTiledInferecer")
      tiled_inference_callback = EpochChangeTiledInferencer(self.model, self.config_file)
      callbacks += [tiled_inference_callback]

    # 2024/06/09 At last add EarlyStopping callback to the callbacks list. 
    if patience >0:
      early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1)
      callbacks += [early_stopping]

    print("=== callbacks {}".format(callbacks))
    
    return callbacks

  def train(self):
    print("==== train")
    self.batch_size = self.config.get(ConfigParser.TRAIN, "batch_size")
    self.epochs     = self.config.get(ConfigParser.TRAIN, "epochs")
    self.eval_dir   = self.config.get(ConfigParser.TRAIN, "eval_dir")
    self.model_dir  = self.config.get(ConfigParser.TRAIN, "model_dir")

    self.create_dirs(self.eval_dir, self.model_dir)
    # Copy current config_file to model_dir
    shutil.copy2(self.config_file, self.model_dir)
    print("-- Copied {} to {}".format(self.config_file, self.model_dir))
    
    self.callbacks = self.create_callbacks()

    # Create a DatasetClass
    DatasetClass = eval(self.config.get(ConfigParser.DATASET, "datasetclass", dvalue="ImageMaskDataset"))
    dataset = DatasetClass(self.config_file)
    print("=== DatasetClass {}".format(dataset))
    generator = self.config.get(ConfigParser.MODEL, "generator", dvalue=False)
    if generator == False:      
      print("=== Creating TRAIN dataset")
      train_x, train_y = dataset.create(dataset=ConfigParser.TRAIN)
      l_x_train = len(train_x)
      l_y_train = len(train_y)
      print("=== Created TRAIN dataset x_train: size {} y_train : size {}".format(l_x_train, l_y_train))

      print("=== Creating EVAL dataset")
      eval_x,  eval_y  = dataset.create(dataset=ConfigParser.EVAL)

      l_eval_x = len(eval_x) 
      l_eval_y = len(eval_y)
      if l_eval_x >0 and l_eval_y > 0:
        print("=== Created EVAL dataset eval_x: size {} eval_y : size {}".format(l_eval_x, l_eval_y))
        history = self.train_by_pre_splitted(train_x, train_y, eval_x, eval_y) 
      else:
        history = self.train_after_splitting(train_x, train_y) 
    else:
      # generator is True
      train_gen = ImageMaskDatasetGenerator(config_file, dataset=ConfigParser.TRAIN)
      train_generator = train_gen.generate()
      valid_gen = ImageMaskDatasetGenerator(config_file, dataset=ConfigParser.EVAL)
      valid_generator = valid_gen.generate()

      history = self.train_by_generator(train_generator, valid_generator)

    self.epoch_change.save_eval_graphs()
    self.save_history(history)

  def save_history(self, history): 
    #print("--- history {}".format(history.history))
    jstring = str(history.history)
    with open(self.HISTORY_JSON, 'wt') as f:
      json.dump(jstring, f, ensure_ascii=False,  indent=4, sort_keys=True, separators=(',', ': '))
      print("=== Save {}".format(self.HISTORY_JSON))

  def train_by_pre_splitted(self, train_x, train_y, valid_x, valid_y): 
      print("=== train_by_pre_splitted ")
      print("--- valid_x len {}".format(len(valid_x)))
      print("--- valid_y len {}".format(len(valid_y)))
      print("=== Start model.fit ")
      history = self.model.fit(train_x, train_y, 
                    batch_size= self.batch_size, 
                    epochs    = self.epochs, 
                    validation_data= (valid_x, valid_y),
                    shuffle   = False,
                    callbacks = self.callbacks,
                    verbose   = 1)
      return history
   
  def train_after_splitting(self, x_train, y_train, ):
      print("=== train_after_splitting ")

      dataset_splitter = self.config.get(ConfigParser.TRAIN, "dataset_splitter", dvalue=False) 
      print("=== Dataset_splitter {}".format(dataset_splitter))
   
      if dataset_splitter:
          """
          Split master dataset (x_train, y_train) into (train_x, train_y) and (valid_x, valid_y)
          This will help to improve the reproducibility of the model.
          """
          print("--- split the master train dataset")
          train_size = int(0.8 * len(x_train)) 
          train_x = x_train[:train_size]
          train_y = y_train[:train_size]
          valid_x = x_train[train_size:]
          valid_y = y_train[train_size:]

          print("--- split the master into train(0.8) and valid(0.2)")
          print("=== Start model.fit ")
          history = self.model.fit(train_x, train_y, 
                    batch_size= self.batch_size, 
                    epochs    = self.epochs, 
                    validation_data= (valid_x, valid_y),
                    shuffle   = False,
                    callbacks = self.callbacks,
                    verbose   = 1)
      else:
          print("--- Split train datasett to  train-subset and valid-subset by validation_split=0.2 ")
          # By the parameter setting : validation_split=0.2,
          # x_train and y_train will be split into real_train (0.8) and 0.2 real_valid (0.2) 
          print("=== Start model.fit ")
          history = self.model.fit(x_train, y_train, 
                    validation_split=0.2, 
                    batch_size = self.batch_size, 
                    epochs     = self.epochs, 
                    shuffle    = False,
                    callbacks  = self.callbacks,
                    verbose    = 1)
      return history
 
  def train_by_generator(self, train_generator, valid_generator):
      print("=== train_by_generator")
      print("--- Use the train and valid gnerators to fit.")
      # train and valid dataset will be used by train_generator and valid_generator respectively
      steps_per_epoch  = self.config.get(ConfigParser.TRAIN, "steps_per_epoch",  dvalue=400)
      validation_steps = self.config.get(ConfigParser.TRAIN, "validation_steps", dvalue=800)
  
      history = self.model.fit(train_generator, 
                    steps_per_epoch = steps_per_epoch,
                    epochs          = self.epochs, 
                    validation_data = valid_generator,
                    validation_steps= validation_steps,
                    shuffle         = False,
                    callbacks       = self.callbacks,
                    verbose         = 1)
      return history
  


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    config   = ConfigParser(config_file)

    trainer = TensorflowUNetTrainer(config_file)

    # 2 Call model.train()
    trainer.train()

  except:  
    traceback.print_exc()
