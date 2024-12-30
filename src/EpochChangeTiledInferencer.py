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
# EpochChangeTiledInferencer.py
# 2024/06/01:  

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import tensorflow as tf

from TiledInferencer import TiledInferencer

class EpochChangeTiledInferencer(tf.keras.callbacks.Callback):
  def __init__(self, model, config_file):
    self.tiled_infernecer = TiledInferencer(model, config_file, on_epoch_change=True)
    print("=== EpochChangeTiledInferencer.__init__ config?file {}".format(config_file))

  def on_epoch_end(self, epoch, logs):
    self.tiled_infernecer.infer(epoch=epoch)
  
