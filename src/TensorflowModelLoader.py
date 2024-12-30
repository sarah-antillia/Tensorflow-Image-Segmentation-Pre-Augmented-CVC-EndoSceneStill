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
import os
import sys
import tensorflow as tf
from ConfigParser import ConfigParser


class TensorflowModelLoader:
  BEST_MODEL_FILE = "best_model.h5"

  def __init__(self, config_file):
    self.config = ConfigParser(config_file)
    best_model_file = self.config.get(ConfigParser.TRAIN, "best_model_file", 
                                      dvalue=self.BEST_MODEL_FILE)
    model_dir       = self.config.get(ConfigParser.TRAIN, "model_dir")
    self.weight_filepath = os.path.join(model_dir, best_model_file)

  def load(self, model) :
    rc = False
    if os.path.exists(self.weight_filepath):
      model.load_weights(self.weight_filepath)
      print("=== Loaded a weight_file {}".format(self.weight_filepath))
      rc = True
    else:
      message = "Not found a weight_file " + self.weight_filepath
      raise Exception(message)
    return rc
