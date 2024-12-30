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
# TensorflowUNetInfer.py
# 2023/06/05 to-arai
# 2024/04/22: Moved infer method in TensorflowModel to this class 
# 2024/06/03: Modified to use Inferencer class.

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import numpy as np
import sys

import traceback

from ConfigParser import ConfigParser


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
from TensorflowModelLoader import TensorflowModelLoader

from Inferencer import Inferencer

class TensorflowUNetInferencer:

  def __init__(self, config_file):
    print("=== TensorflowUNetInferencer.__init__ config?file {}".format(config_file))
    self.config = ConfigParser(config_file)
    
    # Create a UNetMolde and compile
    ModelClass = eval(self.config.get(ConfigParser.MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))

    self.unet  = ModelClass(config_file) 
    print("--- self.unet {}".format(self.unet))
    self.model = self.unet.model

    # 2024/04/22 Load Model
    self.loader = TensorflowModelLoader(config_file)
    self.loader.load(self.model)

    # 2024/06/10
    self.inferencer = Inferencer(self.model, config_file)    
    print("=== Create Inferencer")

  def infer(self):
    print("=== TensorflowUNetInferencer call inferencer.infer()")
    self.inferencer.infer()
  
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    inferencer = TensorflowUNetInferencer(config_file)

    inferencer.infer()

  except:
    traceback.print_exc()
    

