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

# TensorflowUNetEvaluator.py
# 2023/05/30 to-arai
# 2023/08/22 Updated to use test dataset on 
#[model]
#GENERATOR = True
# 2024/04/22: Moved evaluate method in TensorflowModel to this class 

# 2024/04/24 Fixed a bug in evaluate method.
# ConfigParser.DATASETCLASS -> ConfigParser.DATASET

import os
import sys

import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from BaseImageMaskDataset import BaseImageMaskDataset
from NormalizedImageMaskDataset import NormalizedImageMaskDataset
# 2024/04/24 Added the following line
from RGB2GrayscaleImageMaskDataset import RGB2GrayscaleImageMaskDataset

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
from ConfigParser import ConfigParser
from TensorflowModelLoader import TensorflowModelLoader

class TensorflowUNetEvaluator:

  def __init__(self, config_file):
    self.config = ConfigParser(config_file)
    # Create a UNetModel and compile
    ModelClass = eval(self.config.get(ConfigParser.MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))
    # 1 Create model by call the constructor of ModelClass
    self.model= ModelClass(config_file).model

    self.loader = TensorflowModelLoader(config_file)
    self.loader.load(self.model)
    
  def evaluate(self):
    # Create a DatasetClass
    # 2024/03/05 MODEL -> DATASETCLASS
    # 2024/04/24 ConfigParser.DATASETCLASS -> ConfigParser.DATASET
    DatasetClass = eval(self.config.get(ConfigParser.DATASET, "datasetclass", dvalue="ImageMaskDataset"))
    print("=== DatasetClass {}".format(DatasetClass))

    dataset = DatasetClass(config_file)

    # 2024/04/13 You may specify an evaluation section name including
    # both image and mask datapath in [model] section.
    """
    ; train_eval_infer.config
    [model]
    evaluation="test"
    ...
    [test]
    image_datapath = "../../../dataset/Breast-Cancer/test/images/"
    mask_datapath  = "../../../dataset/Breast-Cancer/test/masks/"
    """
    target = self.config.get(ConfigParser.MODEL, "evaluation", dvalue=ConfigParser.TEST)
    # target should be one of config-setion names TEST or EVAL or your own dataset
    x_test, y_test = dataset.create(dataset=target)
  
    self._evaluate(x_test, y_test)

  def _evaluate(self, x_test, y_test): 
    #self.load_model()

    batch_size = self.config.get(ConfigParser.EVAL, "batch_size", dvalue=4)
    print("=== evaluate batch_size {}".format(batch_size))
    scores = self.model.evaluate(x_test, y_test, 
                                batch_size = batch_size,
                                verbose = 1)
    test_loss     = str(round(scores[0], 4))
    test_accuracy = str(round(scores[1], 4))
    print("Test loss    :{}".format(test_loss))     
    print("Test accuracy:{}".format(test_accuracy))
    # Added the following lines to write the evaluation result.
    loss    = self.config.get(ConfigParser.MODEL, "loss")
    metrics = self.config.get(ConfigParser.MODEL, "metrics")
    metric = metrics[0]
    evaluation_result_csv = "./evaluation.csv"    
    with open(evaluation_result_csv, "w") as f:
       metrics = self.model.metrics_names
       for i, metric in enumerate(metrics):
         score = str(round(scores[i], 4))
         line  = metric + "," + score
         print("--- Evaluation  metric:{}  score:{}".format(metric, score))
         f.writelines(line + "\n")     
    print("--- Saved {}".format(evaluation_result_csv))


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]

    evaluator = TensorflowUNetEvaluator(config_file)
    evaluator.evaluate()
    
  except:
    traceback.print_exc()
        
