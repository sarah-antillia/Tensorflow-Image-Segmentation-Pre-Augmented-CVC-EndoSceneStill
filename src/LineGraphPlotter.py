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
# 2024/03/29 (C) antillia.com

import os
import sys

import traceback
from ConfigParser import ConfigParser

import matplotlib.pyplot as plt

from LineGraph import LineGraph

class LineGraphPlotter(LineGraph):
  # Constructor
  def __init__(self):
    super().__init__()

  def plot(self, eval_dir):
    train_metrics = os.path.join(eval_dir, "train_metrics.csv")
    train_losses  = os.path.join(eval_dir, "train_losses.csv")
    super.plot(train_metrics)
    super.plot(train_losses)

if __name__ == "__main__":
  try:
    eval_dir = "./eval"
    config_file = "./train_eval_infer.config"

    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    config   = ConfigParser(config_file)
    eval_dir = config.get("train", "eval_dir")
    if not os.path.exists(eval_dir):
      raise Exception("Not found "+ eval_dir) 
    plotter = LineGraphPlotter()
    plotter.plot(eval_dir)

  except:
    traceback.print_exc()
