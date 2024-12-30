# Copyright 2022 antillia.com All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# LineGraph.py

# 2022/08/17 antillia.com
# 2024/04/02: Updated Constructor of LineGraph class.

import os
import sys
#sys.path.append("../../")

import os
import sys
import traceback

import matplotlib.pyplot as plt
import csv
  
class LineGraph:

  def __init__(self, figsize=(8, 6)):
    self.figsize         = figsize
    self.title_font_size = 20
    self.xticks_rotation = 0
    self.legend_loc      = "upper left"
    self.legend_frameon  = True

  def plot(self, csv_file):
    xlabel = ""
    ylabel = ""
    title = os.path.basename(csv_file)
    title = title.split(".")[0]
    x = []
    y = []
    z = []
    with open(csv_file,'r') as csvfile:
      lines = csv.reader(csvfile, delimiter=',')

      for i, row in enumerate(lines):
        if i == 0:
           xlabel = row[0]
           ylabel = row[1]
           zlabel = row[2]
        #print(row)
        if i>0:
          x.append(float(row[0]))
          y.append(float(row[1]))
          z.append(float(row[2]))
    plt.figure(figsize=self.figsize)

    plt.plot(x, y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = ylabel)

    plt.plot(x, z, color = 'r', linestyle = 'dashed',
         marker = 'o',label = zlabel)
  
    plt.xticks(rotation = self.xticks_rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize = self.title_font_size)
    plt.grid()
    plt.legend(loc=self.legend_loc, frameon=self.legend_frameon)
    #plt.show()
    png_file = csv_file.replace(".csv", ".png")
    #print("--- png_file {}".format(png_file))
    plt.savefig(png_file)
    # 2024/04/02
    plt.close()

if __name__ == "__main__":
  try:
    csv_file = ""
    if len(sys.argv) == 2:
      csv_file = sys.argv[1]
    else:
      raise Exception("Invalid argment")
    linegraph = LineGraph()
    linegraph.plot(csv_file)
  
  except:
    traceback.print_exc()
