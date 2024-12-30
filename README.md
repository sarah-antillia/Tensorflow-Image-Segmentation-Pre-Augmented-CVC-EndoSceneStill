<h2>Tensorflow-Image-Segmentation-Pre-Augmented-CVC-EndoSceneStill (2025/01/01)</h2>

This is the first experiment of Image Segmentation for EndoScene (Endoluminal Scene of Colonoscopy Images)
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a pre-augmented <a href="https://drive.google.com/file/d/1CK3cRrvJqlRKLmKSc7WL708PBxFd9pqX/view?usp=sharing">
EndoScene-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://drive.google.com/file/d/1MuO2SbGgOL_jdBu3ffSf92feBtj8pbnw/view?usp=sharing">
 CVC-EndoSceneStill 
</a>
 dataset 
<br><br>
<b>Data Augmentation Strategy:</b><br>
 To address the limited size of CVC-EndoSceneStill,which contains 547 images and their corresponding masks in the TrainDataset, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a pre-augmented dataset, which supports the following augmentation methods.
<br><br>
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>

<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/5.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/27.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/27.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/27.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/295.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/295.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/295.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this EndoScene Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following google drive 
<a href="https://drive.google.com/file/d/1CK3cRrvJqlRKLmKSc7WL708PBxFd9pqX/view?usp=sharing">
 CVC-EndoSceneStill </a>
<br>
On <b>CVC-EndoSceneStill</b> dataset,   
please refer to :<a href="<b>A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images</b><br>
 <a href="https://onlinelibrary.wiley.com/doi/10.1155/2017/4037190">
<b>A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images</b></a><br>

<br>
<h3>
<a id="2">
2 EndoScene ImageMask Dataset
</a>
</h3>
 If you would like to train this EndoScene Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="">
EndoScene-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─EndoScene
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>

On the derivation of this dataset, please refer to the following Python scripts:
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
The folder structure of the original EndoScene is the following.<br>
<pre>
./EndoScene
├─TestDataset
│  ├─boundary
│  ├─images
│  └─masks
├─TrainDataset
│  ├─boundary
│  ├─images
│  └─masks
└─ValidationDataset
    ├─boundary
    ├─images
    └─masks
</pre>
The is a 512x512 pixels pre-augmented dataset generated by the ImageMaskDatasetGenerator.py from the TrainDataset only.
<br>
<br>
<b>EndoScene Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/EndoScene_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained EndoSceneTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/EndoScene and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.03
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 97  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/train_console_output_at_epoch_97.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/EndoScene</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for EndoScene.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/evaluate_console_output_at_epoch_97.png" width="720" height="auto">
<br><br>Image-Segmentation-EndoScene

<a href="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this EndoScene/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.05
dice_coef,0.9338
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/EndoScene</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for EndoScene.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/22.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/22.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/22.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/52.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/52.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/52.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/162.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/162.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/162.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/241.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/241.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/241.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/295.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/295.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/295.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/images/313.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test/masks/313.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/EndoScene/mini_test_output/313.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images</b><br>
David Vázquez, Jorge Bernal, F. Javier Sánchez, Gloria Fernández-Esparrach,<br>
 Antonio M. López, Adriana Romero, Michal Drozdzal, Aaron Courville<br>
 <a href="https://onlinelibrary.wiley.com/doi/10.1155/2017/4037190">https://onlinelibrary.wiley.com/doi/10.1155/2017/4037190</a><br>
 
First published: 26 July 2017 <br>
<a href="https://doi.org/10.1155/2017/4037190">https://doi.org/10.1155/2017/4037190</a>
<br>
<br>

<b>2. Rethinking the transfer learning for FCN based polyp segmentation in colonoscopy</b><br>
Yan Wen, Lei Zhang1, Xiangli Meng and Xujiong Ye<br>
<a href="https://arxiv.org/pdf/2211.02416">https://arxiv.org/pdf/2211.02416</a>
<br>
<br>
<b>3. Polyp detection and segmentation from endoscopy images</b><br>
Mariia Kokshaikyna, Mariia Dobko, and Oles Dobosevych<br>
<a href="https://s3.eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/sites/8/2022/12/MS-AMLV_2022_paper_7.pdf">
https://s3.eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/sites/8/2022/12/MS-AMLV_2022_paper_7.pdf</a>
<br>


