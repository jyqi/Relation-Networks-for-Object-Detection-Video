# Relation Networks for Object Detection + Flow-Guided Feature Aggregation + Deformable Nets


Forked from [**Relation Networks for Object Detection**](https://github.com/msracver/Relation-Networks-for-Object-Detection) with major contributors [Dazhi Cheng](https://github.com/chengdazhi), [Jiayuan Gu](https://github.com/Jiayuan-Gu), [Han Hu](https://github.com/ancientmooner) and [Zheng Zhang](https://github.com/stupidZZ).

Joined with [**Flow-Guided Feature Aggregation (FGFA)**](https://github.com/msracver/Flow-Guided-Feature-Aggregation) with major contributors [Yuqing Zhu](https://github.com/jeremy43), [Shuhao Fu](https://github.com/howardmumu), and [Xizhou Zhu](https://github.com/einsiedler0408), when they are interns at MSRA.

And with [**Deformable ConvNets**](https://github.com/msracver/Deformable-ConvNets) with major contributors [Yuwen Xiong](https://github.com/Orpine), [Haozhi Qi](https://github.com/Oh233), [Guodong Zhang](https://github.com/gd-zhang), [Yi Li](https://github.com/liyi14), [Jifeng Dai](https://github.com/daijifeng001), [Bin Xiao](https://github.com/leoxiaobin), [Han Hu](https://github.com/ancientmooner) and  [Yichen Wei](https://github.com/YichenWei).

## Introduction

**Relation Networks for Object Detection** is described in an [CVPR 2018 oral paper](https://arxiv.org/abs/1711.11575). 

**Flow-Guided Feature Aggregation (FGFA)** is described in an [ICCV 2017 paper](https://arxiv.org/abs/1703.10025).

**Deformable ConvNets** is described in an [ICCV 2017 oral paper](https://arxiv.org/abs/1703.06211).



## Disclaimer
*From the original Relation Networks README*

This is an official implementation for [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575) based on MXNet. It is worth noting that:

  * This repository is tested on official [MXNet v1.1.0@(commit 629bb6)](https://github.com/apache/incubator-mxnet/commit/e29bb6f76365e45dd44e23941692c9d969959315). You should be able to use it with any version of MXNET that contains required operators like Deformable Convolution. 
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository is based on [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets).


**Our modified code is tested on Ubuntu 16.04 with CUDA 9.1 and MXNet 1.2.1**


## License

© Microsoft, 2018. Licensed under an MIT license.

## Citing

If you find Relation Networks for Object Detection useful in your research, please consider citing:
```
@article{hu2017relation,
  title={Relation Networks for Object Detection},
  author={Hu, Han and Gu, Jiayuan and Zhang, Zheng and Dai, Jifeng and Wei, Yichen},
  journal={arXiv preprint arXiv:1711.11575},
  year={2017}
} 
```


If you find Flow-Guided Feature Aggregation useful in your research, please consider citing:
```
@inproceedings{zhu17fgfa,
    Author = {Xizhou Zhu, Yujie Wang, Jifeng Dai, Lu Yuan, Yichen Wei},
    Title = {Flow-Guided Feature Aggregation for Video Object Detection},
    Conference = {ICCV},
    Year = {2017}
}

@inproceedings{dai16rfcn,
    Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
    Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
    Conference = {NIPS},
    Year = {2016}
}
```

If you find Deformable ConvNets useful in your research, please consider citing:
```
@article{dai17dcn,
    Author = {Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei},
    Title = {Deformable Convolutional Networks},
    Journal = {arXiv preprint arXiv:1703.06211},
    Year = {2017}
}
@inproceedings{dai16rfcn,
    Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
    Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
    Conference = {NIPS},
    Year = {2016}
}
```


## Main Results

#### Faster RCNN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>Post Processing Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 31.8 | 53.9 | 32.2 | 10.5 | 35.2 | 51.5 | 0.168s | 0.025s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 32.3 | 52.8 | 34.1 | 11.1 | 35.9 | 51.8 | 0.200s | 0.060s |
| <sub>2FC + Relation Module + softnms<br />ResNet-101</sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 34.7 | 55.3 | 37.2 | 13.7 | 38.8 | 53.6 | 0.211s | 0.059s |
| <sub>2FC + Learn NMS </br>ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 32.6 | 51.8 |  35.0  | 11.8 | 36.6 | 52.1 | 0.162s | 0.020s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br />ResNet-101</sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 35.2 | 55.5 | 38.0 | 15.2 | 39.2 | 54.1 | 0.175s | 0.022s |

#### Deformable Faster RCNN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>NMS Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 37.2 | 58.1 | 40.0 | 16.4 | 41.3 | 55.5 | 0.180s | 0.022s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 37.5 | 57.3 | 41.0 | 16.6 | 41.7 | 55.8 | 0.208s | 0.052s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 38.4 | 57.6 | 41.6 | 18.2 | 43.1 | 56.6 | 0.188s | 0.023s |

#### FPN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>NMS Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 36.6 | 59.3 | 39.3 | 20.3 | 40.5 | 49.4 | 0.196s | 0.037s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 36.8 | 57.8 | 40.7 | 20.4 | 40.8 | 49.7 | 0.323s | 0.167s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 38.6 | 59.9 | 43.0 | 22.1 | 42.3 | 52.8 | 0.232s | 0.022s |


*Running time is counted on a single Maxwell Titan X GPU (mini-batch size is 1 in inference).*

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/apache/incubator-mxnet). We tested our code on MXNet 1.2.1. Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. 

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.


3. The following Python packages:
  ```
  Cython
  EasyDict
  mxnet-cu91 # changed from mxnet-cu80 used in relation networks code
  opencv-python
  ```


## Requirements: Hardware

Any NVIDIA GPUs with at least 6GB memory should be OK.

## Installation

1. Clone the repository.
```
git clone https://github.com/HaydenFaulkner/Relation-Networks-for-Object-Detection-Video.git
cd Relation-Networks-for-Object-Detection-Video
```

2. Run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

  ***Quick start***

  3.1 Install MXNet and all dependencies by 
  ```
  pip install -r requirements.txt
  ```
  If there is no other error message, MXNet should be installed successfully. 

  If you get an error about not finding `libcudart.so` even after having your environment variables set, try running (with the correct paths):
  ```
  sudo sh -c "echo '/usr/local/cuda/lib64\n/usr/local/cuda/lib' >> /etc/ld.so.conf.d/nvidia.conf"
  sudo ldconfig
  ```

  ***Build from source (alternative way)***

  3.2 Clone MXNet v1.1.0 by
  ```
  git clone -b v1.1.0 --recursive https://github.com/apache/incubator-mxnet.git
  ```
  3.3 Compile MXNet
  ```
  cd ${MXNET_ROOT}
  make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
  ```
  3.4 Install the MXNet Python binding by

  ***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
  ```
  cd python
  sudo python setup.py install
  ```
  3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)/mxnet`, and modify `MXNET_VERSION` in `./experiments/relation_rcnn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.

4. Make sure the correct cuda is on your `LD_LIBRARY_PATH`


## Preparation for Training & Testing

1. Please download the datasets, and use the following structure:
    
    1.1 [MSCOCO 2017 (18 + 1 + 6 + .241 GB)](http://cocodataset.org/#download)
    ```
    ./data/coco/
    ```
    
    1.2 [ImageNetDET 2015 (47 + .015 + .0014 GB)](http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php) (*unchanged from 2014 data*) and [ImageNetVID 2015 (86GB)](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid)
    ```
    ./data/ILSVRC2015/
    ./data/ILSVRC2015/Annotations/DET
    ./data/ILSVRC2015/Annotations/VID
    ./data/ILSVRC2015/Data/DET
    ./data/ILSVRC2015/Data/VID
    ./data/ILSVRC2015/ImageSets
    ```   
   
    1.3 [PascalVOC 2012 (2 GB)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
    ```
    ./data/VOCdevkit/VOC2012/
    ```

2. Please download ImageNet-pretrained ResNet-v1-101 backbone model and Faster RCNN ResNet-v1-101 model manually from [Relation Backbone OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpCxvNTMZDlcDTpSA), and put it under folder `./model/relation/pretrained_model`. Make sure it looks like this:
    ```
    ./models/backbones/resnet_v1_101-0000.params
    ```

    We use a pretrained Faster RCNN and fix its params when training Faster RCNN with Learn NMS head. If you are trying to conduct such experiments, please also include the pretrained Faster RCNN model from OneDrive, making sure it looks like this:

    ```
    ./models/relation/pretrained/coco_resnet_v1_101_rcnn-0008.params
    ```

3. For FPN related experiments, we use proposals generated by a pretrained RPN to speed up our experiments. Please download the proposals from [Proposals OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpEnDg8s4FH33zh8g) and put it under folder `./proposal/resnet_v1_101_fpn/rpn_data`. Make sure it looks like this:

   ```
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_minival2014_rpn.pkl
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_train2014_rpn.pkl
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_valminusminival2014_rpn.pkl
   ```

4. Download the FGFA Flying-Chairs pre-trained backbone FlowNet model from [FGFA Backbone OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMOBdCBiNaKbcjPrA), and make sure it looks like this:
    ```
    ./models/backbones/flownet-0000.params
    ```
    You can delete the `resnet_v1_101-0000.params` downloaded here as it is a duplicate that we downloaded in step 2.

    
## Demo Models

Provided are trained models for each of the problems.

### Relation Networks

1. To try out our pre-trained relation network models, please download manually from [Relation PreTrained OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpD-UHVYNbj25lU0w), and make sure it looks like this:
	```
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_learn_nms_3epoch/train2014_valminusminival2014/rcnn_coco-0003.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_dcn_end2end_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_dcn_end2end_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_fpn_8epoch/train2014_valminusminival2014/rcnn_fpn_coco-0008.params
	./models/relation/pretrained/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_fpn_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_fpn_coco-0008.params
	```
2. To run the Faster RCNN with Relation Module and Learn NMS model, run
	```
	python experiments/relation_rcnn/rcnn_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch.yaml --ignore_cache
	```
	If you want to try other models, just change the config files. There are ten config files in `./experiments/relation_rcnn/cfg` folder, eight of which are provided with pretrained models.


### FGFA

1.  Download the trained FGFA model (on ImageNet DET + VID train) from [FGFA PreTrained OneDrive](https://1drv.ms/u/s!AqfHNsil2nOiiwDiKev7DB6L9ay7), and make sure it looks like this:
	```
	./models/fgfa/pretrained/rfcn_fgfa_flownet_vid-0000.params
    ```
    TODO: put this into output directory
    
2.  Run
	```
	python ./fgfa_rfcn/demo.py
	```
	
### Deformable

1. To use the demo with the pre-trained deformable models, please download manually from [Deformable PreTrained OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMSjehIcCgAhvEAHw) or [BaiduYun](https://pan.baidu.com/s/1dFlPFED), and put it under folder `model/`.

	Make sure it looks like this:
	```
	./models/deformable/pretrained/rfcn_dcn_coco-0000.params
	./models/deformable/pretrained/rfcn_coco-0000.params
	./models/deformable/pretrained/fpn_dcn_coco-0000.params
	./models/deformable/pretrained/fpn_coco-0000.params
	./models/deformable/pretrained/rcnn_dcn_coco-0000.params
	./models/deformable/pretrained/rcnn_coco-0000.params
	./models/deformable/pretrained/deeplab_dcn_cityscapes-0000.params
	./models/deformable/pretrained/deeplab_cityscapes-0000.params
	./models/deformable/pretrained/deform_conv-0000.params
	./models/deformable/pretrained/deform_psroi-0000.params
	```
	
2. To run the R-FCN demo, run
	```
	python ./rfcn/demo.py --rfcn_only
	```
	
3. To visualize the offset of deformable convolution and deformable psroipooling, run
	```
	python ./rfcn/deform_conv_demo.py
	python ./rfcn/deform_psroi_demo.py
	```
	
	
## Usage

1. All of the experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/../cfgs`.

2. To perform experiments, run the python scripts with the corresponding config file as input. For example
 
    2.1 to train and test Faster RCNN with Relation Module and Learn NMS(e2e), use the following command:
    ```
    python experiments/relation_rcnn/rcnn_end2end_train_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `models/relation/output/rcnn/`.

    The rcnn_end2end_train_test.py script is for Faster RCNN and Deformable Faster RCNN experiments that train RPN together with RCNN. To train and test FPN which use previously generated proposals, use the following command:

    ```
    python experiments/relation_rcnn/rcnn_train_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_fpn_relation_learn_nms_8epoch.yaml
    ```

    2.2 To train and test FGFA with R-FCN, use the following command
    ```
    python experiments/fgfa_rfcn/fgfa_rfcn_end2end_train_test.py --cfg experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `models/fgfa/output/fgfa_rfcn/imagenet_vid/`.
    
    2.3 To perform experiments with just deformable nets, run the python scripts with the corresponding config file as input. For example, to train and test deformable convnets on COCO with ResNet-v1-101, use the following command
    ```
    python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/resnet_v1_101_coco_trainval_rfcn_dcn_end2end_ohem.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `models/deformable/output/rfcn_dcn_coco/`.
    
3. Please find more details in config files and in the code.


## FAQ

Q: I encounter `segment fault` at the beginning.

A: A compatibility issue has been identified between MXNet and opencv-python 3.0+. We suggest that you always `import cv2` first before `import mxnet` in the entry script. 

<br/>

