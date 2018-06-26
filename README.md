# Training COCO 2017 Object Detection and Segmentation via Learning Feature Pyramids

The code is provided by [Guangrun Wang](https://wanggrun.github.io/).

Sun Yat-sen University (SYSU)

### Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Citation](#citation)

## Introduction

This repository contains the training & testing code on [COCO 2017](http://cocodataset.org/#home) object detection and instance segmentation via learning feature pyramids (LFP). LFP is originally used for human pose machine, described in the paper "Learning Feature Pyramids for Human Pose Estimation" (https://arxiv.org/abs/1708.01101). We extend it to the object detection and instance segmentation.


## Results

These models are trained on COCO 2017 training set and evaluated on COCO 2017 validation set.
MaskRCNN results contain both bbox and segm mAP. 

+ COCO Object Detection

|Method|`MASKRCNN_BATCH`|resolution |schedule| AP bbox | AP bbox 50 | AP bbox 75
|   -    |    -         |    -      |   -    |   -     |   -        |   -       |
|ResNet50      |512     |(800, 1333)|360k    |37.7     |   57.9     |   40.9    |
|Ours  |512             |(800, 1333)|360k    |39.8     |   60.2     |   43.4    |


+ COCO Instance Segmentation


|Method|`MASKRCNN_BATCH`|resolution |schedule| AP mask | AP mask 50 | AP mask 75
|   -    |    -         |    -      |   -    |   -     |   -        |   -       |
|ResNet50      |512     |(800, 1333)|360k    |32.8     |   54.3     |   34.7    |
|Ours  |512             |(800, 1333)|360k    |34.6     |   56.7     |   36.8    |

The schemes have the same configuration __and mAP__
as the `R50-C4-2x` entries in
[Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#end-to-end-faster--mask-r-cnn-baselines).

## Usage


+ The model is first pretrained on the ImageNet-1K, where the training scripts can be found [Guangrun Wang's github](https://github.com/wanggrun/Learning-Feature-Pyramids/blob/master/README.md). We also provide the trained ImageNet models as follows.

   [Baidu Pan](https://wanggrun.github.io), code: 269o

   [Google Drive](https://wanggrun.github.io)


+ Training script for COCO object detection and instance segmentation:
```
python3 train.py --load /home/grwang/seg/train_log_resnet50/imagenet-resnet-d50/model-510000    --gpu 0,1,2,3,4,5,6,7   --logdir mask-pyramid-train
```

+ Testing script for COCO object detection and instance segmentation:
```
python3 train.py  --evaluate output.json  --load /home/grwang/seg/train_log_resnet50/imagenet-resnet-d50/model-510000    --gpu 0,1,2,3,4,5,6,7   --logdir mask-pyramid-test
```


+ Trained Models of COCO:

   Model trained for evaluation on COCO 2017 object detection and instance segmentation task:

   [Baidu Pan](https://wanggun.github.io), code: 7dl0

   [Google Drive](https://wanggrun.github.io)


## Citation

If you use these models in your research, please cite:

	@inproceedings{yang2017learning,
            title={Learning feature pyramids for human pose estimation},
            author={Yang, Wei and Li, Shuang and Ouyang, Wanli and Li, Hongsheng and Wang, Xiaogang},
            booktitle={The IEEE International Conference on Computer Vision (ICCV)},
            volume={2},
            year={2017}
        }


## Dependencies
+ Python 3; TensorFlow >= 1.4.0 (>=1.6.0 recommended due to a TF bug);
+ [pycocotools](https://github.com/pdollar/coco/tree/master/PythonAPI/pycocotools), OpenCV.
+ Pre-trained ImageNet model.
+ COCO data. It needs to have the following directory structure:
```
DIR/
  annotations/
    instances_train2014.json
    instances_val2014.json
    instances_minival2014.json
    instances_valminusminival2014.json
  train2014/
    COCO_train2014_*.jpg
  val2014/
    COCO_val2014_*.jpg
```
`minival` and `valminusminival` can be download from
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

+ [Tensorpack](https://github.com/ppwwyyxx/tensorpack)
   The code depends on Yuxin Wu's Tensorpack. For convenience, we provide a stable version 'tensorpack-installed' in this repository. 
   ```
   # install tensorpack locally:
   cd tensorpack-installed
   python setup.py install --user
   ```

