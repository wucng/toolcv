# 自定义版
```py
!pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
!pip install openmim
!mim install mmdet
```


# 数据
- [FruitsNutsDataset](https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip)

# 推荐
- [x] <font color=#FF0000>yolo_self、yolo_self_new、yolo_self_new_v2、yolo_self_new_v3、yolo_self_new_v4、yolo_self_new_v4_2、
yolov1、yolov1_yolof、yolov2_center_yolof、yolov3、yolov3_1、yolov3_center、yolo_self_mask</font>
- [x] `do_bbox_offset = False candidate = np.linspace(0.01, 1.0, 100) epochs=100 yolo_self_new_v4_2 mAP(0.5,0.75)=0.990`
- [x] `do_bbox_offset = False candidate = np.linspace(0.01, 1.0, 40) epochs=100 yolo_self_new_v4_2 mAP(0.5)=0.966 mAP(0.75)=0.838`
- [x] <font color=#FF0000>centernet、centernet_ms、fcos_ms</font>
- [x] <font color=#FF0000>ssd_ss、ssd_ms</font>
- [x] <font color=#FF0000>fasterrcnn_simple、fasterrcnn_simple_new、maskrcnn、maskrcnn_v2、unet_segm</font>


# yolo
|      |   mAP(0.5) |  mAP(0.75) | epochs | batch_size | speed(it/s) | size | stride | model |
| ---- | ---- |---- | ---- |---- | ---- | ---- | ---- | ---- |
|   [yolo_self_keypoints](./example/detecte/yolo/yolo_self_keypoints.py)   |  0.957(keypoints 0.418)    |   0.871(keypoints 0.000)   |   100   |   4   |   2.2  |    512x512  |    16  |    resnet50  |
|   [yolo_self_keypointsv2](./example/detecte/yolo/yolo_self_keypointsv2.py)   |  0.720(keypoints 0.308)    |   0.055(keypoints 0.020)   |   100   |   4   |   6.3  |    416x416  |    16  |    dla34  |
|   [yolo_self_mask](./example/detecte/yolo/yolo_self_mask.py)   |  0.989(mask 0.989)    |   0.989(mask 0.728)   |   100   |   4   |   4.0  |    416x416  |    16  |    resnet18  |
|   [yolo_self_mask](./example/detecte/yolo/yolo_self_mask.py)   |  0.989(mask 0.989)    |   0.981(mask 0.854)   |   150   |   4   |   4.0  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self</font>](./example/detecte/yolo/yolo_self.py)   |  0.983    |   0.852   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self</font>](./example/detecte/yolo/yolo_self.py)   |  0.986    |   0.591   |   50   |   4   |    5.0  |    416x416  |    8  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new</font>](./example/detecte/yolo/yolo_self_new.py)   |  0.973    |   0.793   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v2</font>](./example/detecte/yolo/yolo_self_new_v2.py)   |  0.816    |   0.490   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v2</font>](./example/detecte/yolo/yolo_self_new_v2.py)   |  0.986    |   0.876   |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v3</font>](./example/detecte/yolo/yolo_self_new_v3.py)   |  0.654    |   0.201   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v3</font>](./example/detecte/yolo/yolo_self_new_v3.py)   |  0.864    |   0.551   |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v3</font>](./example/detecte/yolo/yolo_self_new_v3.py)   |  0.936    |   0.816   |   150   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v4</font>](./example/detecte/yolo/yolo_self_new_v4.py)   |  0.653    |   0.059   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v4</font>](./example/detecte/yolo/yolo_self_new_v4.py)   |  0.964    |   0.411   |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v4_2</font>](./example/detecte/yolo/yolo_self_new_v4_2.py)   |  0.299    |   0.288   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v4_2</font>](./example/detecte/yolo/yolo_self_new_v4_2.py)   |  0.966    |   0.838   |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolo_self_new_v4_2</font>](./example/detecte/yolo/yolo_self_new_v4_2.py)    |  0.990    |   0.990   |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [yolov1](./example/yolo/yolov1.py)   |  0.494    |   0.087   |   50   |   4   |    8  |    416x416  |    32  |    resnet18  |
|   [<font color=#FF0000>yolov1</font>](./example/detecte/yolo/yolov1.py)   |  0.985    |   0.482   |   50   |   4   |    6.1  |    416x416  |    16  |    dla34  |
|   [<font color=#FF0000>yolov1</font>](./example/detecte/yolo/yolov1.py)   |  0.992    |   0.838   |   50   |   4   |    3.6  |    416x416  |    8   |    resnet34  |
|   [yolov1_2](./example/detecte/yolo/yolov1_2.py) |  0.029    |   0.005   |   50   |   4   |    2.2  |    416x416  |    4  |    resnet18  |
|   [yolov1_yolof](./example/detecte/yolo/yolov1_yolof.py)   | 0.821    |   0.756   |   150   |   4   |    7  |    416x416  |    32  |    resnet18  |
|   [<font color=#FF0000>yolov1_yolof</font>](./example/detecte/yolo/yolov1_yolof.py)   | 0.983    |   0.593   |   50   |   4   |    6  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolov1_yolof</font>](./example/detecte/yolo/yolov1_yolof.py)   | 0.991    |   0.920   |   100   |   4   |    4.6  |    416x416  |    8  |    resnet18  |
|   [yolov2_center_yolof](./example/detecte/yolo/yolov2_center_yolof.py)   | 0.886    |   0.846   |   100   |   4   |    5.6  |    416x416  |    32  |    resnet18  |
|   [<font color=#FF0000>yolov2_center_yolof</font>](./example/detecte/yolo/yolov2_center_yolof.py)   | 0.914    |   0.771   |   50   |   4   |    5.2  |    416x416  |    16  |    resnet18  |
|   [<font color=#FF0000>yolov2_center_yolof</font>](./example/detecte/yolo/yolov2_center_yolof.py)   | 0.983    |   0.862   |   50   |   4   |    4.5  |    416x416  |    8  |    resnet18  |
|   [<font color=#FF0000>yolov2_center_yolof</font>](./example/detecte/yolo/yolov2_center_yolof.py)   | 0.990    |   0.706   |   50   |   4   |    2.8  |    416x416  |    4  |    resnet18  |
|   [<font color=#FF0000>yolov3</font>](./example/detecte/yolo/yolov3.py)   | 0.993    |   0.826   |   50   |   4   |    6.3  |    416x416  |    8,16,32  |    resnet18  |
|   [yolov3](./example/detecte/yolo/yolov3.py)   | 0.992    |   0.963   |   100   |   4   |    6.3  |    416x416  |    8,16,32  |    resnet18  |
|   [yolov3](./example/detecte/yolo/yolov3.py)   | 0.956    |   0.768   |   50   |   4   |    6  |    416x416  |    8,16,32  |    dla34  |
|   [yolov3](./example/detecte/yolo/yolov3.py)   | 0.979    |   0.767   |   50   |   4   |    5.8  |    416x416  |    8,16,32  |    resnet34  |
|   [<font color=#FF0000>yolov3_1</font>](./example/detecte/yolo/yolov3_1.py)   | 0.991    |   0.765   |   50   |   4   |    6.8  |    416x416  |    8,16,32  |    resnet18  |
|   [yolov3_1](./example/detecte/yolo/yolov3_1.py)   | 0.990    |   0.902   |   100   |   4   |    6.8  |    416x416  |    8,16,32  |    resnet18  |
|   [<font color=#FF0000>yolov3_1</font>](./example/detecte/yolo/yolov3_1.py)   | 0.979    |   0.753   |   50   |   4   |    6.3  |    416x416  |    8,16,32  |    resnet34  |
|   [yolov3_1](./example/detecte/yolo/yolov3_1.py)   | 0.991    |   0.845   |   100   |   4   |    6.3  |    416x416  |    8,16,32  |    resnet34  |
|   [<font color=#FF0000>yolov3_center</font>](./example/detecte/yolo/yolov3_center.py)   | 0.992    |   0.881   |   50   |   4   |    5.5  |    416x416  |    8,16,32  |    resnet18  |
|   [yolov3_center](./example/detecte/yolo/yolov3_center.py)   | 0.980    |   0.895  |   50   |   4   |    5.2  |    416x416  |    8,16,32  |    resnet34  |

# yolo（自定义backbon）
|      |   mAP(0.5) |  mAP(0.75) | epochs | batch_size | speed(it/s) | size | stride | model |
| ---- | ---- |---- | ---- |---- | ---- | ---- | ---- | ---- |
|   [classify_yolo](./example/classify/classify_yolo.py) 分类  | mPrecision=0.987    |   mRecall=0.985  |   50   |   4   |    2.2  |    416x416  |    16  |    resnet18  |
|   [yolo_self](./example/classify/yolo_self.py)   | 0.954    |   0.713  |   100   |   4   |    5.2  |    416x416  |    16  |    resnet18  |


# centernet与fcos

|      |   mAP(0.5) |  mAP(0.75) | epochs | batch_size | speed(it/s) | size | stride | model |
| ---- | ---- |---- | ---- |---- | ---- | ---- | ---- | ---- |
|   [centernet](./example/detecte/centernet/centernet.py)   |  0.529    |   0.133   |   50   |   4   |    6.0  |    512x512  |    4  |    resnet18  |
|   [centernet](./example/detecte/centernet/centernet.py)   |  0.992    |   0.425   |   100   |   4   |    6.0  |    512x512  |    4  |    resnet18  |
|   [<font color=#FF0000>centernet</font>](./example/detecte/centernet/centernet.py)   |  0.990    |   0.906   |   200   |   4   |    6.0  |    512x512  |    4  |    resnet18  |
|   [centernet_ms](./example/detecte/centernet/centernet_ms.py)   |  0.300    |   0.110   |   50   |   4   |    3.6  |    512x512  |   8,16,32,64,128  |    dla34  |
|   [centernet_ms](./example/detecte/centernet/centernet_ms.py)   |  0.833    |   0.410   |   100   |   4   |    3.6  |    512x512  |   8,16,32,64,128  |    dla34  |
|   [<font color=#FF0000>centernet_ms</font>](./example/detecte/centernet/centernet_ms.py)   |  0.979    |   0.889   |   200   |   4   |    3.6  |    512x512  |   8,16,32,64,128  |    dla34  |
|   [centernet_v2](./example/detecte/centernet/centernet_v2.py)   |  0.760    |   0.179   |   50   |   4   |    5.2  |    512x512  |   4  |    resnet18  |
|   [centernet_v2](./example/detecte/centernet/centernet_v2.py)   |  0.762    |   0.430   |   100   |   4   |    5.2  |    512x512  |   4  |    resnet18  |
|   [centernet_v2](./example/detecte/centernet/centernet_v2.py)   |  0.749    |   0.540   |   150   |   4   |    5.2  |    512x512  |   4  |    resnet18  |
|   [centernet_yolov1](./example/detecte/centernet/centernet_yolov1.py)   |  0.515    |  0.166   |   50   |   4   |    3  |    512x512  |   4  |    resnet18  |
|   [centernet_yolov1](./example/detecte/centernet/centernet_yolov1.py)   |  0.992    |  0.462   |   100   |   4   |    3  |    512x512  |   4  |    resnet18  |
|   [fcos_ms](./example/detecte/fcos/fcos_ms.py)   |  0.406    |  0.043   |   50   |   4   |    2.5  |    512x512  |   8,16,32,64,128  |    dla34  |
|   [<font color=#FF0000>fcos_ms</font>](./example/detecte/fcos/fcos_ms.py)   |  0.887    |  0.273   |   100   |   4   |    2.5  |    512x512  |   8,16,32,64,128  |    dla34  |
|   [<font color=#FF0000>fcos_ms</font>](./example/detecte/fcos/fcos_ms.py)   |  0.947    |  0.632   |   250   |   4   |    2.5  |    512x512  |   8,16,32,64,128  |    dla34  |


# retinanet
|      |   mAP(0.5) |  mAP(0.75) | epochs | batch_size | speed(it/s) | size | stride | model |
| ---- | ---- |---- | ---- |---- | ---- | ---- | ---- | ---- | 
|   [retinanet_fixsize](./example/detecte/retinanet/retinanet_fixsize.py)   |  0.615    |   0.396   |   50   |   4   |    2.9  |    512x512  |  8,16,32,64,128 |    dla34  |
|   [retinanet_fixsize](./example/detecte/retinanet/retinanet_fixsize.py)   |  0.698    |   0.544   |   100   |   4   |    2.9  |    512x512  |  8,16,32,64,128 |    dla34  |
|   [retinanet_fixsize](./example/detecte/retinanet/retinanet_fixsize.py)   |  0.666    |   0.532   |   150   |   4   |    2.9  |    512x512  |  8,16,32,64,128 |    dla34  |
|   [retinanet_minmax](./example/detecte/retinanet/retinanet_minmax.py)   |  0.673    |   0.007   |   50   |   4   |    2.0  |    800,1333  |  8,16,32,64,128 |    dla34  |

# ssd、retinanet、fasterrcnn

|  |mAP(0.5) | mAP(0.75) | epochs | batch_size | speed(it/s) | size | stride | model |
| ---- | ---- |---- | ---- |---- | ---- | ---- | ---- | ---- |
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py)  | 0.921 |0.566  |50  |4  | 7.5 |300x300  | 16  |resnet18|
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py) | 0.968 |0.930 |100 |4 | 7.5 |300x300 | 16 |resnet18|
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py) | 0.745 |0.639 |50 |4 | 6.0 |300x300 | 8 |resnet18|
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py) | 0.778 |0.739 |100 |4 | 6.0 |300x300 | 8 |resnet18|
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py) | 0.446 |0.369 |50 |4 | 4.2 |300x300 | 4 |resnet18|
|[<font color=#FF0000>ssd_ss</font>](./example/detecte/ssd/ssd_ss.py) | 0.477 |0.431 |100 |4 | 4.2 |300x300 | 4 |resnet18|
|[<font color=#FF0000>ssd_ms</font>](./example/detecte/ssd/ssd_ms.py) RetinaHeadV2 | 0.599 |0.131 |50 |4 | 4.6 |320x320 | 8,16,32,64,128 |resnet18|
|[<font color=#FF0000>ssd_ms</font>](./example/detecte/ssd/ssd_ms.py) RetinaHeadV2 | 0.963 |0.679 |100 |4 | 4.6 |320x320 | 8,16,32,64,128 |resnet18|
|[<font color=#FF0000>ssd_ms</font>](./example/detecte/ssd/ssd_ms.py) RetinaHead | 0.978 |0.904 |100 |4 | 4.6 |320x320 | 8,16,32,64,128 |resnet18|
|[<font color=#FF0000>ssd_ms_v2</font>](./example/detecte/ssd/ssd_ms_v2.py) RetinaHead | 0.629 |0.410 |50 |4 | 4.8 |320x320 | 8,16,32 |resnet18|
|[<font color=#FF0000>ssd_ms_v2</font>](./example/detecte/ssd/ssd_ms_v2.py) RetinaHead | 0.835 |0.773 |100 |4 | 4.8 |320x320 | 8,16,32 |resnet18|
|[<font color=#FF0000>ssd_ms_v2</font>](./example/detecte/ssd/ssd_ms_v2.py) RetinaHeadV2 | 0.804 |0.782 |100 |4 | 4.8 |320x320 | 8,16,32 |resnet18|
|[ssd300.py](./example/detecte/ssd/ssd300.py)  |  0.739|0.533  |50  |4  | 6 |300x300  | 8,16,32,64,100,300  |vgg16|
|[ssd300.py](./example/detecte/ssd/ssd300.py)  |  0.801|0.703  |100  |4  | 6 |300x300  | 8,16,32,64,100,300  |vgg16|
|[ssd300_v2.py](./example/detecte/ssd/ssd300_v2.py)  |  0.806|0.659  |50|4  | 5.5 |300x300  | 8,16,32,64,100,300  |vgg16|
|[ssd300_v2.py](./example/detecte/ssd/ssd300_v2.py)  |  0.814|0.680  |100|4  | 5.5 |300x300  | 8,16,32,64,100,300  |vgg16|
|[ssd300_v3.py](./example/detecte/ssd/ssd300_v3.py)  |  0.812|0.662  |50|4  | 5.6 |300x300  | 8,16,32,64,100,300  |vgg16|
|[ssd512.py](./example/detecte/ssd/ssd512.py)  |  0.893|0.537  |50|4  | 3.5 |512x512  | 8,16,32,64,128,256,512  |vgg16|
|[ssd512.py](./example/detecte/ssd/ssd512.py)  |  0.869|0.688  |100|4  | 3.5 |512x512  | 8,16,32,64,128,256,512  |vgg16|
|[ssd512_v2.py](./example/detecte/ssd/ssd512_v2.py)  |  0.895|0.686  |50|4  | 3.5 |512x512  | 8,16,32,64,128,256,512  |vgg16|
|[retinanet_fixsize.py](./example/detecte/retinanet/retinanet_fixsize.py)  |  0.746|0.364|50|4  | 2.3 |512x512  | 8,16,32,64,128  |dla34|
|[retinanet_fixsize.py](./example/detecte/retinanet/retinanet_fixsize.py)  |  0.710|0.513|100|4  | 2.3 |512x512  | 8,16,32,64,128  |dla34|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py)|  0.844|0.637|50|4  | 5.0 |512x512  | 16  |resnet18|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py)|  0.923|0.735|100|4  | 5.0 |512x512  | 16  |resnet18|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py)|  0.947|0.832|150|4  | 5.0 |512x512  | 16  |resnet18|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py) freeze_at=3|  0.989|0.952|200|4  | 3.0 |512x512  | 16  |resnet18|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py) freeze_at=3|  0.989|0.940|50|4  | 3.0 |512x512  | 16  |resnet18|
|[fasterrcnn_simple.py](./example/detecte/fasterrcnn/fasterrcnn_simple.py) freeze_at=3|  0.990|0.970|100|4  | 3.0 |512x512  | 16  |resnet18|
|[fasterrcnn.py](./example/detecte/fasterrcnn/fasterrcnn.py) (差)  |  0.803|0.099  |50|4  | 3.2 |512x512  | 16  |resnet18|
|[fasterrcnn.py](./example/detecte/fasterrcnn/fasterrcnn.py) (差) |  0.869|0.076  |100|4  | 3.2 |512x512  | 16  |resnet18|
|[fasterrcnn.py](./example/detecte/fasterrcnn/fasterrcnn.py) freeze_at=4 (差)|  0.617|0.151  |150|4  | 2.9 |512x512  | 16  |resnet18|
|[fasterrcnn.py](./example/detecte/fasterrcnn/fasterrcnn.py) freeze_at=3 (差) |  0.662|0.183  |200|4  | 2.9 |512x512  | 16  |resnet18|
|[maskrcnn.py](./example/detecte/fasterrcnn/maskrcnn.py) |  0.878(mask 0.902)|0.775(mask 0.891)  |50|4  | 3.2 |512x512  | 16  |resnet18|
|[maskrcnn_v2.py](./example/detecte/fasterrcnn/maskrcnn_v2.py) |  0.850(mask 0.857)|0.655(mask 0.640)  |50|4  | 3.2 |512x512  | 16  |resnet18|
|[maskrcnn_v2.py](./example/detecte/fasterrcnn/maskrcnn_v2.py)freeze_at=3 |  0.850(mask 0.857)|0.655(mask 0.640)  |50|4  | 3.2 |512x512  | 16  |resnet18|
