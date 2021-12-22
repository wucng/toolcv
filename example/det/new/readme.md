[toc]

# 说明
```python
数据：FruitsNuts
https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip

共18张 classes = ['date', 'fig', 'hazelnut'] train_ratio=0.8
取出其中14张 训练，剩下4张  验证
```
# 验证结果

## yolo_ms

|  model  | mAP@0.5 | mAP@0.50:0.95 | 备注                                                         |                             其他                             |
| :-----: | :-----: | :-----------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| yolov5  |  0.987  |     0.717     | !python train.py --data FruitsNuts.yaml --cfg ./models/yolov5s.yaml --weights ./yolov5s.pt --batch-size 4 --workers 1 --epochs 50 --imgsz 640 --freeze 10 --cache --adam | 10.51it/s [here](https://www.kaggle.com/fengzhongyouxia/det-fruitsnuts/notebook) |
| yolo_ms |  0.900  |     0.459     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True; |                           5.04it/s                           |
| yolo_ms |  0.877  |     0.500     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True; |                           5.04it/s                           |
| yolo_ms |  0.814  |     0.399     | object_target[batch_id, index_ * 1, _cy, _cx] = value * iou                                  use_amp = False;strides = [8, 16, 32];**ignore = True**;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True |                           2.80it/s                           |
| yolo_ms |  0.861  |     0.469     | object_target[batch_id, index_ * 1, _cy, _cx] = value * iou                                  use_amp = False;strides = [8, 16, 32];**ignore = True**;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True |                           2.80it/s                           |
| yolo_ms |  0.744  |     0.380     | **object_target[batch_id, index_ * 1, _cy, _cx] = -1**                       use_amp = False;strides = [8, 16, 32];**ignore = True**;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True |                           2.89it/s                           |
| yolo_ms |  0.915  |     0.508     | **object_target[batch_id, index_ * 1, _cy, _cx] = -1**                       use_amp = False;strides = [8, 16, 32];**ignore = True**;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=5;norm_eval=True |                           2.89it/s                           |
| yolo_ms |  0.202  |     0.077     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=2;norm_eval=True; |                           4.65it/s                           |
| yolo_ms |  0.338  |     0.111     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;lr = 3e-3;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';freeze_at=2;norm_eval=True; |                           4.65it/s                           |
| yolo_ms |  0.905  |     0.479     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; |                           4.65it/s                           |
| yolo_ms |  0.878  |     0.544     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; |                           4.65it/s                           |
| yolo_ms |  0.930  |     0.471     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; **level=1** |                           4.47it/s                           |
| <font color=#FF0000>yolo_ms</font> |  0.980  |     0.564     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; **level=1** |                           4.47it/s                           |
| yolo_ms |  0.902  |     0.506     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;**norm_eval=False**; **level=1** |                           4.47it/s                           |
| <font color=#FF0000>yolo_ms</font> |  0.967  |     0.575     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;**norm_eval=False**; **level=1** |                           4.47it/s                           |
| yolo_ms | 0.708 | 0.411 | use_amp = False;strides = [8, 16, 32];ignore = False;**threds = 0.3**;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; **level=1** | 4.29it/s |
| yolo_ms | 0.834 | 0.515 | use_amp = False;strides = [8, 16, 32];ignore = False;**threds = 0.3**;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;'resnet18';**freeze_at=2**;norm_eval=True; **level=1** | 4.29it/s |
| yolo_ms |  0.932  |     0.411     | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 3.57it/s |
| <font color=#FF0000>yolo_ms</font> | 0.992 | 0.616 | use_amp = False;strides = [8, 16, 32];ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 3.57it/s |
| yolo_ms | 0.944 | 0.509 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 3.45it/s |
| <font color=#FF0000>yolo_ms</font> | 0.980 | 0.636 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 3.45it/s |
| <font color=#FF0000>yolo_ms</font> | 1.000 | 0.654 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 150**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 3.45it/s |
| yolo_ms | 0.900 | 0.551 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 2.54it/s |
| yolo_ms | 0.965 | 0.594 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 2.54it/s |
| <font color=#FF0000>yolo_ms</font> | 1.000 | 0.660 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 150**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 2.54it/s |
| <font color=#FF0000>yolo_ms</font> | 1.000 | 0.670 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 200**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;norm_eval=True; **level=1** | 2.54it/s |
| yolo_ms | 1.000 | 0.690 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.50it/s |
| <font color=#FF0000>yolo_ms</font> | 1.000 | **0.718** | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.50it/s |
| yolo_ms | 0.827 | 0.531 | use_amp = False;**strides = [8, 16, 32]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.60it/s |
| yolo_ms | 0.967 | 0.675 | use_amp = False;**strides = [8, 16, 32]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.60it/s |
| <font color=#FF0000>yolo_ms</font> | 0.994 | 0.675 | use_amp = False;**strides = [8, 16, 32]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 150**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.60it/s |
| yolo_ms | 0.824 | 0.475 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;**'resnet18'**;**freeze_at=2**;norm_eval=True; **level=1** | 4.20it/s |
| yolo_ms | 0.965 | 0.591 | use_amp = False;**strides = [8, 16, 32,64,128]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;**'resnet18'**;**freeze_at=2**;norm_eval=True; **level=1** | 4.20it/s |
| yolo_ms | 0.425 | 0.230 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 1.58it/s |
| yolo_ms | 0.736 | 0.477 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 1.58it/s |
| yolo_ms | 0.528 | 0.215 | use_amp = False;**strides = [16]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.87it/s |
| **yolo_ms** | 0.946 | 0.600 | use_amp = False;**strides = [16]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.87it/s |
| yolo_ms | 0.603 | 0.300 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 416, 416**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.48it/s |
| **yolo_ms** | 0.904 | 0.577 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 416, 416**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** | 2.48it/s |
| yolo_ms | 0.585 | 0.242 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;**h, w = 416, 416**;**'cspdarknet53'**;**freeze_at=5**;**norm_eval=True**; **level=0** | 3.82it/s |
| yolo_ms | 0.697 | 0.273 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;**h, w = 416, 416**;**'cspdarknet53'**;**freeze_at=5**;**norm_eval=True**; **level=0** | 3.82it/s |
| yolo_ms | 0.442 | 0.159 | use_amp = False;**strides = [16]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=5**;norm_eval=True; **level=1** | 4.81it/s |
| yolo_ms | 0.698 | 0.319 | use_amp = False;**strides = [16]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=5**;norm_eval=True; **level=1** | 4.81it/s |
| yolo_ms | 0.506 | 0.266 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 50**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=5**;norm_eval=True; **level=1** | 3.56it/s |
| yolo_ms | 0.827 | 0.420 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 100**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=5**;norm_eval=True; **level=1** | 3.56it/s |
| yolo_ms | 0.933 | 0.474 | use_amp = False;**strides = [8]**;ignore = False;threds = 0.4;accumulate = 2;**lr = 1e-3**;**epochs = 150**;batch_size = 4;h, w = 416, 416;**'cspdarknet53'**;**freeze_at=5**;norm_eval=True; **level=1** | 3.56it/s |

```python
# 推荐 参数
resize = (640,640)
lr = 1e-3
strides = [8,16,32] or [8,16,32,64,128]
threds = 0.4;ignore = False
model_name = 'cspdarknet53';freeze_at=2;norm_eval=False;level=1
```



## yolo_self

|   model   | mAP@0.5 | mAP@0.50:0.95 | 备注                                                         |                             其他                             |
| :-------: | :-----: | :-----------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
|  yolov5   |  0.987  |     0.717     | !python train.py --data FruitsNuts.yaml --cfg ./models/yolov5s.yaml --weights ./yolov5s.pt --batch-size 4 --workers 1 --epochs 50 --imgsz 640 --freeze 10 --cache --adam | 10.51it/s [here](https://www.kaggle.com/fengzhongyouxia/det-fruitsnuts/notebook) |
| yolo_self |  0.195  |     0.116     | use_amp = False;**strides = 8**;threds = 0.3;accumulate = 2;**lr = 1e-3**; **epochs = 50**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** |                           1.64it/s                           |
| yolo_self |  0.406  |     0.262     | use_amp = False;**strides = 8**;threds = 0.3;accumulate = 2;**lr = 1e-3**; **epochs = 100**;batch_size = 4;**h, w = 640, 640**;**'cspdarknet53'**;**freeze_at=2**;**norm_eval=False**; **level=1** |                           1.64it/s                           |

