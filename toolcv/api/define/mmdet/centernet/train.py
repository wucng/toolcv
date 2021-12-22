"""
## Results and models

| Backbone        | DCN |  Mem (GB) | Box AP | Flip box AP| Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :----: |
| ResNet-18 | N | 3.45 | 25.9 | 27.3 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_140e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630.log.json) |
| ResNet-18 | Y | 3.47 | 29.5 | 30.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131.log.json) |

# ------------------------------------------------------------------------------------------------------------
!wget https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_140e_coco.py
!wget https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth

!wget https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py
!wget https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth

"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math

from toolcv.api.define.utils.model.mmdet import CenterNet
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.data.dataset import glob_format

anchors = None
strides = 4
use_amp = False
accumulate = 1
gradient_clip_val = 0.0
lrf = 0.1
lr = 5e-4
weight_decay = 5e-5
epochs = 50
batch_size = 8
resize = (512, 512)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
config = r"D:\zyy\git\mmdetection\configs\centernet\centernet_resnet18_dcnv2_140e_coco.py"
checkpoint = r'D:\zyy\git\mmdetection\checkpoints\centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'
num_classes = len(classes)
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# data
transforms = Compose([RandomHorizontalFlip(), Resize(*resize), ToTensor(), Normalize()])
dataset = FruitsNutsDataset(dir_data, classes, transforms, 0)
# dataset.show(mode='pil')
dataloader = LoadDataloader(dataset, None, 0.1, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# model
model = CenterNet(None, config, checkpoint, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, 1000,
                  0.5, None, None, use_amp,
                  accumulate, gradient_clip_val, device, None, train_dataLoader, val_dataLoader)
model.model.to(device)

# model.fit(model.model)

transforms = Compose([Resize(*resize), ToTensor(), Normalize()])
model.predict(model.model, glob_format(dir_data), transforms, device, visual=False,conf_threshold=0.1)
