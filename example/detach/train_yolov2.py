"""
- https://www.kaggle.com/andrewmvd/car-plate-detection
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset  # ,TensorDataset
# from torchvision.datasets import ImageFolder,DatasetFolder
# from torchvision.models.resnet import resnet34, resnet18, resnet50
from torchvision import transforms as T
# from torchvision.ops.boxes import batched_nms, nms
# from collections import OrderedDict
import numpy as np
import cv2
import json
import random
import math

import os
from pathlib import Path
# from xml.dom.minidom import parse
# from shutil import copyfile
from glob import glob
from PIL import Image
# import matplotlib.pyplot as plt

from toolcv.data.transform import getitem_yolov2, getitem_mutilscale_yolov2
from toolcv.network.net import Yolov1simple
from toolcv.tools.utils import train_yolov2 as train, predict_yolov2 as predict
from toolcv.data.dataset import FruitsNutsDatasetYOLOV2


class FruitsNutsDataset(Dataset):
    """
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[],
                 angle=None,imagu=False,advanced=False):
        self.root = root
        self.transform = transform
        self.classes = classes
        self.classes = classes
        self.resize = resize
        self.strides = strides
        self.fw = int(resize / strides)
        self.fh = int(resize / strides)

        self.muilscale = muilscale
        self.batch_size = batch_size
        self.mode = mode
        self.use_mosaic = use_mosaic
        self.fix_resize = fix_resize
        self.anchor = anchor

        self.angle = angle
        self.imagu = imagu
        self.advanced = advanced

        self.num_anchors = num_anchors
        self.num_classes = len(classes)

        self.json_file = os.path.join(root, "trainval.json")
        with open(self.json_file) as f:
            imgs_anns = json.load(f)

        self.annotations = self.change_csv(imgs_anns)

    def __len__(self):
        return len(self.annotations)

    def load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        # to 0~1
        boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)
        return img, annotations

    def change_csv(self, imgs_anns):
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        annotations = imgs_anns["annotations"]

        # 找到每张图片所以标注，组成一个list
        result = []
        for k, v in images_dict.items():
            img_path = os.path.join(self.root, "images", v)
            boxes = []
            labels = []
            iscrowd = []
            # image_id = []
            area = []
            segment = []
            for item in annotations:
                if item["image_id"] == k:
                    segment.append(item["segmentation"])
                    iscrowd.append(item["iscrowd"])
                    area.append(item["area"])
                    # boxes.append(item["bbox"])
                    bbox = item["bbox"]
                    # boxes.append([bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
                    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    labels.append(self.classes.index(categories_dict[item["category_id"]]))

            result.append({"img_path": img_path, "segment": segment, "iscrowd": iscrowd, "area": area, "boxes": boxes,
                           "labels": labels})

        return result

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor,self.angle,self.imagu,self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle,self.imagu,self.advanced)

        return img, featureMap

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels


def main(training=True, visual=True):
    np.random.seed(100)
    FILE_ROOT = r"D:/data/fruitsNuts/"

    classes = ['date', 'fig', 'hazelnut']

    anchors = [[0.1, 0.1], [0.125, 0.125], [0.3, 0.3], [0.5, 0.5]]  # 缩放到0~1 [w,h]

    angle = None
    imagu = True
    advanced = False
    muilscale = False  # True
    mode = 'fcos'
    # mode = 'fcosv2'
    use_mosaic = True
    fix_resize = False
    resize = 512  # 416
    strides = 16  # 32
    num_anchors = len(anchors)
    num_classes = len(classes)
    batch_size = 16  # 32
    epochs = 100

    transform = T.Compose(
        [
            # T.Resize((resize,resize)),
            # T.RandomApply([T.GaussianBlur(5)]),
            # T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)]),
            # T.RandomApply([T.ColorJitter(0.125, 0.5, 0.5, 0.05)]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # T.RandomErasing()
        ])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Yolov1simple('resnet34', True, num_anchors, num_classes, strides).to(device)

    # if not training:
    if os.path.exists('weight.pth'):
        model.load_state_dict(torch.load('weight.pth', map_location=device))
        print("load weight successful!!!")

    if training:
        # dataset = FruitsNutsDataset(FILE_ROOT, classes, resize, strides, muilscale, batch_size,
        #                             num_anchors, transform, mode, use_mosaic, fix_resize, anchors,
        #                             angle,imagu,advanced)
        dataset = FruitsNutsDatasetYOLOV2(FILE_ROOT, classes, resize, strides, muilscale, batch_size,
                                    num_anchors, transform, mode, use_mosaic, fix_resize, anchors,
                                    angle, imagu, advanced)

        num_datas = len(dataset)
        # num_train = int(0.9 * num_datas)
        # indices = torch.randperm(num_datas).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:num_train])
        train_dataset = torch.utils.data.Subset(dataset, torch.arange(0, num_datas)[2:])

        dataLoader = DataLoader(train_dataset, 1 if muilscale else batch_size, True, collate_fn=dataset.collate_fn)

        # optim = torch.optim.RMSprop([parm for parm in model.parameters() if parm.requires_grad],
        #                             lr=5e-4, weight_decay=5e-5, momentum=0.95)
        optim = torch.optim.AdamW([parm for parm in model.parameters() if parm.requires_grad],
                                  lr=5e-4, weight_decay=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.8)
        lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lf)

        for epoch in range(epochs):
            train(model, optim, dataLoader, device, epoch, muilscale, mode)
            # predict(model, img_path, transform, resize, device)
            scheduler.step()
            torch.save(model.state_dict(), 'weight.pth')
    else:
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        fh = fw = resize // strides
        img_paths = glob((FILE_ROOT + "/images/*.jpg").replace("//", '/'))[:2]
        for img_path in img_paths:
            info = predict(anchors, model, img_path, test_transform, resize, device, visual, fh, fw, fix_resize, mode,
                           conf_threshold=0.3)
            print(info)


if __name__ == "__main__":
    training = True
    # training = False
    main(training=training, visual=False)
