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

# from toolcv.data.transform import getitem_fcosMS, getitem_mutilscale_fcosMS
from toolcv.network.net import Yolov3simple
# from toolcv.tools.utils import train_fcosMS as train, predict_fcosMS as predict
from toolcv.data.dataset import FruitsNutsDatasetFCOSMS

from toolcv.tools.utils import fit, evalute


def main(training=True, visual=True):
    FILE_ROOT = r"D:/data/fruitsNuts/"

    classes = ['date', 'fig', 'hazelnut']

    angle = None
    imagu = True
    advanced = False
    muilscale = False  # True
    mode = 'exp-v1'  # 'sigmoid'
    use_mosaic = True
    fix_resize = False
    resize = 416  # 416
    strides = [8,16,32]  # 32
    num_anchors = 1
    num_classes = len(classes)
    batch_size = 16  # 32
    epochs = 50  # 100
    lr = 5e-4
    weight_decay = 5e-5
    print_step = 20
    reduction = 'mean'
    boxes_weight = 1.0
    max_norm = 0.1
    method = 'fcosms'
    save_path = './output'
    iou_threshold = 0.5
    conf_threshold = 0.2
    anchors = []
    focal_loss = False
    seed = 100

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    model = Yolov3simple('resnet18', True, num_anchors, num_classes, strides,0.1,5,5).to(device)

    if training:
        dataset = FruitsNutsDatasetFCOSMS(FILE_ROOT, classes, resize, strides, muilscale, batch_size,
                                        num_anchors, transform, mode, use_mosaic, fix_resize, angle, imagu, advanced)

        fit(model, dataset, None, None, None, epochs, batch_size, muilscale, lr, weight_decay, seed, device, save_path,
            mode, print_step, reduction, boxes_weight, max_norm, method, focal_loss)
    else:
        img_paths = glob((FILE_ROOT + "/images/*.jpg").replace("//", '/'))  # [:2]
        evalute(model, img_paths, None, resize, device, visual, strides, fix_resize, mode, save_path,
                iou_threshold, conf_threshold, method, anchors, focal_loss)


if __name__ == "__main__":
    training = True
    # main(training=training, visual=False)
    main(training=False, visual=False)
