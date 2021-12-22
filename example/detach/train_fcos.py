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

from toolcv.data.transform import getitem_fcos, getitem_mutilscale
from toolcv.network.net import Yolov1simple, Yolov1simpleV2
from toolcv.tools.utils import train_fcos as train, predict_fcos as predict
from toolcv.data.dataset import FruitsNutsDatasetFCOS

from toolcv.tools.utils import fit, evalute


def main(training=True, visual=True):
    np.random.seed(100)
    FILE_ROOT = r"D:/data/fruitsNuts/"

    classes = ['date', 'fig', 'hazelnut']

    angle = None
    imagu = True
    advanced = False
    muilscale = False  # True
    mode = 'exp'  # 'sigmoid'
    use_mosaic = True
    fix_resize = False
    resize = 416  # 416
    strides = 16  # 32
    num_anchors = 1
    num_classes = len(classes)
    batch_size = 16  # 32
    epochs = 50  # 100

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
    model = Yolov1simple('dla34', True, num_anchors, num_classes, strides).to(device)
    # model = Yolov1simpleV2('dla34', True, num_anchors, num_classes, strides).to(device)

    # if not training:
    if os.path.exists('weight.pth'):
        model.load_state_dict(torch.load('weight.pth', map_location=device))
        print("load weight successful!!!")

    if training:
        dataset = FruitsNutsDatasetFCOS(FILE_ROOT, classes, resize, strides, muilscale, batch_size,
                                        num_anchors, transform, mode, use_mosaic, fix_resize, angle, imagu, advanced)

        # num_datas = len(dataset)
        # # num_train = int(0.9 * num_datas)
        # # indices = torch.randperm(num_datas).tolist()
        # # dataset = torch.utils.data.Subset(dataset, indices[:num_train])
        # train_dataset = torch.utils.data.Subset(dataset, torch.arange(0, num_datas)[2:])

        dataLoader = DataLoader(dataset, 1 if muilscale else batch_size, True, collate_fn=dataset.collate_fn)

        # optim = torch.optim.RMSprop([parm for parm in model.parameters() if parm.requires_grad],
        #                             lr=5e-4, weight_decay=5e-5, momentum=0.95)
        optim = torch.optim.AdamW([parm for parm in model.parameters() if parm.requires_grad],
                                  lr=5e-4, weight_decay=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.8)
        lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lf)

        for epoch in range(epochs):
            train(model, optim, dataLoader, device, epoch, muilscale, mode, reduction='mean', boxes_weight=1.0)
            # predict(model, img_path, transform, resize, device)
            # scheduler.step()
            torch.save(model.state_dict(), 'weight.pth')
    else:
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        fh = fw = resize // strides
        img_paths = glob((FILE_ROOT + "/images/*.jpg").replace("//", '/'))  # [:2]
        for img_path in img_paths:
            info = predict(model, img_path, test_transform, resize, device, visual, fh, fw, fix_resize, mode,
                           conf_threshold=0.2)
            print(info)


def main2(training=True, visual=True):
    FILE_ROOT = r"D:/data/fruitsNuts/"

    classes = ['date', 'fig', 'hazelnut']

    angle = None
    imagu = True
    advanced = False
    muilscale = False  # True
    mode = 'exp'  # 'sigmoid' (推荐 exp)
    use_mosaic = True
    fix_resize = False
    resize = 416  # 416
    strides = 16  # 32
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
    method = 'fcos'
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

    # model = Yolov1simple('resnet18', True, num_anchors, num_classes, strides,0.1,5,5).to(device)
    model = Yolov1simpleV2('resnet18', True, num_anchors, num_classes, strides,0.1,5,5).to(device)

    if training:
        dataset = FruitsNutsDatasetFCOS(FILE_ROOT, classes, resize, strides, muilscale, batch_size,
                                        num_anchors, transform, mode, use_mosaic, fix_resize, angle, imagu, advanced)

        fit(model, dataset, None, None, None, epochs, batch_size, muilscale, lr, weight_decay, seed, device, save_path,
            mode, print_step, reduction, boxes_weight, max_norm, method, focal_loss)
    else:
        img_paths = glob((FILE_ROOT + "/images/*.jpg").replace("//", '/'))  # [:2]
        evalute(model, img_paths, None, resize, device, visual, strides, fix_resize, mode, save_path,
                iou_threshold, conf_threshold, method, anchors, focal_loss)


if __name__ == "__main__":
    training = True
    # training = False
    # main(training=training, visual=False)
    main2(training=training, visual=False)
    main2(training=False, visual=False)
