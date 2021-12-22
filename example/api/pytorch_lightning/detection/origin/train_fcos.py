import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.models.resnet import resnet18
import os
import numpy as np
import random
import pytorch_lightning as pl
import math

from toolcv.network.net import Yolov1simple, Yolov1simpleV2

from toolcv.api.pytorch_lightning.data import FruitsNutsDataset, glob_format
from toolcv.api.pytorch_lightning.net import DetecteModel, get_params
from toolcv.api.pytorch_lightning.data import BaseDataModule
from toolcv.api.pytorch_lightning.utils import load_callbacks, load_logger, set_seed, predict_fcos
from toolcv.api.pytorch_lightning.train import fit, test
from toolcv.api.pytorch_lightning.transform import Transforms
from toolcv.data.augment import bboxAug, bboxAugv2
from toolcv.tools.utils import train_fcos as train


def trans(advanced=False, base_size=448, target_size=416):
    return bboxAug.Compose([
        bboxAug.RandomHorizontalFlip(),
        # bboxAug.Resize((target_size, target_size)),
        # bboxAugv2.ResizeFixMinAndRandomCrop(base_size, (target_size, target_size)),
        bboxAugv2.ResizeV2((target_size, target_size)),
        # bboxAugv2.RandomRotate(),
        # bboxAugv2.RandomAffine(),
        # bboxAugv2.RandomDropPixelV2(),
        # bboxAugv2.RandomMosaic(),
        bboxAug.ToTensor(),  # PIL --> tensor
        bboxAug.Normalize()  # tensor --> tensor
    ])


def main(training=True, visual=False):
    set_seed(100)

    dir_data = r"D:\data\fruitsNuts"
    classes = ['date', 'fig', 'hazelnut']
    useMosaic = False
    mosaic_method = 0
    resize = 224
    transforms = Transforms(trans, False, resize)
    filter_size = 5  # 15
    strides = 16
    num_classes = len(classes)
    num_anchors = 1
    anchors = []
    box_norm = 'log'
    mode = 'exp'
    method = 'fcos'
    multiscale = False
    batch_size = 32
    scales = []

    dropout = 0.0
    freeze_at = 5
    epochs = 50
    lr = 5e-4
    weight_decay = 5e-5 * (resize / 224)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gamma = 0.6
    boxes_weight = 5.0
    thres = 0.2
    reduction = 'mean'

    model = Yolov1simple('dla34', True, num_anchors, num_classes, strides, dropout, 5, freeze_at).to(device)
    # model = Yolov1simpleV2('dla34', True, num_anchors, num_classes, strides,dropout, 5, freeze_at).to(device)


    if os.path.exists('weight.pth'):
        model.load_state_dict(torch.load('weight.pth', map_location=device))
        print("load weight successful!!!")

    if training:
        # method = "none"
        dataset = FruitsNutsDataset(dir_data, classes, transforms, useMosaic, mosaic_method, filter_size,
                                    resize, strides, anchors, num_anchors, box_norm, mode, method,
                                    multiscale, batch_size, scales)
        # dataset.show(15,'plt')
        nums = len(dataset)
        nums_train = int(0.8 * nums)
        nums_val = nums - nums_train
        train_dataset, val_dataset = random_split(dataset, [nums_train, nums_val])

        train_dataloader = DataLoader(train_dataset, 1 if multiscale else batch_size, True, num_workers=0,
                                      collate_fn=dataset.collate_fnV2)
        val_dataloader = DataLoader(val_dataset, 1 if multiscale else batch_size, False, num_workers=0,
                                    collate_fn=dataset.collate_fnV2)

        """

        num_datas = len(dataset)
        # num_train = int(0.9 * num_datas)
        # indices = torch.randperm(num_datas).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:num_train])
        train_dataset = torch.utils.data.Subset(dataset, torch.arange(0, num_datas)[2:])
        dataLoader = DataLoader(train_dataset, 1 if muilscale else batch_size, True, collate_fn=dataset.collate_fn)
        # """
        # params = [parm for parm in model.parameters() if parm.requires_grad]
        params = get_params(model.modules(), lr=lr, weight_decay=weight_decay, gamma=gamma)
        # optim = torch.optim.RMSprop(params,lr=lr, weight_decay=weight_decay, momentum=0.95)
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.8)
        lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lf)

        for epoch in range(epochs):
            train(model, optim, train_dataloader, device, epoch, multiscale, mode, reduction='mean')
            # predict(model, img_path, transform, resize, device)
            scheduler.step()
            torch.save(model.state_dict(), 'weight.pth')
    else:
        test_transform = bboxAug.Compose([
            # bboxAug.Resize((resize, resize)),
            # bboxAugv2.ResizeAndAlign((resize,resize)),
            bboxAugv2.ResizeV2((resize, resize)),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])


        img_paths = glob_format(dir_data)
        predict_fcos(model, img_paths, test_transform, resize, device, visual, strides, fix_resize=True, mode=mode,
                     iou_threshold=0.3, conf_threshold=0.1)


if __name__ == "__main__":
    main()
    main(False, visual=False)
