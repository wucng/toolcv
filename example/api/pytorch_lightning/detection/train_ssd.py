import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.models.resnet import resnet18
import os
import numpy as np
import random
import pytorch_lightning as pl

from toolcv.network.net import SSDsimple, SSDsimpleV2
from toolcv.tools.anchor import getAnchorsV2_s

from toolcv.api.pytorch_lightning.data import FruitsNutsDataset, glob_format
from toolcv.api.pytorch_lightning.net import DetecteModel
from toolcv.api.pytorch_lightning.data import BaseDataModule
from toolcv.api.pytorch_lightning.utils import load_callbacks, load_logger, set_seed, predict_ssd
from toolcv.api.pytorch_lightning.train import fit, test
from toolcv.api.pytorch_lightning.transform import Transforms
from toolcv.data.augment import bboxAug, bboxAugv2


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
    filter_size = 5  # 15l
    strides = 16
    num_classes = len(classes)
    # scales = [128, 256, 512]
    # ratios = [0.5, 1, 2]
    scales = [32, 64, 128, 256]
    ratios = [1]

    anchors = getAnchorsV2_s((resize, resize), strides, scales, ratios)  # 缩放到0~1
    num_anchors = len(scales) * len(ratios)

    focal_loss = True
    box_norm = 'log'
    mode = 'centernet'
    method = 'ssd'
    multiscale = False
    batch_size = 32
    scales = []

    dropout = 0.0
    freeze_at = 5
    epochs = 50
    lr = 5e-4
    weight_decay = 5e-5 * (resize / 224)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    symbol = "\\"  # '/'
    checkpoit_paths = sorted(glob_format('./logs', (".ckpt",)),
                             key=lambda x: int(x.split('version_')[1].split(symbol)[0]))
    checkpoint_path = "" if len(checkpoit_paths) == 0 else checkpoit_paths[-1]
    gamma = 0.9
    boxes_weight = 1.0
    thres = 0.3
    reduction = 'mean'

    _model = SSDsimple('dla34', True, num_anchors, num_classes, strides, dropout, freeze_at)
    model = DetecteModel(_model, None, epochs=epochs, warpstep=1000, lr=lr, lrf=0.1, weight_decay=weight_decay,
                         gamma=gamma, mode=mode, method=method, boxes_weight=boxes_weight, thres=thres,
                         box_norm=box_norm, multiscale=multiscale, reduction=reduction, focal_loss=focal_loss)

    if os.path.exists(checkpoint_path):
        # model.load_from_checkpoint(checkpoint_path)
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['state_dict'])
        print("------load weight successful!!---------")

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

    if training:
        config = {"callbacks": load_callbacks(False, 'val_loss'), "logger": load_logger()}
        fit(model, epochs, config, train_dataloader, val_dataloader)
    else:
        test_transform = bboxAug.Compose([
            # bboxAug.Resize((resize, resize)),
            # bboxAugv2.ResizeAndAlign((resize,resize)),
            bboxAugv2.ResizeV2((resize, resize)),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])

        _model = model.model.to(device)
        img_paths = glob_format(dir_data)
        predict_ssd(anchors, _model, img_paths, test_transform, resize, device, visual, strides, fix_resize=True,
                    iou_threshold=0.3, conf_threshold=0.1, focal_loss=focal_loss)


if __name__ == "__main__":
    main()
    main(False, visual=False)
