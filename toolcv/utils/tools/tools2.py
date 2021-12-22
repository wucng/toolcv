import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader
import numpy as np

from torch.hub import load_state_dict_from_url

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms as T
from prefetch_generator import BackgroundGenerator
import time
import os
import csv
from tqdm import tqdm
import math
import csv
import random

from timm.optim import RAdam, RMSpropTF

from toolcv.utils.tools.lr_scheduler import SineAnnealingLROnecev2


class DataLoaderX(DataLoader):
    """然后用DataLoaderX替换原本的DataLoader
    prefetch_generator 加速 Pytorch 数据读取
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_train_val_dataset(train_dataset, val_dataset, train_ratio=0.8):
    """
    nums = len(train_dataset)
    nums_train = int(nums * train_ratio)
    nums_val = nums - nums_train
    train_dataset, val_dataset = random_split(train_dataset, [nums_train, nums_val])
    """

    # train_ratio = 1 - 0.1
    num_datas = len(train_dataset)
    num_train = int(train_ratio * num_datas)
    indices = torch.randperm(num_datas).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[num_train:])

    return train_dataset, val_dataset


def get_optim_scheduler(model, T_max, lr=2e-3, weight_decay=1e-4, mode="radam",
                        scheduler_mode="CosineAnnealingLR".lower(), lrf=0.01, gamma=0.6):
    if gamma <= 0:
        params = model.parameters()
        params = [param for param in params if param.requires_grad]
    else:
        params = get_params(model.modules(), lr, weight_decay, gamma)

    if mode == "radam":
        optim = RAdam(params, lr, weight_decay=weight_decay)
    elif mode == "adam":
        optim = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif mode == "adamw":
        optim = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    elif mode == "RMSprop".lower():
        optim = torch.optim.RMSprop(params, lr, weight_decay=weight_decay, momentum=0.9)
    elif mode == "RMSpropTF".lower():
        optim = RMSpropTF(params, lr, weight_decay=weight_decay, momentum=0.9)
    elif mode == "SGD".lower():
        optim = torch.optim.SGD(params, lr, weight_decay=weight_decay, momentum=0.9)

    if scheduler_mode == "CosineAnnealingLR".lower():
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max, lr * lrf)
    elif scheduler_mode == "OneCycleLR".lower():
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, lr, T_max)
    else:
        scheduler = SineAnnealingLROnecev2(optim, T_max, lrf)  # len(train_dataloader) * 4

    return optim, scheduler


def set_seed(seed, only_torch=True):
    # set the seed
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
    if not only_torch:
        random.seed(seed)
        np.random.seed(seed)


def get_device(device='cuda:0'):
    if torch.cuda.is_available():
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_params(modules, lr=5e-4, weight_decay=5e-5, gamma=0.8):
    # params = [param for param in self.parameters() if param.requires_grad]
    params = []
    memo = set()
    for module in modules:  # self.modules()
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            _lr = lr
            _weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                _weight_decay = 0.0
            elif "bias" in key:
                _lr = lr
                _weight_decay = weight_decay

            elif 'backbone' in key:
                _lr = lr * gamma
                _weight_decay = weight_decay * gamma

            params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]

    return params


def load_weight(model, weight_path='weight.pth', url="", device="cpu"):
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, device)
        else:
            state_dict = load_state_dict_from_url(url, map_location=device)
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].numel() == v.numel():
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = v
                print("%s not load weight" % k)
        model.load_state_dict(new_state_dict)
        print("---------load weight successful-----------")
        del state_dict
        del new_state_dict
    except:
        print("---------load weight fail-----------")


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(1.0, x / warmup_iters))
    # """
