import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pytorch_lightning.callbacks as plc
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import time
import math

from toolcv.api.pytorch_lightning.utils import test as _test


def fit(model, epochs, config=None, train_dataloader=None, val_dataloaders=None, datamodule=None):
    default_config = {
        "max_epochs": epochs,
        "gpus": [0],
        "log_every_n_steps": 50,
        "gradient_clip_val": 0.1,  # 梯度裁剪
        "precision": 16,  # 半精度 16,32,64 （起到加速） 默认是32 不能与 "amp_backend":'apex' 同时使用
        # "amp_backend":'apex', # using NVIDIA Apex （起到加速） 安装参考：https://github.com/NVIDIA/apex#linux
        "accumulate_grad_batches": 4,  # 每4个batch作一次梯度更新（起到加速），原来是每个batch都作梯度更新
        "stochastic_weight_avg": True,
        # "auto_scale_batch_size":'binsearch', # 根据内存选择合适的batch_size (# run batch size scaling, result overrides hparams.batch_size)
        # "auto_lr_find":True # 自动寻找合适的初始化学习率
    }

    if config is not None:
        default_config.update(config)

    config = default_config

    trainer = pl.Trainer(**config)

    if datamodule is not None:
        datamodule.setup('fit')

    if ("auto_scale_batch_size" in config and config["auto_scale_batch_size"] != False) or \
            ("auto_lr_find" in config and config["auto_lr_find"] != False):
        trainer.tune(model, train_dataloader, val_dataloaders, datamodule) # {"batch_size": 1}, {"lr": 0}
        # 仅仅用于辅助 fit
    else:
        trainer.fit(model, train_dataloader, val_dataloaders, datamodule)


def test(model, checkpoint_path=None, test_dataloader=None, datamodule=None, device='cpu', mode=0):
    if datamodule is not None:
        datamodule.setup('test')

    if mode == 0:
        if checkpoint_path is not None:
            # 恢复模型
            # model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
            ckpt = torch.load(checkpoint_path)
            model.load_state_dict(ckpt['state_dict'])
        # 定义trainer并测试
        trainer = pl.Trainer(gpus=1, precision=16, limit_test_batches=0.1)
        trainer.test(model, test_dataloader, datamodule=datamodule)
    else:
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            model.load_state_dict(ckpt['state_dict'])
        model.to(device)
        mean_acc = _test(model, test_dataloader if test_dataloader is not None else datamodule.test_dataloader(),
                         device)
