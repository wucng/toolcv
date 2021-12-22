import torch
from torch import nn
from torch.nn import functional as Func
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import math

from toolcv.api.define.utils.model.mmdet import _BaseNet, _initParmas, load_model, batched_nms
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.data.dataset import glob_format
from toolcv.api.define.utils.model.net import CBA,Backbone


class Fcos(_BaseNet):
    def __init__(self, model, config, checkpoint, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters, gamma,
                         optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def train_step(self, model, batch, step):
        pass

    @torch.no_grad()
    def pred_step(self, model, img, iou_threshold, conf_threshold, scale_factors, in_shape, with_nms):
        pass
