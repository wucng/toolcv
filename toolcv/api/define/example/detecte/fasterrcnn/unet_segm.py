"""
使用Unet搭建 segment（语义分割模型）

# 0 对应背景
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms, RoIAlign
import numpy as np
from tqdm import tqdm

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, FPNv2, UnetHead, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh
from toolcv.api.define.utils.tools.anchor import rcnn_anchors, get_anchor_cfg
from toolcv.tools.anchor import getAnchorsV2_s

from toolcv.api.pytorch_lightning.net import get_params
from toolcv.api.mobileNeXt.codebase.optim.radam import RAdam, PlainRAdam
from toolcv.api.mobileNeXt.codebase.scheduler.plateau_lr import PlateauLRScheduler

ignore = True

seed = 100
torch.manual_seed(seed)
strides = 16

# h, w = 800, 1330  # 最大边1333；最小边800
# h, w = 512, 512
h, w = 256, 256
resize = (h, w)

anchors = None

use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 30
batch_size = 4

dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
num_classes = len(classes) + 1  # +1 表示 加上背景
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.backends.cudnn.benchmark = True
# else:
#     device = torch.device('cpu')

# -------------------data-------------------
use_mosaic = False
method = 'both_sides'
if use_mosaic:
    transforms = aug.Compose([
        # aug.RandomHorizontalFlip(),
        # aug.Resize(*resize),
        # aug.ResizeMax(*resize), aug.Padding(),
        aug.ToTensor(), aug.Normalize()])

    transforms_mosaic = aug.Compose([
        aug.RandomBlur(),
        aug.RandomHorizontalFlip(),
        aug.ResizeMax(*resize), aug.Padding(),
        aug.RandomColorJitter([0.3, 0.6], [0.3, 0.6], [0.3, 0.6])
    ])
else:
    transforms = aug.Compose([
        aug.RandomHorizontalFlip(),
        aug.ResizeMax(*resize), aug.Padding(pad_mode=method),
        aug.ToTensor(), aug.Normalize()
    ])
    # transforms = aug.Compose([
    #     aug.RandomHorizontalFlip(),
    #     # aug.Crop(),aug.WarpAffine(0),
    #     # aug.Resize(*resize),
    #     # aug.ResizeMax(*resize), aug.Padding(),
    #     # aug.RandomChoice([baug.RandomBlur(), baug.RandomNoise(), aug.RandomColorJitter()]),
    #     # aug.RandomChoice([baug.RandomDropPixelV2(0), baug.RandomRotate(), baug.RandomMosaicV2()]),
    #     # aug.RandomChoice([[aug.ResizeMax(*resize), aug.Padding()],
    #     #                   [baug.ResizeFixMinAndRandomCropV2(ratio_range=[0.5, 0.6, 0.7, 0.8, 0.9]),
    #     #                    aug.Resize(*resize)]]),
    #     baug.ResizeFixMinAndRandomCropV2AndPatch([100, 200, 300, 400, 500, 600], [0.5, 0.6, 0.7, 0.8, 0.9]),
    #     aug.Resize(*resize),
    #     aug.ToTensor(), aug.Normalize()])
    transforms_mosaic = None

test_transforms = aug.Compose([
    # aug.Resize(*resize),
    aug.ResizeMax(*resize), aug.Padding(pad_mode=method),
    aug.ToTensor(), aug.Normalize()])
"""
dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, use_mosaic=use_mosaic,
                            transforms_mosaic=transforms_mosaic, h=h, w=w)
dataset.show(20,mode='cv2')
exit(0)
dataloader = LoadDataloader(dataset, None, 0.2, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()
"""
train_dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, use_mosaic=use_mosaic,
                                  transforms_mosaic=transforms_mosaic, h=h, w=w, mosaic_mode=0, masks=True)
val_dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, use_mosaic=False,
                                transforms_mosaic=None, h=h, w=w, masks=True)
dataloader = LoadDataloader(train_dataset, val_dataset, 0.0, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
"""
backbone = Backbone('resnet18', True, num_out=4)
fpn = FPNv2(backbone.out_channels, 256, True)
head = UnetHead(256, num_classes)

_initParmas(fpn.modules(), mode='normal')
_initParmas(head.modules(), mode='normal')

model = nn.Sequential(backbone, fpn, head).to(device)

"""
backbone = Backbone('resnet18', True, num_out=4)
fpn = FPNv2(backbone.out_channels, 256, True)
# head = UnetHead(256, num_classes)
out_channles = 256
head = nn.Sequential(
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(out_channles, out_channles, 3, 2, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(out_channles, out_channles, 3, 2, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, num_classes, 3, 1, 1)
)

_initParmas(fpn.modules(), mode='normal')
_initParmas(head.modules(), mode='normal')

model = nn.Sequential(backbone, fpn, head).to(device)

model.forward = lambda x: model[2](model[1](model[0](x))[0])
# """


"""
optimizer = None
scheduler=None
"""
optimizer = PlainRAdam(get_params(model.modules(), lr, weight_decay, gamma=0.5), lr, weight_decay=weight_decay)
# scheduler = PlateauLRScheduler(optimizer)
scheduler = None


# """

class FasterRcnn(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        losses = {}
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            output = self.model(imgs)

        bs = output.size(0)
        target_segm = torch.stack([
            (targets[i]['masks'] * (1 + targets[i]['labels'])[..., None, None]).sum(0).clamp(0, self.num_classes - 1)
            for i in range(bs)], 0).to(self.device)  # 0对应背景

        # loss_segm = F.cross_entropy(output, target_segm, reduction='sum') / (target_segm > 0).sum()
        loss_segm = F.cross_entropy(output, target_segm, reduction='mean')

        losses.update(dict(loss_segm=loss_segm))

        return losses

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        in_shape = imgs.shape[-2:]
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        scale_factors = kwargs['scale_factors']
        padding = kwargs['padding']
        with_nms = kwargs['with_nms']

        output = self.model(imgs).softmax(1)

        # 缩放到原始图像上
        h, w = in_shape
        px1, py1, px2, py2 = list(map(int, padding))
        scale_x, scale_y = scale_factors[:2]
        output = output[:, :, py1:h - py2, px1:w - px2]
        output = F.interpolate(output, scale_factor=(1 / scale_y, 1 / scale_x), recompute_scale_factor=True)

        values, indices = output.max(1)
        pred_segm = indices.float()[0]
        # pred_segm *= (values > 0.5).float()

        return pred_segm

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        # iou_threshold = kwargs['iou_threshold']
        # conf_threshold = kwargs['conf_threshold']
        # with_nms = kwargs['with_nms']

        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        output = self.model(imgs).softmax(1)
        values, indices = output.max(1)
        pred_segm = indices.float().cpu()[0]
        # pred_segm *= (values > 0.5).float()

        if 'segm' in targets[0]:
            target_segm = targets[0]['segm']
        elif 'masks' in targets[0]:
            # 0 对应 背景
            target_segm = (targets[0]['masks'] *
                           (1 + targets[0]['labels'])[..., None, None]).sum(0).clamp(0, self.num_classes - 1)

        # return pred_segm, target_segm

        # 统计每个类别的像素精度
        pixel_accuracy_each = []
        for i in range(1, self.num_classes):  # 0 对应 背景
            pixel_accuracy_each.append((((pred_segm == i).float() == (
                    target_segm == i)).float().sum() / pred_segm.numel()).item())

        return pixel_accuracy_each

    @torch.no_grad()
    def evalute(self, **kwargs):
        pixel_accuracy = {}

        pbar = tqdm(enumerate(self.val_dataloader))
        for step, (imgs, targets) in pbar:
            pixel_accuracy_each = self.evalute_step((imgs, targets), step, **kwargs)
            for i in range(self.num_classes - 1):
                if i not in pixel_accuracy: pixel_accuracy[i] = []
                pixel_accuracy[i].append(pixel_accuracy_each[i])

            pbar.set_description(str(pixel_accuracy_each))

        desc = ""
        for i in range(self.num_classes - 1):
            pixel_accuracy[i] = np.mean(pixel_accuracy[i])
            desc += "label:%d pixel accuracy:%.3f\n" % (i, pixel_accuracy[i])
        print(desc)

        mpa = np.mean(list(pixel_accuracy.values()))
        print('mean pixel accuracy:%.3f' % mpa)


network = FasterRcnn(**dict(model=model, num_classes=num_classes,
                            img_shape=resize, anchors=anchors,
                            strides=strides, epochs=epochs,
                            lr=lr, weight_decay=weight_decay, lrf=lrf,
                            warmup_iters=1000, gamma=0.5, optimizer=optimizer,
                            scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                            gradient_clip_val=gradient_clip_val, device=device,
                            criterion=None, train_dataloader=train_dataLoader,
                            val_dataloader=val_dataLoader))

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path='weight.pth', iou_threshold=0.3,
                       conf_threshold=0.1, with_nms=False, mode='coco', iou_types='masks'))
# # -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data),
                       transform=test_transforms, device=device,
                       weight_path='weight.pth',
                       save_path='output', visual=False,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.1, method=method, draw='draw_segms'))
