"""
# (每个正样本 可能对应 n个anchor n>=1)
1、每个anchor 对应于最大的gt 如果 iou>0.5为正，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# 参考yolo 改进为 (每个正样本 只对应 1个anchor)  本程序采用这种方式实现
1、每个anchor 对应于最大的gt 如果 iou>0.5忽略，否则为负
2、每个gt 对应的最大的先验anchor 为正样本
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
import torch.nn.functional as F

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Vgg16Backbone, SSDHead300, SSDHead512, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxes_ssd, grid_torch, get_bboxes_ssdv2
from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh
from toolcv.api.define.utils.tools.anchor import ssd300_anchors, ssd512_anchors

ignore = True

seed = 100
torch.manual_seed(seed)

featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
num_anchors = [4, 6, 6, 6, 4, 4]
anchors = ssd300_anchors(featmap_sizes=featmap_sizes, clip=True, normalization=True)
anchors = torch.cat(anchors, 0)

strides = 4
use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
h, w = 300, 300
resize = (h, w)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']  # 使用centernet方式可以不包括背景
num_classes = len(classes)
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
                                  transforms_mosaic=transforms_mosaic, h=h, w=w, mosaic_mode=0)
val_dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, use_mosaic=False,
                                transforms_mosaic=None, h=h, w=w)
dataloader = LoadDataloader(train_dataset, val_dataset, 0.0, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
backbone = Vgg16Backbone('vgg16', True, 30)
head = SSDHead300(num_classes, num_anchors)
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, head).to(device)


class SSD300(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def get_targets(self, imgs, targets, cls_shapes, reg_shapes):
        anchors_xywh = x1y1x2y22xywh(self.anchors)
        bs, _, img_h, img_w = imgs.shape
        cls_targets = imgs[0].new_zeros(cls_shapes)
        reg_targets = imgs[0].new_zeros(reg_shapes)
        cls_reg_weight = imgs[0].new_zeros(cls_targets.shape[:-1])[..., None]
        cls_reg_ious = torch.zeros_like(cls_reg_weight)

        for batch_id in range(bs):
            # 缩放到 0~1
            gt_bbox = targets[batch_id]['boxes'] / torch.tensor([[img_w, img_h, img_w, img_h]], dtype=torch.float32)
            gt_label = targets[batch_id]['labels']

            gt_xywh = x1y1x2y22xywh(gt_bbox)

            ious = box_iou(self.anchors, gt_bbox)

            # 每个anchor 对应于最大的gt
            scores, indexs = ious.max(1)
            if ignore:
                # iou > 0.5 的 忽略
                keep = scores > 0.5
                cls_reg_weight[batch_id][keep] = -1  # 忽略
            else:
                # 类似centernet的标记方式 (类似于heatmap)
                for j, index in enumerate(indexs):
                    label = gt_label[index]
                    # cls_targets[batch_id, j, label] = scores[j]
                    v = scores[j] * 0.8
                    cls_targets[batch_id, j, label] = v if v > 0.3 else 0

            # 每个gt 对应的最大的先验anchor 为正样本
            scores, indexs = ious.max(0)
            for j, index in enumerate(indexs):
                gx, gy, gw, gh = gt_xywh[j]
                label = gt_label[j]
                ax, ay, ah, aw = anchors_xywh[index]

                cls_reg_weight[batch_id, index] = 1
                cls_reg_ious[batch_id, index] = scores[j]

                cls_targets[batch_id, index, label] = 1

                dx = (gx - ax) / aw
                dy = (gy - ay) / ah
                dw = (gw / aw).log()
                dh = (gh / ah).log()
                reg_targets[batch_id, index, 0] = dx
                reg_targets[batch_id, index, 1] = dy
                reg_targets[batch_id, index, 2] = dw
                reg_targets[batch_id, index, 3] = dh

        avg_factor = max(1, cls_reg_weight.eq(1).sum())  # 正样本总个数

        target_result = dict(cls_targets=cls_targets,
                             reg_targets=reg_targets,
                             cls_reg_weight=cls_reg_weight,
                             cls_reg_ious=cls_reg_ious
                             )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_preds, reg_preds = self.model(imgs)

        cls_shape = cls_preds.shape
        reg_shape = reg_preds.shape
        # get_target
        # target
        target_result, avg_factor = self.get_targets(imgs, targets, cls_shape, reg_shape)
        cls_targets = target_result['cls_targets']
        reg_targets = target_result['reg_targets']
        cls_reg_weight = target_result['cls_reg_weight']
        cls_reg_ious = target_result['cls_reg_ious']

        # loss
        keep = (cls_reg_weight != -1).squeeze(-1)
        loss_cls = GaussianFocalLoss()(cls_preds.sigmoid()[keep], cls_targets[keep], avg_factor=avg_factor)
        # loss_cls = FocalLoss()(cls_preds[keep], cls_targets[keep].argmax(-1), avg_factor=avg_factor) # 会自动加上 sigmoid

        # """
        _cls_reg_weight = (cls_reg_weight == 1).float().expand_as(reg_targets)
        loss_reg = L1Loss()(reg_preds, reg_targets, _cls_reg_weight, avg_factor=avg_factor)
        """
        keep = (cls_reg_weight == 1).squeeze(-1)
        loss_reg = ((2 - cls_reg_ious[keep]) *  # iou越小权重越大
                    F.smooth_l1_loss(reg_preds[keep],
                                     reg_targets[keep],
                                     reduction='none')).sum() / (avg_factor)
        # """
        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg)

    @torch.no_grad()
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        cls_preds, reg_preds = self.model(img)
        cls_preds = cls_preds.sigmoid()
        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(reg_preds)
        # decode
        reg_preds[..., :2] *= anchors_xywh[..., 2:]
        reg_preds[..., :2] += anchors_xywh[..., :2]
        reg_preds[..., 2:] = reg_preds[..., 2:].exp()
        reg_preds[..., 2:] *= anchors_xywh[..., 2:]
        """
        boxes, scores, labels = get_bboxes_ssd(None,cls_preds, reg_preds, with_nms, iou_threshold, conf_threshold,
                                               scale_factors, padding, in_shape, True, 200)
        """
        boxes, scores, labels = get_bboxes_ssdv2(featmap_sizes, num_anchors, None, cls_preds, reg_preds, with_nms,
                                                 iou_threshold, conf_threshold,
                                                 scale_factors, padding, in_shape, True, 100)
        # """
        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        cls_preds, reg_preds = self.model(imgs)
        cls_preds = cls_preds.sigmoid()
        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(reg_preds)
        # decode
        reg_preds[..., :2] *= anchors_xywh[..., 2:]
        reg_preds[..., :2] += anchors_xywh[..., :2]
        reg_preds[..., 2:] = reg_preds[..., 2:].exp()
        reg_preds[..., 2:] *= anchors_xywh[..., 2:]

        """
        boxes, scores, labels = get_bboxes_ssd(None,cls_preds, reg_preds, with_nms, iou_threshold, conf_threshold,
                                               scale_factors, padding, in_shape, False, 200)
        """
        boxes, scores, labels = get_bboxes_ssdv2(featmap_sizes, num_anchors, None, cls_preds, reg_preds, with_nms,
                                                 iou_threshold, conf_threshold,
                                                 scale_factors, padding, in_shape, False, 100)
        # """
        return boxes, scores, labels


network = SSD300(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                 gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                 gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                 train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(mode='mmdet', conf_threshold=0.2)
# -----------------------predict ---------------------------
# network.predict(glob_format(dir_data), test_transforms, device, visual=False,
#                 conf_threshold=0.2, with_nms=True, method=method)
