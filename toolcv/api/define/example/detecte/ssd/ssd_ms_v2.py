"""
使用多个分支的 ssd  multiple stage

# (每个正样本 可能对应 n个anchor n>=1)
1、每个anchor 对应于最大的gt 如果 iou>0.5为正，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# 参考yolo 改进为 (每个正样本 只对应 1个anchor) 本程序采用这种方式实现
1、每个anchor 对应于最大的gt 如果 iou>0.5忽略，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
import numpy as np

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, FPN, RetinaHead, RetinaHeadV2, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh
from toolcv.api.define.utils.tools.anchor import rcnn_anchors, get_anchor_cfg
from toolcv.tools.anchor import getAnchorsV2_s, getAnchors_FPNV2

ignore = True
top5 = False

topk = 200
seed = 100
torch.manual_seed(seed)
strides = [8, 16, 32]

# h, w = 800, 1330  # 最大边1333；最小边800
# h, w = 512, 512
h, w = 320, 320
resize = (h, w)

anchor_size = ((30,50,70), (80,120,160), (180,230,290))
aspect_ratios = (1.0,)
num_anchors = 3
# 最终的anchor size 就是 anchor_size
anchors = getAnchors_FPNV2(resize, strides, anchor_size, aspect_ratios, True, False)
anchors = torch.from_numpy(anchors).float()

use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
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
backbone = Backbone('resnet18', True, num_out=3)
neck = FPN(backbone.out_channels, 256, True)
head = RetinaHeadV2(256, [num_anchors] * len(strides), num_classes + 1, True)  # +1 加上背景
# head = RetinaHead(256, num_anchors, num_classes + 1, True)  # +1 加上背景
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)


class SSDMS(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def get_targets(self, imgs, targets, feat_shape):
        anchors_xywh = x1y1x2y22xywh(self.anchors)
        cls_target, bbox_target, object_target = imgs[0].new_zeros(feat_shape[0]), imgs[0].new_zeros(feat_shape[1]), \
                                                 imgs[0].new_zeros(feat_shape[2])
        bs, _, img_h, img_w = imgs.shape

        # feat_h, feat_w = feat_shape[-1]
        # width_ratio = float(feat_w / img_w)  # 1/stride
        # height_ratio = float(feat_h / img_h)  # 1/stride
        # ratio = torch.tensor([width_ratio, height_ratio], dtype=imgs.dtype)

        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes'] / torch.tensor([img_w, img_h, img_w, img_h],
                                                                dtype=torch.float32)  # 统一缩放到0~1
            gt_label = targets[batch_id]['labels']
            # 与先验anchor计算iou 分配正负样本(统一缩放到0~1)
            ious = box_iou(self.anchors, gt_bbox)

            gt_xywh = x1y1x2y22xywh(gt_bbox)

            # 每个anchor 对应的最大的gt
            scores, indexs = ious.max(1)
            # assert indexs.unique().size(0) == indexs.size(0)
            # for j, index in enumerate(indexs):
            #     if ignore:
            #         if scores[j] > 0.5: object_target[batch_id, j, 0] = -1
            #     else:
            #         object_target[batch_id, j, 0] = scores[j]

            if ignore:
                # iou > 0.5 的 忽略
                keep = scores > 0.5
                object_target[batch_id][keep] = -1  # 忽略
            else:
                # 类似centernet的标记方式 (类似于heatmap)
                for j, index in enumerate(indexs):
                    # label = gt_label[index]
                    # cls_targets[batch_id, j, label] = scores[j]
                    v = scores[j] * 0.8
                    # cls_targets[batch_id, j, label] = v if v > 0.3 else 0
                    object_target[batch_id, j, 0] = v if v > 0.1 else 0

            if top5:
                values, indices = torch.topk(ious, 5, 0)
                for i in range(indices.size(0)):
                    for j in range(indices.size(1)):
                        index = indices[i, j]
                        label = gt_label[j]
                        if object_target[batch_id, index, 0] == 1:  # 重复了
                            object_target[batch_id, index, 0] = -1
                        else:
                            object_target[batch_id, index, 0] = 1  # scores[j]
                            gx, gy, gw, gh = gt_xywh[j]
                            ax, ay, aw, ah = anchors_xywh[index]
                            cls_target[batch_id, index, label] = 1
                            bbox_target[batch_id, index, 0] = (gx - ax) / aw
                            bbox_target[batch_id, index, 1] = (gy - ay) / ah
                            bbox_target[batch_id, index, 2] = (gw / aw).log()
                            bbox_target[batch_id, index, 3] = (gh / ah).log()

            else:  # top1
                # 每个gt 对应的最大的先验anchor
                scores, indexs = ious.max(0)
                assert indexs.unique().size(0) == indexs.size(0)  # 有可能出现 多个gt 对应 同一个 anchor
                for j, index in enumerate(indexs):
                    label = gt_label[j]
                    gx, gy, gw, gh = gt_xywh[j]
                    ax, ay, aw, ah = anchors_xywh[index]
                    cls_target[batch_id, index, label] = 1
                    bbox_target[batch_id, index, 0] = (gx - ax) / aw
                    bbox_target[batch_id, index, 1] = (gy - ay) / ah
                    bbox_target[batch_id, index, 2] = (gw / aw).log()
                    bbox_target[batch_id, index, 3] = (gh / ah).log()
                    object_target[batch_id, index, 0] = 1  # scores[j]

        avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            cls_target=cls_target,
            bbox_target=bbox_target,
            object_target=object_target
        )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # -----------------计算 rpn-----------------
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            # cls_score, bbox_pred, object_pred = self.model(imgs)
            cls_pred, bbox_pred = self.model(imgs)
            object_pred = cls_pred[..., [-1]]
            cls_score = cls_pred[..., :-1]

        # bs, _, h, w = cls_score.shape
        # bs = cls_score.size(0)

        feat_shape = (cls_score.shape, bbox_pred.shape, object_pred.shape)

        target_result, avg_factor = self.get_targets(imgs, targets, feat_shape)
        cls_target = target_result['cls_target']
        bbox_target = target_result['bbox_target']
        object_target = target_result['object_target']

        # loss
        keep = (object_target != -1).squeeze(-1)
        loss_centerness = GaussianFocalLoss()(object_pred.sigmoid()[keep], object_target[keep], avg_factor=avg_factor)

        loss_cls = CrossEntropyLoss(True)(cls_score, cls_target, (object_target == 1).expand_as(cls_target),
                                          avg_factor=avg_factor)

        loss_boxes = L1Loss()(bbox_pred, bbox_target, (object_target == 1).expand_as(bbox_target),
                              avg_factor=avg_factor)

        return dict(
            loss_centerness=loss_centerness,
            loss_cls=loss_cls,
            loss_boxes=loss_boxes)

    @torch.no_grad()
    def pred_step(self, imgs, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        # cls_score, bbox_pred, object_pred = self.model(imgs)
        cls_pred, bbox_pred = self.model(imgs)
        object_pred = cls_pred[..., -1]
        cls_score = cls_pred[..., :-1]
        # bs, na, fh, fw = object_pred.shape
        bs = object_pred.size(0)
        cls_score = cls_score.sigmoid()
        object_pred = object_pred.sigmoid()

        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(bbox_pred)
        # decode
        bbox_pred[..., :2] *= anchors_xywh[..., 2:]
        bbox_pred[..., :2] += anchors_xywh[..., :2]
        bbox_pred[..., 2:] = bbox_pred[..., 2:].exp()
        bbox_pred[..., 2:] *= anchors_xywh[..., 2:]

        batch_score, batch_index = torch.topk(object_pred, k=min(topk, object_pred.size(1)), dim=1)
        cls_score = torch.stack([cls_score[i][batch_index[i]] for i in range(bs)], 0)
        bbox_pred = torch.stack([bbox_pred[i][batch_index[i]] for i in range(bs)], 0)

        boxes = xywh2x1y1x2y2(bbox_pred)
        img_h, img_w = in_shape
        # 恢复到输入图像上
        boxes *= torch.tensor([img_w, img_h, img_w, img_h], device=self.device)[None, None]
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

        cls_score, cls_label = cls_score.max(-1)

        scores, labels = batch_score * cls_score, cls_label

        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if with_nms:
            keep = batched_nms(boxes, scores, labels, iou_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        # cls_score, bbox_pred, object_pred = self.model(imgs)
        cls_pred, bbox_pred = self.model(imgs)
        object_pred = cls_pred[..., -1]
        cls_score = cls_pred[..., :-1]
        # bs, na, fh, fw = object_pred.shape
        bs = object_pred.size(0)
        cls_score = cls_score.sigmoid()
        object_pred = object_pred.sigmoid()

        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(bbox_pred)
        # decode
        bbox_pred[..., :2] *= anchors_xywh[..., 2:]
        bbox_pred[..., :2] += anchors_xywh[..., :2]
        bbox_pred[..., 2:] = bbox_pred[..., 2:].exp()
        bbox_pred[..., 2:] *= anchors_xywh[..., 2:]

        batch_score, batch_index = torch.topk(object_pred, k=min(topk, object_pred.size(1)), dim=1)
        cls_score = torch.stack([cls_score[i][batch_index[i]] for i in range(bs)], 0)
        bbox_pred = torch.stack([bbox_pred[i][batch_index[i]] for i in range(bs)], 0)

        boxes = xywh2x1y1x2y2(bbox_pred)
        img_h, img_w = in_shape
        # 恢复到输入图像上
        boxes *= torch.tensor([img_w, img_h, img_w, img_h], device=self.device)[None, None]
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

        cls_score, cls_label = cls_score.max(-1)

        scores, labels = batch_score * cls_score, cls_label

        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if with_nms:
            keep = batched_nms(boxes, scores, labels, iou_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        return boxes, scores, labels


network = SSDMS(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(mode='coco', conf_threshold=0.2)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.3, with_nms=True, method=method)
