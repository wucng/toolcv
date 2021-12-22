"""
使用单个分支的 ssd  single stage

# (每个正样本 可能对应 n个anchor n>=1)
1、每个anchor 对应于最大的gt 如果 iou>0.5为正，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# 参考yolo 改进为 (每个正样本 只对应 1个anchor) 本程序采用这种方式实现
1、每个anchor 对应于最大的gt 如果 iou>0.5忽略，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

ignore = True;top5 = False;surrounding = False;epochs=100 (13it/s)
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.728
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.982
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.915
---------------------------------------------------------------------------------
ignore = True;top5 = False;surrounding = True;epochs=100 (6it/s)
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.687
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.972
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.883
---------------------------------------------------------------------------------
ignore = True;top5 = True;surrounding = True;epochs=100 (6it/s)
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.631
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.957
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.784
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
import numpy as np
from torch.nn import functional as F

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, ciou_loss, \
    diou_loss  # , giou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHead, _initParmas
# from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
# from toolcv.api.define.utils.data import augment as aug
# from toolcv.data.dataset import glob_format
from toolcv.utils.data.data import DataDefine, FruitsNutsDataset
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, _nms  # , get_bboxes_DA as get_bboxes
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight
from toolcv.tools.anchor import kmean_gen_anchorv2, getAnchors, generate_anchorsV3

from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh

# from toolcv.api.define.utils.tools.tools import get_bboxesv3  # , grid_torch

# from toolcv.api.define.utils.tools.anchor import rcnn_anchors, get_anchor_cfg
# from toolcv.tools.anchor import getAnchorsV2_s

ignore = True
top5 = False
surrounding = False

seed = 100
set_seed(seed)
strides = 16

# h, w = 800, 1330  # 最大边1333；最小边800
# h, w = 512, 512
h, w = 300, 300
resize = (h, w)

"""
featmap_sizes = [(math.ceil(h / strides), math.ceil(w / strides))]
cfg = dict(
    # type='AnchorGenerator',
    scales=[4, 8, 16],  # , 32   # base_sizes = scales*strides
    ratios=[0.5, 1.0, 2.0],
    strides=[strides])
num_anchors = len(cfg['scales']) * len(cfg['ratios'])
anchors = rcnn_anchors(cfg, h, w, featmap_sizes, clip=True, normalization=True)
anchors = torch.cat(anchors, 0)
# """
"""
scales = [32, 64, 128, 256]  # base_sizes = scales
ratios = [1.]
num_anchors = len(scales) * len(ratios)
anchors = getAnchorsV2_s(resize, strides, scales=scales, ratios=ratios)
anchors = torch.from_numpy(anchors).float()
# """
anchors = [[0.10363208, 0.13572326],
           [0.17367187, 0.17468749],
           [0.17218749, 0.29986113],
           [0.15228449, 0.24137932],
           [0.24194446, 0.22580247]]
num_anchors = len(anchors)
base_anchors = generate_anchorsV3(list(resize), np.array(anchors))
anchors = getAnchors((math.ceil(h / strides), math.ceil(w / strides)), strides,
                     base_anchors=base_anchors)  # [h,w,a,4] -->[-1,4]
anchors = torch.tensor(anchors, dtype=torch.float32)

use_amp = True  # 推荐 先 True 再 False
accumulate = 2  # 推荐 先 >1 再 1
gradient_clip_val = 0.0 if use_amp else 1.0
lrf = 0.1
lr = 5e-4 if use_amp else 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
num_classes = len(classes)
device = get_device()
weight_path = 'weight.pth'
each_batch_scheduler = False  # 推荐 先 False 再 True
use_iouloss = False  # 推荐 先 False 再 True
enable_gradscaler = True

# 通过聚类生成先验框
# kmean_gen_anchorv2(FruitsNutsDataset(dir_data,classes),5)
# exit(0)
# wh [[0.10363208 0.13572326]
#  [0.17367187 0.17468749]
#  [0.17218749 0.29986113]
#  [0.15228449 0.24137932]
#  [0.24194446 0.22580247]]

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
data.get_dataloader()
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# ----------------model --------------------------
backbone = Backbone('resnet18', True, stride=strides)
spp = SPPv2(backbone.out_channels, backbone.out_channels // 4)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHead(backbone.out_channels // 4, num_anchors, num_classes)
_initParmas(spp.modules(), mode='kaiming')
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, spp, neck, head).to(device)
load_weight(model, weight_path, "", device=device)

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
else:
    optimizer, scheduler = None, None


class SSDSS(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None,
                 each_batch_scheduler=False, enable_gradscaler=True):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader, each_batch_scheduler, enable_gradscaler)

    def get_targets(self, imgs, targets, feat_shape, feat_dtype):
        anchors_xywh = x1y1x2y22xywh(self.anchors)
        cls_target, bbox_target, object_target = imgs[0].new_zeros(feat_shape[0], dtype=feat_dtype), \
                                                 imgs[0].new_zeros(feat_shape[1], dtype=feat_dtype), \
                                                 imgs[0].new_zeros(feat_shape[2], dtype=feat_dtype)

        feat_h, feat_w = feat_shape[-1]

        bs, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride
        ratio = torch.tensor([width_ratio, height_ratio], dtype=imgs.dtype)

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
                        if values[i, j] < 0.3: continue
                        index = indices[i, j]
                        label = gt_label[j]
                        if object_target[batch_id, index, 0] == 1:  # 重复了
                            object_target[batch_id, index, 0] = -1
                        else:
                            gx, gy, gw, gh = gt_xywh[j]
                            ax, ay, aw, ah = anchors_xywh[index]
                            cls_target[batch_id, index, label] = 1
                            bbox_target[batch_id, index, 0] = (gx - ax) / aw
                            bbox_target[batch_id, index, 1] = (gy - ay) / ah
                            bbox_target[batch_id, index, 2] = (gw / aw).log()
                            bbox_target[batch_id, index, 3] = (gh / ah).log()

                            if surrounding:
                                # 使用 centernet 方式
                                # 通过anchor 反算 heatmap 位置信息
                                cty_int = index // (feat_w * num_anchors)
                                ctx_int = index % (feat_w * num_anchors) // num_anchors
                                ca = index % (feat_w * num_anchors) % num_anchors
                                _object_target = object_target[batch_id].contiguous().view(feat_h, feat_w, num_anchors,
                                                                                           1)

                                radius = gaussian_radiusv1([gh * feat_h, gw * feat_w], 0.3)
                                radius = max(0, int(radius))
                                gen_gaussian_target(_object_target[:, :, ca, 0], [ctx_int, cty_int], radius)

                                object_target[batch_id] = _object_target.contiguous().view(-1, 1)

                            else:
                                # yolo方式（只标记中心点）
                                object_target[batch_id, index, 0] = 1  # scores[j]

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
                    if surrounding:
                        # 使用 centernet 方式
                        # 通过anchor 反算 heatmap 位置信息
                        cty_int = index // (feat_w * num_anchors)
                        ctx_int = index % (feat_w * num_anchors) // num_anchors
                        ca = index % (feat_w * num_anchors) % num_anchors
                        _object_target = object_target[batch_id].contiguous().view(feat_h, feat_w, num_anchors, 1)

                        radius = gaussian_radiusv1([gh * feat_h, gw * feat_w], 0.3)
                        radius = max(0, int(radius))
                        gen_gaussian_target(_object_target[:, :, ca, 0], [ctx_int, cty_int], radius)

                        object_target[batch_id] = _object_target.contiguous().view(-1, 1)

                    else:
                        # yolo方式（只标记中心点）
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
            cls_score, bbox_pred, object_pred = self.model(imgs)

            bs, _, h, w = cls_score.shape
            # cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, h, w, num_anchors, -1)
            # bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, num_anchors, -1)
            # object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, num_anchors, 1)
            cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
            object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)

            feat_shape = (cls_score.shape, bbox_pred.shape, object_pred.shape, (h, w))
            feat_dtype = cls_score.dtype

            target_result, avg_factor = self.get_targets(imgs, targets, feat_shape, feat_dtype)
            cls_target = target_result['cls_target']
            bbox_target = target_result['bbox_target']
            object_target = target_result['object_target']

            # loss
            keep = (object_target != -1).squeeze(-1)
            loss_centerness = gaussian_focal_loss(object_pred.sigmoid()[keep],
                                                  object_target[keep]).sum() / avg_factor

            loss_cls = (F.binary_cross_entropy_with_logits(cls_score, cls_target, reduction="none") *
                        (object_target == 1).expand_as(cls_target)).sum() / avg_factor

            loss_boxes = (F.l1_loss(bbox_pred, bbox_target, reduction="none") *
                          (object_target == 1).expand_as(bbox_target)).sum() / avg_factor

        return dict(
            loss_centerness=loss_centerness,
            loss_cls=loss_cls,
            loss_boxes=loss_boxes)

    @torch.no_grad()
    def pred_step(self, imgs, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        cls_score, bbox_pred, object_pred = self.model(imgs)
        bs, na, fh, fw = object_pred.shape
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cls_score = cls_score.sigmoid()
        object_pred = object_pred.sigmoid()
        object_pred = _nms(object_pred)
        object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1)

        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(bbox_pred)
        # decode
        bbox_pred[..., :2] *= anchors_xywh[..., 2:]
        bbox_pred[..., :2] += anchors_xywh[..., :2]
        bbox_pred[..., 2:] = bbox_pred[..., 2:].exp()
        bbox_pred[..., 2:] *= anchors_xywh[..., 2:]

        batch_score, batch_index = torch.topk(object_pred, k=min(100, object_pred.size(1)), dim=1)
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

        cls_score, bbox_pred, object_pred = self.model(imgs)
        bs, na, fh, fw = object_pred.shape
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        cls_score = cls_score.sigmoid()
        object_pred = object_pred.sigmoid()
        object_pred = _nms(object_pred)
        object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1)

        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(bbox_pred)
        # decode
        bbox_pred[..., :2] *= anchors_xywh[..., 2:]
        bbox_pred[..., :2] += anchors_xywh[..., :2]
        bbox_pred[..., 2:] = bbox_pred[..., 2:].exp()
        bbox_pred[..., 2:] *= anchors_xywh[..., 2:]

        batch_score, batch_index = torch.topk(object_pred, k=min(100, object_pred.size(1)), dim=1)
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


network = SSDSS(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                gamma=0.5, optimizer=optimizer, scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                each_batch_scheduler=each_batch_scheduler, enable_gradscaler=enable_gradscaler)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(mode='coco', conf_threshold=0.2, with_nms=not surrounding)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.2, with_nms=not surrounding, method="pad")
