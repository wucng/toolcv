"""
# (每个正样本 可能对应 n个anchor n>=1)
1、每个anchor 对应于最大的gt 如果 iou>0.5为正，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# 参考yolo 改进为 (每个正样本 只对应 1个anchor)  本程序采用这种方式实现
1、每个anchor 对应于最大的gt 如果 iou>0.5忽略，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# rcnn 阶段 只做分类 不做 bbox 回归
# num_classes-1 对应背景
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
import numpy as np

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

# from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, RpnHead, RoiHeadC4, FasterRcnnBBoxHeadC4V2, _initParmas
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
train_mode = 'rpn+rcnn'
if train_mode == 'rpn':
    train_rpn = True
    train_rcnn = False
    only_pred_rpn = True
elif train_mode == 'rcnn':
    train_rpn = False
    train_rcnn = True
    only_pred_rpn = False
else:
    train_rpn = True
    train_rcnn = True
    only_pred_rpn = False

seed = 100
torch.manual_seed(seed)
strides = 16

# h, w = 800, 1330  # 最大边1333；最小边800
h, w = 512, 512
resize = (h, w)

"""
featmap_sizes = [(32, 32)]
num_anchors = 9
cfg = dict(
    # type='AnchorGenerator',
    scales=[4, 8, 16],
    ratios=[0.5, 1.0, 2.0],
    strides=[strides])
anchors = rcnn_anchors(cfg, 512, 512, featmap_sizes, clip=True, normalization=True)
anchors = torch.cat(anchors, 0)
"""
scales = [16, 32, 64, 128, 256, 512]  # base_sizes = scales
ratios = [1.]
num_anchors = len(scales) * len(ratios)
anchors = getAnchorsV2_s(resize, strides, scales=scales, ratios=ratios)
anchors = torch.from_numpy(anchors).float()
# """

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
                                  transforms_mosaic=transforms_mosaic, h=h, w=w, mosaic_mode=0)
val_dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, use_mosaic=False,
                                transforms_mosaic=None, h=h, w=w)
dataloader = LoadDataloader(train_dataset, val_dataset, 0.0, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
backbone = Backbone('resnet18', True, num_out=2)
rpn = RpnHead(backbone.out_channels // 2, num_anchors)
roihead = RoiHeadC4(backbone.backbone[-1], strides)
bboxhead = FasterRcnnBBoxHeadC4V2(backbone.out_channels, num_classes)

_initParmas(rpn.modules(), mode='normal')
# _initParmas(roihead.modules(), mode='kaiming')
_initParmas(bboxhead.modules(), mode='normal')

model = nn.Sequential(backbone, rpn, roihead, bboxhead).to(device)

model.forward_backbone = lambda x: model[0](x)[0]
model.forward_rpn = lambda x: model[1](x)
model.forward_roipool = lambda x, proposal: model[2](x, proposal)
model.forward_rcnn = lambda x: model[3](x)

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

    def get_rpn_targets(self, imgs, targets, feat_shape):
        anchors_xywh = x1y1x2y22xywh(self.anchors)
        bs, _, img_h, img_w = imgs.shape
        cls_targets = imgs[0].new_zeros(feat_shape[0])
        reg_targets = imgs[0].new_zeros(feat_shape[1])
        cls_reg_weight = imgs[0].new_zeros(cls_targets.shape)
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
                    # label = gt_label[index]
                    # cls_targets[batch_id, j, label] = scores[j]
                    # cls_targets[batch_id, j] = scores[j] * 0.8  # 作为centerness 分支
                    v = scores[j] * 0.8
                    cls_targets[batch_id, j] = v if v > 0.3 else 0

            # 每个gt 对应的最大的先验anchor 为正样本
            scores, indexs = ious.max(0)
            for j, index in enumerate(indexs):
                gx, gy, gw, gh = gt_xywh[j]
                label = gt_label[j]
                ax, ay, ah, aw = anchors_xywh[index]

                cls_reg_weight[batch_id, index] = 1
                cls_reg_ious[batch_id, index] = scores[j]

                cls_targets[batch_id, index] = 1  # 作为centerness 分支

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

    @torch.no_grad()
    def get_proposal(self, imgs, targets, cls_preds, reg_preds, training=True):
        cls_preds = cls_preds.sigmoid()
        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(reg_preds)
        # decode
        reg_preds[..., :2] *= anchors_xywh[..., 2:]
        reg_preds[..., :2] += anchors_xywh[..., :2]
        reg_preds[..., 2:] = reg_preds[..., 2:].exp()
        reg_preds[..., 2:] *= anchors_xywh[..., 2:]

        bs, _, img_h, img_w = imgs.shape

        batch_score, batch_index = torch.topk(cls_preds.view(bs, -1), k=1000 if training else 500, dim=-1)

        reg_preds = torch.stack([reg_preds[i][batch_index[i]] for i in range(bs)], 0)

        # 恢复到输入图像上
        reg_preds[..., [0, 2]] *= img_w
        reg_preds[..., [1, 3]] *= img_h
        reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp(0, img_w - 1)
        reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp(0, img_h - 1)

        # xywh 2 x1y1x2y2
        boxes = xywh2x1y1x2y2(reg_preds)
        scores = batch_score

        proposal_list = []
        labels_list = []
        scores_list = []
        cls_targets = []

        for i, _boxes in enumerate(boxes):
            _scores = scores[i]
            # keep = scores[i] > 0.005
            # _boxes = _boxes[keep]
            # _scores = _scores[keep]

            # nms
            keep = batched_nms(_boxes, _scores, torch.ones_like(_scores), iou_threshold=0.7)
            keep = keep[:100]
            _scores = _scores[keep]
            _boxes = _boxes[keep]

            if training:
                gt_boxes = targets[i]['boxes'].to(self.device)
                gt_labels = targets[i]['labels'].to(self.device)

                ious = box_iou(_boxes, gt_boxes)
                # 预测框对应的最大gt
                values, indices = ious.max(1)
                # keep = values > 0.5  # 为正 否则为负样本

                cls_target = torch.ones([_boxes.size(0)], device=self.device) * (self.num_classes - 1)  # 默认都为背景

                for j, v in enumerate(values):
                    if v > 0.5:
                        label = gt_labels[indices[j]]
                        labels_list.append(label)
                        cls_target[j] = label

                cls_targets.append(cls_target)
            proposal_list.append(_boxes)
            scores_list.append(_scores)

        return dict(proposal=proposal_list, scores=scores_list, labels=labels_list,
                    cls_targets=cls_targets)

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        losses = {}
        # -----------------计算 rpn-----------------
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            c4 = self.model.forward_backbone(imgs)
            rpn_cls_preds, rpn_reg_preds = self.model.forward_rpn(c4)

        if train_rpn:
            target_result, avg_factor = self.get_rpn_targets(imgs, targets, (rpn_cls_preds.shape, rpn_reg_preds.shape))
            rpn_cls_targets = target_result['cls_targets']
            rpn_reg_targets = target_result['reg_targets']
            rpn_cls_reg_weight = target_result['cls_reg_weight']
            rpn_cls_reg_ious = target_result['cls_reg_ious']

            # loss
            keep = (rpn_cls_reg_weight != -1).squeeze(-1)
            loss_rpn_centerness = GaussianFocalLoss()(rpn_cls_preds.sigmoid()[keep], rpn_cls_targets[keep],
                                                      avg_factor=avg_factor)

            _cls_reg_weight = (rpn_cls_reg_weight[..., None] == 1).float().expand_as(rpn_reg_targets)
            loss_rpn_reg = L1Loss()(rpn_reg_preds, rpn_reg_targets, _cls_reg_weight, avg_factor=avg_factor)

            losses.update(dict(
                loss_rpn_centerness=loss_rpn_centerness,
                loss_rpn_reg=loss_rpn_reg))

        if train_rcnn:
            proposals_dict = self.get_proposal(imgs, targets, rpn_cls_preds.detach(), rpn_reg_preds.detach())
            proposal = proposals_dict['proposal']
            # labels = torch.tensor(proposals_dict['labels'])
            cls_targets = torch.cat(proposals_dict['cls_targets'], 0)

            with torch.cuda.amp.autocast(self.use_amp):
                roi_out = self.model.forward_roipool(c4, proposal)
                rcnn_out = self.model.forward_rcnn(roi_out)

            # keep = cls_targets != (self.num_classes - 1)
            # avg_factor = keep.sum()
            # labels = cls_targets[keep].long()

            # loss_rcnn_cls = CrossEntropyLoss()(rcnn_out, cls_targets.long(), avg_factor=avg_factor)  # 默认使用 softmax
            loss_rcnn_cls = F.cross_entropy(rcnn_out, cls_targets.long())

            losses.update(dict(
                loss_rcnn_cls=loss_rcnn_cls))

        return losses

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        in_shape = imgs.shape[-2:]
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        scale_factors = kwargs['scale_factors']
        padding = kwargs['padding']
        with_nms = kwargs['with_nms']

        preds = self.do_predict(imgs, iou_threshold, conf_threshold)
        # 缩放到原始图像上
        boxes = preds['boxes']
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]
        preds['boxes'] = boxes

        return preds

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        with_nms = kwargs['with_nms']

        imgs, targets = batch
        # in_shape = imgs.shape[-2:]
        imgs = torch.stack(imgs, 0).to(self.device)
        preds = self.do_predict(imgs, iou_threshold, conf_threshold)

        return preds

    def do_predict(self, imgs, iou_threshold, conf_threshold):
        c4 = self.model.forward_backbone(imgs)
        rpn_cls_preds, rpn_reg_preds = self.model.forward_rpn(c4)

        proposals_dict = self.get_proposal(imgs, None, rpn_cls_preds.detach(), rpn_reg_preds.detach(), False)
        proposal = proposals_dict['proposal']
        boxes_scores = proposals_dict['scores']

        if only_pred_rpn:
            boxes, scores, labels = proposal[0], boxes_scores[0], torch.zeros_like(boxes_scores[0])
            keep = scores > conf_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # nms
            keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            return dict(boxes=boxes, scores=scores, labels=labels)

        proposal_boxes = torch.cat(proposal, 0)
        boxes_scores = torch.cat(boxes_scores, 0)
        roi_out = self.model.forward_roipool(c4, proposal)
        rcnn_cls_pred = self.model.forward_rcnn(roi_out)
        cls_scores, labels = rcnn_cls_pred.softmax(-1).max(-1)
        # cls_scores, labels = rcnn_cls_pred.sigmoid().max(-1)
        scores = boxes_scores * cls_scores

        keep = torch.bitwise_and(scores > conf_threshold, labels != self.num_classes - 1)
        scores = scores[keep]
        labels = labels[keep]
        boxes = proposal_boxes[keep]

        # nms
        keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        return dict(boxes=boxes, scores=scores, labels=labels)


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
                       conf_threshold=0.1, with_nms=False, mode='coco'))
# -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data),
                       transform=test_transforms, device=device,
                       weight_path='weight.pth',
                       save_path='output', visual=False,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.1, method=method))
