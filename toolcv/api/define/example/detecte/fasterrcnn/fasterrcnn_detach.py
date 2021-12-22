"""
丢弃（舍弃 效果不好）

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
import numpy as np

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, RpnHead, RoiHeadC4, FasterRcnnBBoxHeadC4, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh
from toolcv.api.define.utils.tools.anchor import rcnn_anchors, get_anchor_cfg

ignore = False
train_rpn = True

rcnn_ignore = False
train_rcnn = True

only_pred_rpn = False

seed = 100
torch.manual_seed(seed)
strides = 16

featmap_sizes = [(32, 32)]
num_anchors = [9]
cfg = dict(
    # type='AnchorGenerator',
    scales=[4, 8, 16],
    ratios=[0.5, 1.0, 2.0],
    strides=[strides])
anchors = rcnn_anchors(cfg, 512, 512, featmap_sizes, clip=True, normalization=True)
anchors = torch.cat(anchors, 0)

use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
# h, w = 800, 1330  # 最大边1333；最小边800
h, w = 512, 512
resize = (h, w)
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
rpn = RpnHead(backbone.out_channels // 2, num_anchors[0])
roihead = RoiHeadC4(backbone.backbone[-1], strides)
bboxhead = FasterRcnnBBoxHeadC4(backbone.out_channels, num_classes)

_initParmas(rpn.modules(), mode='normal')
_initParmas(roihead.modules(), mode='kaiming')
_initParmas(bboxhead.modules(), mode='normal')


class Model(nn.Module):
    def __init__(self, backbone, rpn, roihead, bboxhead):
        super().__init__()
        self.backbone, self.rpn, self.roihead, self.bboxhead = backbone, rpn, roihead, bboxhead

    def forward_backbone(self, x):
        c4, _ = self.backbone(x)
        return c4

    def forward_rpn(self, c4):
        rpn_out = self.rpn(c4)
        return rpn_out

    def forward_roipool(self, c4, proposal):
        """proposal = torch.tensor([[10, 20, 100, 100], [15, 40, 130, 170]]).float()"""
        roi_out = self.roihead(c4, [proposal])
        return roi_out

    def forward_rcnn(self, roi_out):
        rcnn_out = bboxhead(roi_out)
        return rcnn_out


model = Model(backbone, rpn, roihead, bboxhead).to(device)


class FasterRcnn(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

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

    def get_proposal(self, imgs, targets, cls_preds, reg_preds, training=True):
        cls_preds = cls_preds.sigmoid()
        anchors_xywh = (x1y1x2y22xywh(self.anchors).to(self.device)[None]).expand_as(reg_preds)
        # decode
        reg_preds[..., :2] *= anchors_xywh[..., 2:]
        reg_preds[..., :2] += anchors_xywh[..., :2]
        reg_preds[..., 2:] = reg_preds[..., 2:].exp()
        reg_preds[..., 2:] *= anchors_xywh[..., 2:]

        bs, _, img_h, img_w = imgs.shape

        batch_score, batch_index = torch.topk(cls_preds.view(bs, -1), k=2000 if training else 1000, dim=-1)

        reg_preds = torch.stack([reg_preds[i][batch_index[i]] for i in range(bs)], 0)

        # 恢复到输入图像上
        reg_preds[..., [0, 2]] *= img_w
        reg_preds[..., [1, 3]] *= img_h
        reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp(0, img_w - 1)
        reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp(0, img_h - 1)

        # xywh 2 x1y1x2y2
        boxes = xywh2x1y1x2y2(reg_preds)
        scores = batch_score

        proposal = []
        scores_list = []
        cls_targets = []
        reg_targets = []
        cls_reg_weights = []
        cls_reg_ious = []

        for batch_id in range(bs):
            _scores = scores[batch_id]
            _boxes = boxes[batch_id]
            keep = _scores > 0.05 if training else _scores > 0.3
            _scores = _scores[keep]
            _boxes = _boxes[keep]

            # nms
            keep = batched_nms(_boxes, _scores, torch.ones_like(_scores), iou_threshold=0.5 if training else 0.3)
            keep = keep[:100 if training else 50]
            _scores = _scores[keep]
            _boxes = _boxes[keep]

            if training:
                cls_target = torch.zeros([_boxes.size(0), self.num_classes])
                cls_target[:, -1] = 1  # 对应背景
                reg_target = torch.zeros([_boxes.size(0), (self.num_classes - 1) * 4])  # [:,self.num_classes - 1,4]
                cls_reg_weight = torch.zeros([_boxes.size(0)])
                cls_reg_iou = torch.zeros([_boxes.size(0)])

                # 再与gt box 计算iou 筛选正负样本
                gt_bbox = targets[batch_id]['boxes'].to(self.device)
                gt_label = targets[batch_id]['labels']

                ious = box_iou(_boxes, gt_bbox)

                gt_xywh = x1y1x2y22xywh(gt_bbox)
                anchors_xywh = x1y1x2y22xywh(_boxes)

                # 每个anchor 对应于最大的gt
                scores_ious, indexs = ious.max(1)
                # iou > 0.5 的 忽略
                if rcnn_ignore:
                    keep = scores_ious > 0.5
                    cls_reg_weight[keep] = -1  # 忽略
                else:
                    for j, index in enumerate(indexs):
                        label = gt_label[index]
                        v = scores_ious[j] * 0.8
                        if v > 0.3:
                            cls_target[j, label] = v
                            cls_target[j, -1] = 0

                # 每个gt 对应的最大的先验anchor 为正样本
                scores_ious, indexs = ious.max(0)

                for j, index in enumerate(indexs):
                    gx, gy, gw, gh = gt_xywh[j]
                    label = gt_label[j]
                    ax, ay, ah, aw = anchors_xywh[index]

                    cls_reg_weight[index] = 1
                    cls_reg_iou[index] = scores_ious[j]

                    cls_target[index, label] = 1
                    cls_target[index, -1] = 0  # 不是背景

                    dx = (gx - ax) / aw
                    dy = (gy - ay) / ah
                    dw = (gw / aw).log()
                    dh = (gh / ah).log()
                    reg_target[index, label * 4 + 0] = dx
                    reg_target[index, label * 4 + 1] = dy
                    reg_target[index, label * 4 + 2] = dw
                    reg_target[index, label * 4 + 3] = dh

                # pos_num = (cls_reg_weight == 1).sum().item()
                # neg_num = (cls_reg_weight == 0).sum().item()
                #
                # pos_keep = torch.nonzero(cls_reg_weight == 1).squeeze(-1).cpu().numpy()
                # if neg_num > 2 * pos_num:
                #     a = torch.nonzero(cls_reg_weight == 0).squeeze(-1)
                #     neg_keep = np.random.choice(a, 2 * pos_num, replace=False)
                # else:
                #     neg_keep = torch.nonzero(cls_reg_weight == 0).squeeze(-1).cpu().numpy()
                #
                # keep = np.concatenate((pos_keep, neg_keep), 0)
                # keep.sort()

                """
                # 选择正负样本 一共 32个 其中正样本 1/2 负样本 1/2
                pos_num = (cls_reg_weight == 1).sum().item()
                neg_num = (cls_reg_weight == 0).sum().item()

                if pos_num > 16:
                    a = torch.nonzero(cls_reg_weight == 1).squeeze(-1)
                    pos_keep = np.random.choice(a, 16, replace=False)
                    pos_num = 16
                else:
                    pos_keep = torch.nonzero(cls_reg_weight == 1).squeeze(-1).cpu().numpy()

                # 负样本
                a = torch.nonzero(cls_reg_weight == 0).squeeze(-1)
                neg_keep = np.random.choice(a, 32 - pos_num, replace=False)

                keep = np.concatenate((pos_keep, neg_keep), 0)
                keep.sort()
                """
                keep = cls_reg_weight != -1
                # """

                cls_targets.append(cls_target[keep])
                reg_targets.append(reg_target[keep])
                cls_reg_weights.append(cls_reg_weight[keep])
                cls_reg_ious.append(cls_reg_iou[keep])

                proposal.append(_boxes[keep])
                scores_list.append(_scores[keep])

            else:
                proposal.append(_boxes)
                scores_list.append(_scores)

        if training:
            return proposal, scores_list, cls_targets, reg_targets, cls_reg_weights, cls_reg_ious
        else:
            return proposal, scores_list

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
            proposal, scores_list, cls_targets, reg_targets, cls_reg_weights, cls_reg_ious = \
                self.get_proposal(imgs,
                                  targets,
                                  rpn_cls_preds.detach(),
                                  rpn_reg_preds.detach())

            bs = len(proposal)
            with torch.cuda.amp.autocast(self.use_amp):
                # roi_out = self.model.forward_roipool(c4, proposal)
                roi_out = [self.model.forward_roipool(c4, proposal[i]) for i in range(bs)]
                rcnn_out = [self.model.forward_rcnn(out) for out in roi_out]

            loss_rcnn_cls = Variable(torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True))
            loss_rcnn_reg = Variable(torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True))

            for i in range(bs):
                rcnn_cls_pred, rcnn_reg_pred = rcnn_out[i]
                cls_target, reg_target = cls_targets[i], reg_targets[i]
                reg_weight = cls_reg_weights[i] == 1
                avg_factor = reg_weight.sum()
                loss_rcnn_cls = loss_rcnn_cls + CrossEntropyLoss(True)(rcnn_cls_pred, cls_target.to(self.device),
                                                                       avg_factor=avg_factor)  # 自动加上 sigmoid
                loss_rcnn_reg = loss_rcnn_reg + L1Loss()(rcnn_reg_pred, reg_target.to(self.device),
                                                         reg_weight[..., None].expand_as(reg_target).to(self.device),
                                                         avg_factor=avg_factor)

            losses.update(dict(
                loss_rcnn_cls=loss_rcnn_cls / bs,
                loss_rcnn_reg=loss_rcnn_reg / bs))

        return losses

    @torch.no_grad()
    def pred_step(self, imgs, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        c4 = self.model.forward_backbone(imgs)
        rpn_cls_preds, rpn_reg_preds = self.model.forward_rpn(c4)

        proposal, boxes_scores = self.get_proposal(imgs, None, rpn_cls_preds.detach(), rpn_reg_preds.detach(), False)

        if only_pred_rpn:
            boxes, scores, labels = proposal[0], boxes_scores[0], torch.zeros_like(boxes_scores[0])
            keep = scores > conf_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # nms
            keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # 缩放到原始图像上
            boxes -= torch.tensor(padding, device=boxes.device)[None]
            boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

            return boxes, scores, labels

        bs = len(proposal)
        roi_out = [self.model.forward_roipool(c4, proposal[i]) for i in range(bs)]
        rcnn_out = [self.model.forward_rcnn(out) for out in roi_out]

        boxes_list, scores_list, labels_list = [], [], []
        for i in range(bs):
            rcnn_cls_pred, rcnn_reg_pred = rcnn_out[i]
            proposal_boxes = proposal[i]
            proposal_scores = boxes_scores[i]
            # cls_scores, labels = rcnn_cls_pred.softmax(-1).max(-1)
            cls_scores, labels = rcnn_cls_pred.sigmoid().max(-1)
            scores = proposal_scores * cls_scores

            keep = torch.bitwise_and(scores > conf_threshold, labels != self.num_classes - 1)
            scores = scores[keep]
            labels = labels[keep]
            rcnn_reg_pred = rcnn_reg_pred[keep].contiguous().view(-1, self.num_classes - 1, 4)
            rcnn_reg_pred = torch.stack([rcnn_reg_pred[i, label] for i, label in enumerate(labels)], 0)
            proposal_boxes = proposal_boxes[keep]

            anchors_xywh = x1y1x2y22xywh(proposal_boxes)
            rcnn_reg_pred[..., :2] *= anchors_xywh[..., 2:]
            rcnn_reg_pred[..., :2] += anchors_xywh[..., :2]
            rcnn_reg_pred[..., 2:] = rcnn_reg_pred[..., 2:].exp()
            rcnn_reg_pred[..., 2:] *= anchors_xywh[..., 2:]

            boxes = xywh2x1y1x2y2(rcnn_reg_pred)

            # nms
            keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # 缩放到原始图像上
            boxes -= torch.tensor(padding, device=boxes.device)[None]
            boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes = torch.stack(boxes_list, 0)[0]
        scores = torch.stack(scores_list, 0)[0]
        labels = torch.stack(labels_list, 0)[0]

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        c4 = self.model.forward_backbone(imgs)
        rpn_cls_preds, rpn_reg_preds = self.model.forward_rpn(c4)

        proposal, boxes_scores = self.get_proposal(imgs, None, rpn_cls_preds.detach(), rpn_reg_preds.detach(), False)

        if only_pred_rpn:
            boxes, scores, labels = proposal[0], boxes_scores[0], torch.zeros_like(boxes_scores[0])
            keep = scores > conf_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # nms
            keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # 缩放到原始图像上
            # boxes -= torch.tensor(padding, device=boxes.device)[None]
            # boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

            return boxes, scores, labels

        bs = len(proposal)
        roi_out = [self.model.forward_roipool(c4, proposal[i]) for i in range(bs)]
        rcnn_out = [self.model.forward_rcnn(out) for out in roi_out]

        boxes_list, scores_list, labels_list = [], [], []
        for i in range(bs):
            rcnn_cls_pred, rcnn_reg_pred = rcnn_out[i]
            proposal_boxes = proposal[i]
            proposal_scores = boxes_scores[i]
            cls_scores, labels = rcnn_cls_pred.sigmoid().max(-1)
            scores = proposal_scores * cls_scores

            keep = torch.bitwise_and(scores > conf_threshold, labels != self.num_classes - 1)
            scores = scores[keep]
            labels = labels[keep]
            rcnn_reg_pred = rcnn_reg_pred[keep].contiguous().view(-1, self.num_classes - 1, 4)
            rcnn_reg_pred = torch.stack([rcnn_reg_pred[i, label] for i, label in enumerate(labels)], 0)
            proposal_boxes = proposal_boxes[keep]

            anchors_xywh = x1y1x2y22xywh(proposal_boxes)
            rcnn_reg_pred[..., :2] *= anchors_xywh[..., 2:]
            rcnn_reg_pred[..., :2] += anchors_xywh[..., :2]
            rcnn_reg_pred[..., 2:] = rcnn_reg_pred[..., 2:].exp()
            rcnn_reg_pred[..., 2:] *= anchors_xywh[..., 2:]

            boxes = xywh2x1y1x2y2(rcnn_reg_pred)

            # nms
            keep = batched_nms(boxes, scores, labels, iou_threshold=iou_threshold)

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # 缩放到原始图像上
            # boxes -= torch.tensor(padding, device=boxes.device)[None]
            # boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes = torch.stack(boxes_list, 0)[0]
        scores = torch.stack(scores_list, 0)[0]
        labels = torch.stack(labels_list, 0)[0]

        return boxes, scores, labels


network = FasterRcnn(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                     gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                     gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                     train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
# network.evalute(mode='coco', conf_threshold=0.2)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.1, with_nms=True, method=method)
