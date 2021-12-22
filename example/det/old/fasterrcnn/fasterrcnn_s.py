"""
# (每个正样本 可能对应 n个anchor n>=1)
1、每个anchor 对应于最大的gt 如果 iou>0.5为正，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# 参考yolo 改进为 (每个正样本 只对应 1个anchor)  本程序采用这种方式实现
1、每个anchor 对应于最大的gt 如果 iou>0.5忽略，否则为负
2、每个gt 对应的最大的先验anchor 为正样本

# rcnn 阶段 只做分类 不做 bbox 回归
# num_classes-1 对应背景

surrounding = True;epochs=200;freeze_at=4
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.353
--------------------------------------------------------------------------------
surrounding = False;epochs=200;freeze_at=4
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from timm.optim import RAdam

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, ciou_loss, \
    diou_loss  # , giou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.backbone import BackboneResnet
from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, RpnHead, RoiHeadC4, FasterRcnnBBoxHeadC4V2, _initParmas
# from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
# from toolcv.api.define.utils.data import augment as aug
# from toolcv.data.dataset import glob_format
from toolcv.utils.data.data import DataDefine, FruitsNutsDataset
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, _nms
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

# from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, xywh2x1y1x2y2, x1y1x2y22xywh
# from toolcv.api.define.utils.tools.anchor import rcnn_anchors, get_anchor_cfg
from toolcv.tools.anchor import getAnchorsV2_s

from toolcv.tools.anchor import kmean_gen_anchorv2, getAnchors, \
    generate_anchorsV3  # ,kmean_gen_anchorv2,kmean_gen_anchor,gen_anchor

# from toolcv.api.pytorch_lightning.net import get_params
# from toolcv.api.mobileNeXt.codebase.optim.radam import RAdam, PlainRAdam
# from toolcv.api.mobileNeXt.codebase.scheduler.plateau_lr import PlateauLRScheduler

ignore = True
surrounding = True
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
set_seed(seed)
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
"""
scales = [16, 32, 64, 128, 256, 512]  # base_sizes = scales
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
anchors = getAnchors((h // strides, w // strides), strides, base_anchors=base_anchors)  # [h,w,a,4] -->[-1,4]
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
num_classes = len(classes) + 1  # +1 表示 加上背景
device = get_device()
weight_path = 'weight.pth'
each_batch_scheduler = False  # 推荐 先 False 再 True
use_iouloss = False  # 推荐 先 False 再 True
save_path = 'output'
summary = SummaryWriter(save_path)

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
backbone = Backbone('resnet18', True, num_out=2, freeze_at=4)
# backbone = BackboneResnet("seresnext26t_32x4d",False,freeze_at=4,num_out=2)
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

load_weight(model, weight_path, "", device=device)

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
else:
    optimizer, scheduler = None, None

params = get_params(model[0].modules(), lr, weight_decay, 0.6)
if train_rpn:
    params += get_params(model[1].modules(), lr, weight_decay, 0.6)
if train_rcnn:
    params += get_params(model[3].modules(), lr, weight_decay, 0.6)
optimizer = RAdam(params, lr, weight_decay=weight_decay)


class FasterRcnn(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_rpn_targets(self, imgs, targets, feat_shape, feat_dtype):
        anchors_xywh = x1y1x2y22xywh(self.anchors)
        bs, _, img_h, img_w = imgs.shape
        cls_targets = imgs[0].new_zeros(feat_shape[0], dtype=feat_dtype)
        reg_targets = imgs[0].new_zeros(feat_shape[1], dtype=feat_dtype)
        cls_reg_weight = imgs[0].new_zeros(cls_targets.shape, dtype=feat_dtype)
        cls_reg_ious = torch.zeros_like(cls_reg_weight)

        feat_h, feat_w = math.ceil(img_h / strides), math.ceil(img_w / strides)

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

                dx = (gx - ax) / aw
                dy = (gy - ay) / ah
                dw = (gw / aw).log()
                dh = (gh / ah).log()
                reg_targets[batch_id, index, 0] = dx
                reg_targets[batch_id, index, 1] = dy
                reg_targets[batch_id, index, 2] = dw
                reg_targets[batch_id, index, 3] = dh

                if surrounding:
                    # 使用 centernet 方式
                    # 通过anchor 反算 heatmap 位置信息
                    cty_int = index // (feat_w * num_anchors)
                    ctx_int = index % (feat_w * num_anchors) // num_anchors
                    ca = index % (feat_w * num_anchors) % num_anchors
                    _cls_targets = cls_targets[batch_id].contiguous().view(feat_h, feat_w, num_anchors)

                    radius = gaussian_radiusv1([gh * feat_h, gw * feat_w], 0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(_cls_targets[:, :, ca], [ctx_int, cty_int], radius)
                    cls_targets[batch_id] = _cls_targets.contiguous().view(-1)
                else:
                    cls_targets[batch_id, index] = 1  # 作为centerness 分支

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
        feat_h, feat_w = math.ceil(img_h / strides), math.ceil(img_w / strides)
        _cls_preds = cls_preds.contiguous().view(bs, feat_h, feat_w, num_anchors).permute(0, 3, 1, 2)
        _cls_preds = _nms(_cls_preds)
        cls_preds = _cls_preds.permute(0, 2, 3, 1).contiguous().view(bs, -1)
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

            feat_dtype = rpn_cls_preds.dtype
            if train_rpn:
                target_result, avg_factor = self.get_rpn_targets(imgs, targets,
                                                                 (rpn_cls_preds.shape, rpn_reg_preds.shape), feat_dtype)
                rpn_cls_targets = target_result['cls_targets']
                rpn_reg_targets = target_result['reg_targets']
                rpn_cls_reg_weight = target_result['cls_reg_weight']
                rpn_cls_reg_ious = target_result['cls_reg_ious']

                # loss
                keep = (rpn_cls_reg_weight != -1).squeeze(-1)
                loss_rpn_centerness = gaussian_focal_loss(rpn_cls_preds.sigmoid()[keep],
                                                          rpn_cls_targets[keep]).sum() / avg_factor

                _cls_reg_weight = (rpn_cls_reg_weight[..., None] == 1).float().expand_as(rpn_reg_targets)
                loss_rpn_reg = (F.l1_loss(rpn_reg_preds, rpn_reg_targets, reduction="none") *
                                _cls_reg_weight).sum() / avg_factor

                losses.update(dict(
                    loss_rpn_centerness=loss_rpn_centerness,
                    loss_rpn_reg=loss_rpn_reg))

            if train_rcnn:
                proposals_dict = self.get_proposal(imgs, targets, rpn_cls_preds.detach(), rpn_reg_preds.detach())
                proposal = proposals_dict['proposal']
                # labels = torch.tensor(proposals_dict['labels'])
                cls_targets = torch.cat(proposals_dict['cls_targets'], 0)

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
                            criterion=None, train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader, each_batch_scheduler=each_batch_scheduler,
                            summary=summary))

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path='weight.pth', iou_threshold=0.3,
                       conf_threshold=0.2, with_nms=not surrounding, mode='coco'))
# -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data),
                       transform=test_transforms, device=device,
                       weight_path='weight.pth',
                       save_path='output', visual=False,
                       with_nms=not surrounding, iou_threshold=0.3,
                       conf_threshold=0.2, method="pad"))
