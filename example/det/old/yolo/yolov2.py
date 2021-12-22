"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.811
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.969
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
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
from toolcv.utils.tools.tools import glob_format  # , get_bboxes_DA as get_bboxes, xywh2x1y1x2y2
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

from toolcv.api.define.utils.tools.tools import get_bboxes_yolov3 as get_bboxes
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

from toolcv.tools.anchor import kmean_gen_anchorv2  # ,kmean_gen_anchorv2,kmean_gen_anchor,gen_anchor

ignore = True
surrounding = True

seed = 100
set_seed(seed)

# anchors = [[0.1, 0.1], [0.15, 0.15], [0.2, 0.2],
#            [0.25, 0.25], [0.3, 0.3], [0.35, 0.35],
#            [0.4, 0.4], [0.45, 0.45], [0.5, 0.5]]  # 缩放到0~1 [w,h]
anchors = [[0.10363208, 0.13572326],
           [0.17367187, 0.17468749],
           [0.17218749, 0.29986113],
           [0.15228449, 0.24137932],
           [0.24194446, 0.22580247]]
anchors = torch.tensor(anchors, dtype=torch.float32)

strides = 16
use_amp = True  # 推荐 先 True 再 False
accumulate = 2  # 推荐 先 >1 再 1
gradient_clip_val = 0.0 if use_amp else 1.0
lrf = 0.1
lr = 5e-4 if use_amp else 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
h, w = 416, 416
resize = (h, w)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
num_classes = len(classes)
device = get_device()
weight_path = 'weight.pth'
each_batch_scheduler = False  # 推荐 先 False 再 True
use_iouloss = False  # 推荐 先 False 再 True

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
head = YOLOFHead(backbone.out_channels // 4, len(anchors), num_classes)
_initParmas(spp.modules(), mode='kaiming')
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, spp, neck, head).to(device)
load_weight(model, weight_path, "", device=device)
model.train()

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
else:
    optimizer, scheduler = None, None


class YoloF(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None, each_batch_scheduler=False):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader,each_batch_scheduler)

    def get_targets(self, imgs, targets, feat_shape, feat_dtype):
        out_targets = []
        ratios = []
        bs, _, img_h, img_w = imgs.shape
        for shape in feat_shape:
            out_targets.append(imgs[0].new_zeros(shape, dtype=feat_dtype))
            feat_h, feat_w = shape[1:3]

            width_ratio = float(feat_w / img_w)  # 1/stride
            height_ratio = float(feat_h / img_h)  # 1/stride

            ratio = torch.tensor([width_ratio, height_ratio], dtype=imgs.dtype)
            ratios.append(ratio)

        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes']
            gt_label = targets[batch_id]['labels']
            gt_centers = (gt_bbox[..., :2] + gt_bbox[..., 2:]) / 2  # 还未缩放
            gt_wh = (gt_bbox[..., 2:] - gt_bbox[..., :2])  # 还未缩放
            gt_wh = gt_wh / torch.tensor([img_w, img_h], dtype=gt_wh.dtype)  # 统一缩放到0~1
            # 与先验anchor计算iou 分配正负样本(统一缩放到0~1)
            ious = wh_iou(self.anchors, gt_wh)
            # 每个gt 对应的最大的先验anchor
            scores, indexs = ious.max(0)

            for j, index in enumerate(indexs):
                n_layer = 0
                n_anchor = index
                ratio = ratios[n_layer]
                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]  # * ratio
                ind = gt_label[j]

                # 对应的先验anchor
                a_w, a_h = self.anchors[index]

                if surrounding:
                    fh, fw = out_targets[n_layer].shape[1:3]
                    radius = gaussian_radiusv1([scale_box_h * fh, scale_box_w * fw], 0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(out_targets[n_layer][batch_id, :, :, n_anchor, 0], [ctx_int, cty_int], radius)
                else:
                    out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 0] = 1
                out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 1] = ctx - ctx_int
                out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 2] = cty - cty_int
                # out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 3] = scale_box_w
                # out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 4] = scale_box_h
                out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 3] = (scale_box_w / a_w).log()
                out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 4] = (scale_box_h / a_h).log()
                out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 5 + ind] = 1

                # 忽略掉iou>0.5的非最大的anchor
                for idx, iou in enumerate(ious[:, j]):
                    if iou > 0.5 and idx != index.item():
                        n_layer = 0
                        n_anchor = idx
                        ratio = ratios[n_layer]
                        # 缩放到heatmap
                        ct = gt_centers[j] * ratio
                        ctx_int, cty_int = ct.int()
                        out_targets[n_layer][batch_id, cty_int, ctx_int, n_anchor, 0] = -1

        c = out_targets[0].size(-1)
        out_targets = torch.cat([out_target.contiguous().view(-1, c) for out_target in out_targets], 0)

        avg_factor = max(1, out_targets[:, 0].eq(1).sum())  # 正样本总个数

        return out_targets, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_score, bbox_pred, object_pred = self.model(imgs)
            bs, _, h, w = cls_score.shape
            feat_dtype = object_pred.dtype
            cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
            object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), 1)

            outputs = torch.cat((object_pred, bbox_pred, cls_score), -1)

            feat_shape = [outputs.shape]
            # get_target
            # target
            out_targets, avg_factor = self.get_targets(imgs, targets, feat_shape, feat_dtype)

            outputs = outputs.contiguous().view(-1, out_targets.size(-1))
            centerness_pred = outputs[..., [0]]
            offset_pred = outputs[..., 1:3]
            wh_pred = outputs[..., 3:5]
            cls_pred = outputs[..., 5:]

            centerness_target = out_targets[..., [0]]
            offset_target = out_targets[..., 1:3]
            wh_target = out_targets[..., 3:5]
            cls_target = out_targets[..., 5:]
            wh_offset_target_weight = (centerness_target == 1).float()

            # loss
            keep = centerness_target != -1
            loss_center_heatmap = gaussian_focal_loss(centerness_pred.sigmoid()[keep],
                                                      centerness_target[keep]).sum() / avg_factor

            loss_wh = (F.l1_loss(wh_pred, wh_target, reduction="none") *
                       wh_offset_target_weight.expand_as(wh_target)).sum() / avg_factor

            loss_offset = (F.l1_loss(offset_pred, offset_target, reduction="none") *
                           wh_offset_target_weight.expand_as(offset_target)).sum() / avg_factor

            loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                        wh_offset_target_weight.expand_as(cls_target)).sum() / avg_factor

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    @torch.no_grad()
    def pred_step(self, imgs, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        # outputs = self.model(imgs)
        cls_score, bbox_pred, object_pred = self.model(imgs)
        bs, _, h, w = cls_score.shape

        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
        object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), 1)

        outputs = [torch.cat((object_pred, bbox_pred, cls_score), -1)]

        boxes_list, scores_list, labels_list = [], [], []
        for j, output in enumerate(outputs):
            fh, fw = output.shape[1:3]
            centerness_preds = output[..., [0]]
            offset_preds = output[..., 1:3]
            wh_preds = output[..., 3:5]
            cls_preds = output[..., 5:]
            centerness_preds = centerness_preds.sigmoid()
            cls_preds = cls_preds.sigmoid()
            anchor = self.anchors.to(self.device).expand_as(wh_preds)
            wh_preds = wh_preds.exp() * anchor  # 0~1
            # 恢复到heatmap上
            wh_preds[..., 0] *= fw
            wh_preds[..., 1] *= fh

            boxes, scores, labels = get_bboxes(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                               iou_threshold, conf_threshold, scale_factors, padding, in_shape)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes_list = torch.cat(boxes_list, 0)
        scores_list = torch.cat(scores_list, 0)
        labels_list = torch.cat(labels_list, 0)

        # nms
        # keep = batched_nms(boxes_list, scores_list, labels_list, iou_threshold)
        # boxes, scores, labels = boxes_list[keep], scores_list[keep], labels_list[keep]
        boxes, scores, labels = boxes_list, scores_list, labels_list

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        cls_score, bbox_pred, object_pred = self.model(imgs)
        bs, _, h, w = cls_score.shape

        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), -1)
        object_pred = object_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, len(self.anchors), 1)

        outputs = [torch.cat((object_pred, bbox_pred, cls_score), -1)]

        boxes_list, scores_list, labels_list = [], [], []
        for j, output in enumerate(outputs):
            fh, fw = output.shape[1:3]
            centerness_preds = output[..., [0]]
            offset_preds = output[..., 1:3]
            wh_preds = output[..., 3:5]
            cls_preds = output[..., 5:]
            centerness_preds = centerness_preds.sigmoid()
            cls_preds = cls_preds.sigmoid()
            anchor = self.anchors.to(self.device).expand_as(wh_preds)
            wh_preds = wh_preds.exp() * anchor  # 0~1
            # 恢复到heatmap上
            wh_preds[..., 0] *= fw
            wh_preds[..., 1] *= fh

            boxes, scores, labels = get_bboxes(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                               iou_threshold, conf_threshold, scale_factors, padding, in_shape, False)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes_list = torch.cat(boxes_list, 0)
        scores_list = torch.cat(scores_list, 0)
        labels_list = torch.cat(labels_list, 0)

        # nms
        # keep = batched_nms(boxes_list, scores_list, labels_list, iou_threshold)
        # boxes, scores, labels = boxes_list[keep], scores_list[keep], labels_list[keep]
        boxes, scores, labels = boxes_list, scores_list, labels_list

        return boxes, scores, labels


network = YoloF(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                gamma=0.5, optimizer=optimizer, scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                each_batch_scheduler=each_batch_scheduler)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit(weight_path)
# -----------------------eval ---------------------------
network.evalute(mode='coco', conf_threshold=0.2, with_nms=True)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.1, with_nms=True, method='pad')
