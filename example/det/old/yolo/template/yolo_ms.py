"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
yolov3方式 计算 wh—iou 选择 anchor

可以 是 单分支 也 可以是多个分支（ strides = [8, 16, 32] or [8, 16, 32, 64, 128] or [8, 16, 32, 64, 128,256],....）
strides = [8] or [16] or [32] ....
strides = [8,16] or [8,32] or [16,32] ....
strides = [8, 16, 32] or [8, 16, 32, 64, 128] or [8, 16, 32, 64, 128,256],...
strides = [8, 32, 128] or [16, 64, 256],...
--------------------------------------------------------------------------------
strides = [8, 16, 32];global_max = True;epochs=200
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.819
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.991
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.991
--------------------------------------------------------------------------------
strides = [8, 16, 32];global_max = False;epochs=200
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.717
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.909
--------------------------------------------------------------------------------
strides = [16];global_max = True;epochs=200
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.746
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.991
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.956
--------------------------------------------------------------------------------
strides = [8, 16, 32,64,128];global_max = True;epochs=200
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.902
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.989
--------------------------------------------------------------------------------
strides = [16, 32];global_max = True;epochs=200
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.744
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.901
--------------------------------------------------------------------------------
strides = [16, 32];global_max = True;epochs=250
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.804
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.972
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, giou_loss, ciou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.backbone import BackboneCspnet, BackboneResnet
from toolcv.utils.net.neck import FPN, PAN
from toolcv.utils.net.head import YOLOFHeadSelfMS
from toolcv.utils.net.net import SPPv2, _initParmasV2
from toolcv.utils.net.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, _initParmas, FPN as FPN2
# from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
# from toolcv.api.define.utils.data import augment as aug
from toolcv.utils.data.data import DataDefine, FruitsNutsDataset
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, get_bboxes_anchors as get_bboxes
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

# from toolcv.api.define.utils.tools.tools import get_bboxesv3
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2
from toolcv.tools.anchor import kmean_gen_anchorv2


# from toolcv.api.define.utils.data.data import FruitsNutsDataset

class FilterOutput(nn.Module):
    def __init__(self, index=[0, 1, 2]):
        super().__init__()
        self.index = index

    def forward(self, x):
        out = []
        for i in self.index:
            out.append(x[i])
        return out


# ignore = True
surrounding = True
global_max = True

seed = 100
set_seed(seed)

# strides = [16]
# strides = [16, 32]
strides = [8, 16, 32]
# strides = [8, 16, 32,64,128]


# anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
# anchors = torch.tensor(anchors, dtype=torch.float32)
# anchors = torch.stack((anchors, anchors), -1)

# 通过聚类生成先验框
if len(strides) == 1:
    anchors = [[0.10363208, 0.13572326], [0.17367187, 0.17468749], [0.15228449, 0.24137932],
               [0.17218749, 0.29986113], [0.24194446, 0.22580247]]
elif len(strides) == 2:
    anchors = [[0.10190789, 0.12438597], [0.10437501, 0.1759375], [0.15151316, 0.16535087], [0.19992188, 0.1828125],
               [0.14776042, 0.2657639], [0.18328947, 0.23570174], [0.254375, 0.2255], [0.18423077, 0.3155128]]
elif len(strides) == 3:
    anchors = [[0.0971, 0.1149], [0.1062, 0.1524], [0.1513, 0.1686],
               [0.1355, 0.2330], [0.1999, 0.1828], [0.1584, 0.2816],
               [0.1904, 0.2393], [0.2544, 0.2255], [0.1911, 0.3336]]
elif len(strides) == 5:
    anchors = [[0.05, 0.05], [0.1, 0.1], [0.15, 0.15],
               [0.2, 0.2], [0.25, 0.25], [0.3, 0.3],
               [0.35, 0.35], [0.4, 0.4], [0.45, 0.45],
               [0.5, 0.5], [0.55, 0.55], [0.6, 0.6],
               [0.65, 0.65], [0.7, 0.7], [0.8, 0.8]]

anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)

n_anchors = len(anchors) // len(strides)
use_amp = False  # 推荐 先 True 再 False
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
save_path = 'output'
summary = SummaryWriter(save_path)

# 通过聚类生成先验框
# kmean_gen_anchorv2(FruitsNutsDataset(dir_data,classes),5)
# exit(0)

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
data.get_dataloader()
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# ----------------model --------------------------
activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
if len(strides) == 5:  # strides=[8,16,32,64,128]
    backbone = Backbone('resnet18', True, num_out=3)
    neck = nn.Sequential(
        SPPv2(backbone.out_channels, backbone.out_channels // 4),
        FPN2(backbone.out_channels, 256)
    )
    head = YOLOFHeadSelfMS([256, 256, 256, 256, 256], 1, n_anchors, num_classes, False, activate, True, share=True,
                           do_wh_cls=False)

elif len(strides) == 3:  # strides=[8,16,32]
    backbone = Backbone('resnet18', True, num_out=3)
    # backbone = BackboneCspnet("cspdarknet53", True, num_out=3)
    # backbone = BackboneResnet("resnet18", False, num_out=3)
    out_channels = backbone.out_channels
    neck = nn.Sequential(
        SPPv2(out_channels, out_channels // 2),
        FPN([out_channels // 4, out_channels // 2, out_channels], [256, 256, 256], activate, "cat"),
        PAN([256, 256, 256], [256, 256, 256], activate, 'cat')
    )
    head = YOLOFHeadSelfMS([256, 256, 256], 1, n_anchors, num_classes, False, activate, True, share=True,
                           do_wh_cls=False)

elif len(strides) == 2:  # strides=[16,32]
    backbone = Backbone('resnet18', True, num_out=3)
    # backbone = BackboneCspnet("cspdarknet53", True, num_out=3)
    # backbone = BackboneResnet("resnet18", False, num_out=3)
    out_channels = backbone.out_channels
    neck = nn.Sequential(
        SPPv2(out_channels, out_channels // 2),
        FPN([out_channels // 4, out_channels // 2, out_channels], [256, 256, 256], activate, "cat"),
        PAN([256, 256, 256], [256, 256, 256], activate, 'cat'), FilterOutput([1, 2])
    )
    head = YOLOFHeadSelfMS([256, 256], 1, n_anchors, num_classes, False, activate, True, share=True, do_wh_cls=False)

elif len(strides) == 1:
    backbone = Backbone('resnet18', True, num_out=1, stride=strides[0])
    neck = nn.Sequential(SPPv2(backbone.out_channels, backbone.out_channels // 4), YOLOFNeck(backbone.out_channels))
    head = YOLOFHeadSelfMS(backbone.out_channels // 4, 1, n_anchors, num_classes, False, activate, True, share=True,
                           do_wh_cls=False)

_initParmasV2(neck.modules(), mode='kaiming_normal')
_initParmasV2(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)
load_weight(model, weight_path, "", device=device)
model.train()

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
else:
    optimizer, scheduler = None, None


class YoloMS(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_targets(self, idx, imgs, targets, feat_shape, feat_dtype):
        out_targets = [imgs[0].new_zeros(shape, dtype=feat_dtype) for shape in feat_shape]
        object_target, cls_target, bbox_offset_target = out_targets
        bs, c, feat_h, feat_w = object_target.shape
        bs, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride

        ratio = torch.tensor([width_ratio, height_ratio], dtype=imgs.dtype)

        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes']
            gt_label = targets[batch_id]['labels']
            gt_centers = (gt_bbox[..., :2] + gt_bbox[..., 2:]) / 2  # 还未缩放
            gt_wh = (gt_bbox[..., 2:] - gt_bbox[..., :2])  # 还未缩放
            gt_wh = gt_wh / torch.tensor([img_w, img_h], dtype=gt_wh.dtype)  # 统一缩放到0~1

            # 根据gt的wh 选择合适的先验框（动态选择）
            for j in range(len(gt_wh)):
                gw, gh = gt_wh[j]

                if global_max:
                    # 寻找全局最佳 合适的 anchor
                    ious = wh_iou(self.anchors, torch.tensor([[gw, gh]]))[:, 0]
                    # 找到 最大 iou 对应的 stage
                    idx_ = ious.argmax()
                    a_w, a_h = self.anchors[idx_]
                    index = idx_ // n_anchors  # 对应那个 分支
                    if index != idx: continue
                    index = idx_ % n_anchors  # 某个分支对应那个anchor
                else:
                    s, e = idx * n_anchors, (idx + 1) * n_anchors
                    ious = wh_iou(self.anchors[s:e], torch.tensor([[gw, gh]]))[:, 0]
                    index = ious.argmax()
                    a_w, a_h = self.anchors[s:e][index]
                    if ious[index] < 0.3: continue

                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                if surrounding:
                    radius = gaussian_radiusv1([scale_box_h * feat_h, scale_box_w * feat_w], 0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(object_target[batch_id, index * 1], [ctx_int, cty_int], radius)
                else:
                    object_target[batch_id, index * 1, cty_int, ctx_int] = 1

                cls_target[batch_id, index * num_classes + ind, cty_int, ctx_int] = 1
                bbox_offset_target[batch_id, index * 4 + 0, cty_int, ctx_int] = ctx - ctx_int
                bbox_offset_target[batch_id, index * 4 + 1, cty_int, ctx_int] = cty - cty_int
                bbox_offset_target[batch_id, index * 4 + 2, cty_int, ctx_int] = (scale_box_w / a_w).log()
                bbox_offset_target[batch_id, index * 4 + 3, cty_int, ctx_int] = (scale_box_h / a_h).log()

        avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            object_target=object_target,
            cls_target=cls_target,
            bbox_offset_target=bbox_offset_target
        )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        losses = dict(
            loss_heatmap=0,
            loss_cls=0,
            loss_boxes=0)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)
            # feat_shape = [out.shape for out in outputs]
            feat_dtype = outputs[0].dtype
            for i, output in enumerate(outputs):
                object_pred, cls_pred, bbox_offset = output[:, 0:1 * n_anchors], \
                                                     output[:, 1 * n_anchors:(num_classes + 1) * n_anchors], \
                                                     output[:,
                                                     (1 + num_classes) * n_anchors:(5 + num_classes) * n_anchors]
                feat_shape = [object_pred.shape, cls_pred.shape, bbox_offset.shape]

                # get_target
                # target
                target_result, avg_factor = self.get_targets(i, imgs, targets, feat_shape, feat_dtype)

                object_target = target_result['object_target']
                cls_target = target_result['cls_target']
                bbox_offset_target = target_result['bbox_offset_target']
                wh_offset_target_weight = (object_target == 1).float()
                bs, _, h, w = wh_offset_target_weight.shape

                # loss
                keep = object_target != -1
                loss_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep],
                                                   object_target[keep]).sum() / avg_factor

                if not use_iouloss:
                    loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                                  torch.stack([wh_offset_target_weight] * 4, 2).contiguous().
                                  view(bs, -1, h, w)).sum() / avg_factor
                else:
                    keep = (object_target == 1).flatten(0)
                    bbox_offset = bbox_offset.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                    bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
                    bbox_offset[:, 2:] = bbox_offset[:, 2:].exp() + 2
                    bbox_offset_target = bbox_offset_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                    bbox_offset_target[:, 2:] = bbox_offset_target[:, 2:].exp() + 2
                    # loss_boxes = giou_loss(bbox_offset, bbox_offset_target, "mean")
                    loss_boxes = ciou_loss(bbox_offset, bbox_offset_target).mean()

                loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                            torch.stack([wh_offset_target_weight] * num_classes, 2).contiguous().
                            view(bs, -1, h, w)).sum() / avg_factor

                losses["loss_heatmap"] += loss_heatmap
                losses["loss_boxes"] += loss_boxes
                losses["loss_cls"] += loss_cls

        return losses

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        in_shape = imgs.shape[-2:]

        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        scale_factors = kwargs['scale_factors']
        padding = kwargs['padding']
        with_nms = kwargs['with_nms']

        result = dict(boxes=[], scores=[], labels=[])
        outputs = self.model(imgs)
        for idx, output in enumerate(outputs):
            object_pred, cls_pred, bbox_offset = output[:, 0:1 * n_anchors], \
                                                 output[:, 1 * n_anchors:(num_classes + 1) * n_anchors], \
                                                 output[:, (1 + num_classes) * n_anchors:(5 + num_classes) * n_anchors]

            object_pred = object_pred.sigmoid()
            cls_pred = cls_pred.sigmoid()

            if use_iouloss:
                bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2

            s, e = idx * n_anchors, (idx + 1) * n_anchors
            boxes, scores, labels = get_bboxes(self.anchors[s:e], object_pred, cls_pred, bbox_offset, with_nms,
                                               iou_threshold, conf_threshold, scale_factors, padding, in_shape)

            if labels.size(0) > 0:
                result["boxes"].extend(boxes)
                result["scores"].extend(scores)
                result["labels"].extend(labels)

        # nms
        if len(result["labels"]) > 0:
            result["boxes"] = torch.stack(result["boxes"], 0)
            result["scores"] = torch.stack(result["scores"], 0)
            result["labels"] = torch.stack(result["labels"], 0)
        else:
            result["boxes"] = torch.zeros([1, 4])
            result["scores"] = torch.zeros([1])
            result["labels"] = torch.zeros([1])
            return result

        if len(strides) > 1:
            keep = batched_nms(result["boxes"], result["scores"], result["labels"], iou_threshold)
            result["boxes"] = result["boxes"][keep]
            result["scores"] = result["scores"][keep]
            result["labels"] = result["labels"][keep]

        return result

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        with_nms = kwargs['with_nms']

        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        result = dict(boxes=[], scores=[], labels=[])
        outputs = self.model(imgs)
        for idx, output in enumerate(outputs):
            object_pred, cls_pred, bbox_offset = output[:, 0:1 * n_anchors], \
                                                 output[:, 1 * n_anchors:(num_classes + 1) * n_anchors], \
                                                 output[:, (1 + num_classes) * n_anchors:(5 + num_classes) * n_anchors]

            object_pred = object_pred.sigmoid()
            cls_pred = cls_pred.sigmoid()

            if use_iouloss:
                bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2

            s, e = idx * n_anchors, (idx + 1) * n_anchors
            boxes, scores, labels = get_bboxes(self.anchors[s:e], object_pred, cls_pred, bbox_offset, with_nms,
                                               iou_threshold, conf_threshold, None, None, in_shape, False)

            if labels.size(0) > 0:
                result["boxes"].extend(boxes)
                result["scores"].extend(scores)
                result["labels"].extend(labels)

        # nms
        if len(result["labels"]) > 0:
            result["boxes"] = torch.stack(result["boxes"], 0)
            result["scores"] = torch.stack(result["scores"], 0)
            result["labels"] = torch.stack(result["labels"], 0)
        else:
            result["boxes"] = torch.zeros([1, 4])
            result["scores"] = torch.zeros([1])
            result["labels"] = torch.zeros([1])
            return result

        if len(strides) > 1:
            keep = batched_nms(result["boxes"], result["scores"], result["labels"], iou_threshold)
            result["boxes"] = result["boxes"][keep]
            result["scores"] = result["scores"][keep]
            result["labels"] = result["labels"][keep]

        return result


network = YoloMS(**dict(model=model, num_classes=num_classes,
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
# network.fit(weight_path)
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path=weight_path, iou_threshold=0.3,
                       conf_threshold=0.2, with_nms=False, mode='coco'))
# -----------------------predict ---------------------------
# network.predict(**dict(img_paths=glob_format(dir_data),
#                        transform=test_transforms, device=device,
#                        weight_path=weight_path,
#                        save_path=save_path, visual=False,
#                        with_nms=False, iou_threshold=0.3,
#                        conf_threshold=0.2, method='pad'))
