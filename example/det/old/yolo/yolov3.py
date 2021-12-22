"""
使用centernet方式 同时考虑中心附近的点

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.785
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.870
--------------------------------------------------------------------------------
epochs=50;yolov4
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.736
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.962
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.789
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

from toolcv.utils.net.net import SPPv2, _initParmasV2
from toolcv.utils.net.base import _BaseNetV2
# from toolcv.api.define.utils.model.net import Backbone, YOLOV3Neck, YOLOV3Head, _initParmas, FPN
# from toolcv.tools.segment.net.YOLOP.net import YOLOPV2
# from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
# from toolcv.api.define.utils.data import augment as aug
# from toolcv.data.dataset import glob_format
from toolcv.utils.net.backbone import BackboneCspnet, BackboneResnet
from toolcv.utils.net.neck import FPN, PAN
from toolcv.utils.net.head import YOLOV3Head

from toolcv.utils.data.data import DataDefine, FruitsNutsDataset
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format  # , get_bboxes_DA as get_bboxes, xywh2x1y1x2y2
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

from toolcv.api.define.utils.tools.tools import get_bboxes_yolov3 as get_bboxes  # , grid_torch
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

from toolcv.tools.anchor import kmean_gen_anchorv2

ignore = True
surrounding = True

seed = 100
set_seed(seed)

# anchors = [[[0.1, 0.1], [0.15, 0.15], [0.2, 0.2]],  # s =8
#            [[0.25, 0.25], [0.3, 0.3], [0.35, 0.35]],  # s =16
#            [[0.4, 0.4], [0.45, 0.45], [0.5, 0.5]]]  # s =32 # 缩放到0~1 [w,h]

# anchors = [[[0.09713541, 0.11486112], [0.10620192, 0.1524359], [0.1355357, 0.2329762]],
#            [[0.15125, 0.16857141], [0.19992188, 0.1828125], [0.15840909, 0.2815909]],
#            [[0.19041668, 0.23933333], [0.19107142, 0.33357143], [0.254375, 0.2255]]]
anchors = [[0.0971, 0.1149],
           [0.1062, 0.1524],
           [0.1513, 0.1686],
           [0.1355, 0.2330],
           [0.1999, 0.1828],
           [0.1584, 0.2816],
           [0.1904, 0.2393],
           [0.2544, 0.2255],
           [0.1911, 0.3336]]
anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)

strides = 4
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
save_path = "./output"

# 通过聚类生成先验框
# kmean_gen_anchorv2(FruitsNutsDataset(dir_data,classes),9)
# exit(0)
# wh [[0.254375   0.2255    ]
#  [0.10620192 0.1524359 ]
#  [0.1355357  0.2329762 ]
#  [0.19992188 0.1828125 ]
#  [0.19107142 0.33357143]
#  [0.15125    0.16857141]
#  [0.19041668 0.23933333]
#  [0.15840909 0.2815909 ]
#  [0.09713541 0.11486112]]

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
data.get_dataloader()
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# ----------------model --------------------------
# backbone = Backbone('resnet18', True, num_out=3)
# neck = YOLOV3Neck(backbone.out_channels, True)
# head = YOLOV3Head(backbone.out_channels // 2, 3, num_classes, True)
# _initParmas(neck.modules(), mode='kaiming')
# _initParmas(head.modules(), mode='normal')
# model = nn.Sequential(backbone, neck, head).to(device)
# load_weight(model, weight_path, "", device=device)

# --------------yolov4----------------------------------
backbone = BackboneCspnet("cspdarknet53", True, num_out=3)
# backbone = BackboneResnet("resnet18", False, num_out=3)
out_channels = backbone.out_channels
activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
neck = nn.Sequential(
    SPPv2(out_channels, out_channels // 2),
    FPN([out_channels // 4, out_channels // 2, out_channels], [256, 256, 256], activate, "cat"),
    PAN([256, 256, 256], [256, 256, 256], activate, 'cat')
)
head = YOLOV3Head([256, 256, 256], 3, num_classes, activate, True)
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


class Yolov3(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None, each_batch_scheduler=False):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader, each_batch_scheduler)

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
                n_layer = index // 3
                n_anchor = index % 3
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
                        n_layer = idx // 3
                        n_anchor = idx % 3
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
            outputs = self.model(imgs)

            feat_shape = [out.shape for out in outputs]
            feat_dtype = outputs[0].dtype
            # get_target
            # target
            out_targets, avg_factor = self.get_targets(imgs, targets, feat_shape, feat_dtype)

            c = feat_shape[0][-1]
            outputs = torch.cat([output.contiguous().view(-1, c) for output in outputs], 0)

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
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        outputs = self.model(img)
        anchors = self.anchors.view(3, 3, -1).to(self.device)
        boxes_list, scores_list, labels_list = [], [], []
        for j, output in enumerate(outputs):
            fh, fw = output.shape[1:3]
            centerness_preds = output[..., [0]]
            offset_preds = output[..., 1:3]
            wh_preds = output[..., 3:5]
            cls_preds = output[..., 5:]
            centerness_preds = centerness_preds.sigmoid()
            cls_preds = cls_preds.sigmoid()
            anchor = anchors[j].expand_as(wh_preds)
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
        keep = batched_nms(boxes_list, scores_list, labels_list, iou_threshold)
        boxes, scores, labels = boxes_list[keep], scores_list[keep], labels_list[keep]

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        outputs = self.model(imgs)
        anchors = self.anchors.view(3, 3, -1).to(self.device)
        boxes_list, scores_list, labels_list = [], [], []
        for j, output in enumerate(outputs):
            fh, fw = output.shape[1:3]
            centerness_preds = output[..., [0]]
            offset_preds = output[..., 1:3]
            wh_preds = output[..., 3:5]
            cls_preds = output[..., 5:]
            centerness_preds = centerness_preds.sigmoid()
            cls_preds = cls_preds.sigmoid()
            anchor = anchors[j].expand_as(wh_preds)
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
        keep = batched_nms(boxes_list, scores_list, labels_list, iou_threshold)
        boxes, scores, labels = boxes_list[keep], scores_list[keep], labels_list[keep]

        return boxes, scores, labels


network = Yolov3(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
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
