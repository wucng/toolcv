"""
使用 giou loss
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点

-------------------------------------------------------------------------------
use_iouloss = False

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.848
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.981

-------------------------------------------------------------------------------
use_iouloss = True
each_batch_scheduler = True

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.735
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.991
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.915

--------------------------------------------------------------------------------
use_iouloss = True
each_batch_scheduler = False

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.992
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

from toolcv.utils.net.backbone import BackboneEfficientnet, BackboneResnest, BackboneResnet, BackboneCspnet
from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.utils.data.data import DataDefine
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, get_bboxes_DA as get_bboxes, xywh2x1y1x2y2
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

# from toolcv.api.define.utils.tools.tools import get_bboxesv3
# from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2
from toolcv.tools.anchor import gen_anchorv2  # ,kmean_gen_anchorv2,kmean_gen_anchor,gen_anchor
from toolcv.api.define.utils.data.data import FruitsNutsDataset

# ignore = True
surrounding = True

seed = 100
set_seed(seed)

# anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
# anchors = torch.tensor(anchors, dtype=torch.float32)
# anchors = torch.stack((anchors, anchors), -1)

# 通过聚类生成先验框
wsize = [0.09784091, 0.13189815, 0.16025642, 0.1969853, 0.2532143]
hsize = [0.12607142, 0.17412879, 0.22333333, 0.27459598, 0.3354762]
anchors_w = torch.tensor(wsize, dtype=torch.float32)
anchors_h = torch.tensor(hsize, dtype=torch.float32)
anchors = torch.stack((anchors_w, anchors_h), -1)  # 5x2

n_anchors = len(anchors)
strides = 16
use_amp = True  # 推荐 先 True 再 False
accumulate = 2  # 推荐 先 >1 再 1
gradient_clip_val = 0.0 if use_amp else 1.0
lrf = 0.1
lr = 5e-4 if use_amp else 3e-3
weight_decay = 5e-6
gamma = 0.8
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
# gen_anchorv2(FruitsNutsDataset(dir_data,classes),5)
# exit(0)
# w [0.09784091 0.13189815 0.16025642 0.1969853  0.2532143 ]
# h [0.12607142 0.17412879 0.22333333 0.27459598 0.3354762 ]

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
data.get_dataloader()
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# ----------------model --------------------------
backbone = Backbone('resnet18', True, 2, stride=strides)
# backbone = BackboneEfficientnet("efficientnet_b0",False,stride=strides,dilation=True)
# backbone.out_channels = 320
# backbone = BackboneResnet("seresnext26t_32x4d",True,stride=strides)
# backbone = BackboneCspnet("cspdarknet53",True,stride=strides,dilation=True)

# backbone = BackboneResnest("resnest26d",True,stride=strides)
spp = SPPv2(backbone.out_channels, backbone.out_channels // 2)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelf(backbone.out_channels // 4, n_anchors, 1, num_classes)
_initParmas(spp.modules(), mode='kaiming')
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, spp, neck, head).to(device)
load_weight(model, weight_path, "", device=device)
model.train()

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, gamma)
else:
    optimizer, scheduler = None, None


class YoloSelf(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None, each_batch_scheduler=False):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader, each_batch_scheduler)

    def get_targets(self, imgs, targets, feat_shape, feat_dtype):
        out_targets = [imgs[0].new_zeros(shape, dtype=feat_dtype) for shape in feat_shape]
        object_target, cls_target, bbox_offset_target, wh_cls_target = out_targets
        bs, c, feat_h, feat_w = wh_cls_target.shape
        wh_cls_target = wh_cls_target.contiguous().view(bs, c // 2, 2, feat_h, feat_w)

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
                iw = (self.anchors[:, 0] - gw).abs().argmin()
                ih = (self.anchors[:, 1] - gh).abs().argmin()

                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                # 对应的先验anchor
                a_w, a_h = self.anchors[:, 0][iw], self.anchors[:, 1][ih]

                if surrounding:
                    radius = gaussian_radiusv1([scale_box_h * feat_h, scale_box_w * feat_w], 0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(object_target[batch_id, 0], [ctx_int, cty_int], radius)
                else:
                    object_target[batch_id, 0, cty_int, ctx_int] = 1

                cls_target[batch_id, ind, cty_int, ctx_int] = 1
                bbox_offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                bbox_offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                bbox_offset_target[batch_id, 2, cty_int, ctx_int] = (scale_box_w / a_w).log()
                bbox_offset_target[batch_id, 3, cty_int, ctx_int] = (scale_box_h / a_h).log()

                wh_cls_target[batch_id, iw, 0, cty_int, ctx_int] = 1
                wh_cls_target[batch_id, ih, 1, cty_int, ctx_int] = 1

        avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            object_target=object_target,
            cls_target=cls_target,
            bbox_offset_target=bbox_offset_target,
            wh_cls_target=wh_cls_target
        )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)
            feat_shape = [out.shape for out in outputs]
            feat_dtype = outputs[0].dtype
            object_pred, cls_pred, bbox_offset, wh_cls = outputs
            bs, c, feat_h, feat_w = wh_cls.shape
            wh_cls = wh_cls.contiguous().view(bs, c // 2, 2, feat_h, feat_w)

            # get_target
            # target
            target_result, avg_factor = self.get_targets(imgs, targets, feat_shape, feat_dtype)

            object_target = target_result['object_target']
            cls_target = target_result['cls_target']
            bbox_offset_target = target_result['bbox_offset_target']
            wh_cls_target = target_result['wh_cls_target']
            wh_cls_target = wh_cls_target.argmax(1)
            wh_offset_target_weight = (object_target == 1).float()

            # loss
            keep = object_target != -1
            loss_center_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep],
                                                      object_target[keep]).sum() / avg_factor
            # 使用sigmoid_focal_loss 效果差
            # loss_center_heatmap = sigmoid_focal_loss(object_pred[keep],object_target[keep],0.45,4,"none").sum()/avg_factor

            if not use_iouloss:
                loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                              wh_offset_target_weight.expand_as(bbox_offset_target)).sum() / avg_factor
            else:
                # giou loss
                keep = (object_target == 1).flatten(0)
                bbox_offset = bbox_offset.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
                bbox_offset[:, 2:] = bbox_offset[:, 2:].exp() + 2
                bbox_offset_target = bbox_offset_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                bbox_offset_target[:, 2:] = bbox_offset_target[:, 2:].exp() + 2
                # loss_boxes = giou_loss(bbox_offset, bbox_offset_target, "mean")
                loss_boxes = ciou_loss(bbox_offset, bbox_offset_target).mean()

            loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                        wh_offset_target_weight.expand_as(cls_target)).sum() / avg_factor

            loss_wh = (F.cross_entropy(wh_cls, wh_cls_target, reduction="none") *
                       wh_offset_target_weight.expand_as(wh_cls_target)).sum() / avg_factor

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_boxes=loss_boxes)

    @torch.no_grad()
    def pred_step(self, imgs, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        # outputs = self.model(imgs)
        object_pred, cls_pred, bbox_offset, wh_cls = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        if use_iouloss: bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
        boxes, scores, labels = get_bboxes(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                                           iou_threshold, conf_threshold, scale_factors, padding, in_shape)

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        object_pred, cls_pred, bbox_offset, wh_cls = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        if use_iouloss: bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
        boxes, scores, labels = get_bboxes(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                                           iou_threshold, conf_threshold, scale_factors, padding, in_shape, False)

        return boxes, scores, labels


network = YoloSelf(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                   gamma=gamma, optimizer=optimizer, scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                   gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                   train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   each_batch_scheduler=each_batch_scheduler)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit(weight_path)
# -----------------------eval ---------------------------
network.evalute(mode='coco', conf_threshold=0.2)
# # -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.2, with_nms=True, method='pad')
