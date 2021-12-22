"""
backbone + neck + head
neck 包含 spp fpn pan
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof

yolov1+centernet (anchor free) 效果 不如 基于anchor的收敛快
----------------------------------------------------------------------------------
epochs = 150;stride=16
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.931
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.696
----------------------------------------------------------------------------------
epochs = 200;stride=16
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.727
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.983
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
----------------------------------------------------------------------------------
epochs = 150;stride=16;freeze_at=4
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.798
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.909
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
epochs = 150;stride=8
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.945
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422
----------------------------------------------------------------------------------
epochs = 150;stride=8;freeze_at=4
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.718
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.930
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, giou_loss, ciou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHead, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.utils.data.data import DataDefine
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, get_bboxes
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

# from toolcv.api.define.utils.tools.tools import get_bboxesv3
# from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2
from toolcv.tools.anchor import gen_anchorv2  # ,kmean_gen_anchorv2,kmean_gen_anchor,gen_anchor

# from toolcv.api.define.utils.data.data import FruitsNutsDataset

# ignore = True
surrounding = True

seed = 100
set_seed(seed)

anchors = None
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
save_path = 'output'
summary = SummaryWriter(save_path)

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
head = YOLOFHead(backbone.out_channels // 4, 1, num_classes)
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


class YoloV1(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_targets(self, imgs, targets, feat_shape, feat_dtype):
        out_targets = [imgs[0].new_zeros(shape, dtype=feat_dtype) for shape in feat_shape]
        cls_target, bbox_offset_target, object_target = out_targets

        bs, _, img_h, img_w = imgs.shape
        _, _, feat_h, feat_w = cls_target.shape

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
                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                if surrounding:
                    radius = gaussian_radiusv1([scale_box_h * feat_h, scale_box_w * feat_w], 0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(object_target[batch_id, 0], [ctx_int, cty_int], radius)
                else:
                    object_target[batch_id, 0, cty_int, ctx_int] = 1

                cls_target[batch_id, ind, cty_int, ctx_int] = 1
                bbox_offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                bbox_offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                bbox_offset_target[batch_id, 2, cty_int, ctx_int] = scale_box_w * feat_w
                bbox_offset_target[batch_id, 3, cty_int, ctx_int] = scale_box_h * feat_h

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

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)
            feat_shape = [out.shape for out in outputs]
            feat_dtype = outputs[0].dtype
            cls_pred, bbox_offset, object_pred = outputs

            # get_target
            # target
            target_result, avg_factor = self.get_targets(imgs, targets, feat_shape, feat_dtype)

            object_target = target_result['object_target']
            cls_target = target_result['cls_target']
            bbox_offset_target = target_result['bbox_offset_target']
            wh_offset_target_weight = (object_target == 1).float()

            # loss
            keep = object_target != -1
            loss_center_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep],
                                                      object_target[keep]).sum() / avg_factor

            if not use_iouloss:
                # loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                #               wh_offset_target_weight.expand_as(bbox_offset_target)).sum() / avg_factor

                loss_boxes = ((F.l1_loss(bbox_offset[:, :2].sigmoid() * 2, bbox_offset_target[:, :2],
                                         reduction="none") + \
                               F.l1_loss(bbox_offset[:, 2:].exp(), bbox_offset_target[:, 2:], reduction="none")) * \
                              wh_offset_target_weight.expand_as(bbox_offset_target[:, :2])).sum() / avg_factor
            else:
                keep = (object_target == 1).flatten(0)
                bbox_offset = bbox_offset.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
                bbox_offset[:, 2:] = bbox_offset[:, 2:].exp()
                bbox_offset_target = bbox_offset_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                # loss_boxes = giou_loss(bbox_offset, bbox_offset_target, "mean")
                loss_boxes = ciou_loss(bbox_offset, bbox_offset_target).mean()

            loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                        wh_offset_target_weight.expand_as(cls_target)).sum() / avg_factor

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_boxes=loss_boxes)

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        in_shape = imgs.shape[-2:]

        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        scale_factors = kwargs['scale_factors']
        padding = kwargs['padding']
        with_nms = kwargs['with_nms']

        cls_pred, bbox_offset, object_pred = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        offset_preds = bbox_offset[:, :2].sigmoid() * 2
        wh_preds = bbox_offset[:, 2:].exp()

        boxes, scores, labels = get_bboxes(object_pred, cls_pred, wh_preds, offset_preds, with_nms,
                                           iou_threshold, conf_threshold, scale_factors, padding, in_shape,
                                           True, 100, "grid", "stride")

        return dict(boxes=boxes, scores=scores, labels=labels)

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        with_nms = kwargs['with_nms']

        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        cls_pred, bbox_offset, object_pred = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        offset_preds = bbox_offset[:, :2].sigmoid() * 2
        wh_preds = bbox_offset[:, 2:].exp()

        boxes, scores, labels = get_bboxes(object_pred, cls_pred, wh_preds, offset_preds, with_nms,
                                           iou_threshold, conf_threshold, None, None, in_shape, False,
                                           100, "grid", "stride")

        return dict(boxes=boxes, scores=scores, labels=labels)


network = YoloV1(**dict(model=model, num_classes=num_classes,
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
network.fit(weight_path)
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
