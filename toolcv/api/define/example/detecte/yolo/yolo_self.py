"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms

from torch.nn import functional as F
from toolcv.utils.loss.loss import gaussian_focal_loss
from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

# ignore = True
surrounding = True

seed = 100
torch.manual_seed(seed)

anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
anchors = torch.tensor(anchors, dtype=torch.float32)

strides = 4
use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
h, w = 416, 416
resize = (h, w)
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
        # aug.Resize(*resize),
        aug.ResizeMax(*resize), aug.Padding(),
        aug.ToTensor(), aug.Normalize()])
    transforms_mosaic = None

test_transforms = aug.Compose([
    # aug.Resize(*resize),
    aug.ResizeMax(*resize), aug.Padding(),
    aug.ToTensor(), aug.Normalize()])

dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, use_mosaic=use_mosaic,
                            transforms_mosaic=transforms_mosaic, h=h, w=w)
val_dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, use_mosaic=False,
                                transforms_mosaic=None, h=h, w=w)
# dataset.show(mode='cv2')
dataloader = LoadDataloader(dataset, val_dataset, 0.0, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
backbone = Backbone('resnet18', True, stride=16)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelf(backbone.out_channels // 4, len(anchors), 1, num_classes)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)


class YoloSelf(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def get_targets(self, imgs, targets, feat_shape):
        out_targets = [imgs[0].new_zeros(shape) for shape in feat_shape]
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
                iw = (self.anchors - gw).abs().argmin()
                ih = (self.anchors - gh).abs().argmin()

                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                # 对应的先验anchor
                a_w, a_h = self.anchors[iw], self.anchors[ih]

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
        object_pred, cls_pred, bbox_offset, wh_cls = outputs
        bs, c, feat_h, feat_w = wh_cls.shape
        wh_cls = wh_cls.contiguous().view(bs, c // 2, 2, feat_h, feat_w)

        # get_target
        # target
        target_result, avg_factor = self.get_targets(imgs, targets, feat_shape)

        object_target = target_result['object_target']
        cls_target = target_result['cls_target']
        bbox_offset_target = target_result['bbox_offset_target']
        wh_cls_target = target_result['wh_cls_target']
        wh_cls_target = wh_cls_target.argmax(1)
        wh_offset_target_weight = (object_target == 1).float()

        # loss
        keep = object_target != -1
        """
        loss_center_heatmap = GaussianFocalLoss()(object_pred.sigmoid()[keep], object_target[keep],
                                                  avg_factor=avg_factor)
        loss_boxes = L1Loss()(
            bbox_offset,
            bbox_offset_target,
            wh_offset_target_weight.expand_as(bbox_offset_target),
            avg_factor=avg_factor)

        loss_cls = CrossEntropyLoss(True)(cls_pred, cls_target, wh_offset_target_weight.expand_as(cls_target),
                                          avg_factor=avg_factor)  # 会自动添加 sigmoid

        loss_wh = CrossEntropyLoss()(wh_cls, wh_cls_target, wh_offset_target_weight.expand_as(wh_cls_target),
                                     avg_factor=avg_factor)  # 自动加上 softmax
        """
        loss_center_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep], object_target[keep]).sum() / avg_factor

        loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                      wh_offset_target_weight.expand_as(bbox_offset_target)).sum() / avg_factor

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
        boxes, scores, labels = get_bboxesv3(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
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
        boxes, scores, labels = get_bboxesv3(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                                             iou_threshold, conf_threshold, scale_factors, padding, in_shape, False)

        return boxes, scores, labels


network = YoloSelf(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                   gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                   gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                   train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(mode='coco', conf_threshold=0.2)
# -----------------------predict ---------------------------
# network.predict(glob_format(dir_data), test_transforms, device, visual=False,
#                 conf_threshold=0.1, with_nms=True, method='pad')
