"""
1、网络使用的是 yolo 网络 stride=32
2、正样本为中心点，centerness也只标记中心点（中心附近的点不标记 默认为 0）
3、预测样本与gt_box 的iou 大于0.5的负样本忽略 （减少负样本的数量）
4、使用GaussianFocalLoss or QualityFocalLoss 计算 centerness的loss
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, YOLOV1Neck, YOLOV1Head, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv2, grid_torch, xywh2x1y1x2y2
from toolcv.tools.utils import box_iou

ignore = True

seed = 100
torch.manual_seed(seed)

anchors = None
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
backbone = Backbone('resnet18', True)
neck = YOLOV1Neck(backbone.out_channels, True)
head = YOLOV1Head(backbone.out_channels // 2, 1, num_classes, True)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)


class Yolov1(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def get_targets(self, imgs, targets, feat_shape):
        bs, _, feat_h, feat_w = feat_shape
        _, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride

        ratio = torch.tensor([[width_ratio, height_ratio]], dtype=imgs.dtype)

        # 转换成最终的target
        centerness_target = imgs[0].new_zeros([bs, 1, feat_h, feat_w])
        cls_target = imgs[0].new_zeros([bs, self.num_classes, feat_h, feat_w])
        wh_target = imgs[0].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = imgs[0].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = imgs[0].new_zeros([bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes']
            gt_label = targets[batch_id]['labels']
            gt_centers = (gt_bbox[..., :2] + gt_bbox[..., 2:]) / 2 * ratio  # 缩放到feature map上
            gt_wh = (gt_bbox[..., 2:] - gt_bbox[..., :2]) * ratio  # 缩放到feature map上
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                # 只标记正样本 # yolov1 只标记中心点未正样本
                centerness_target[batch_id, 0, cty_int, ctx_int] = 1

                cls_target[batch_id, ind, cty_int, ctx_int] = 1

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, centerness_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            centerness_target=centerness_target,
            cls_target=cls_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)

        return target_result, avg_factor, ratio

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            output = self.model(imgs)
            centerness_pred = output[:, [0]]
            offset_pred = output[:, 1:3]
            wh_pred = output[:, 3:5]
            cls_pred = output[:, 5:]

        bs, _, fh, fw = output.shape
        # get_target
        # target
        target_result, avg_factor, ratio = self.get_targets(imgs, targets, centerness_pred.shape)
        centerness_target = target_result['centerness_target'].permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
        cls_target = target_result['cls_target'].permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        wh_target = target_result['wh_target'].permute(0, 2, 3, 1).contiguous().view(bs, -1, 2)
        offset_target = target_result['offset_target'].permute(0, 2, 3, 1).contiguous().view(bs, -1, 2)
        wh_offset_target_weight = target_result['wh_offset_target_weight'].permute(0, 2, 3, 1).contiguous().view(bs, -1,
                                                                                                                 2)

        centerness_pred = centerness_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        wh_pred = wh_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 2).abs()
        offset_pred = offset_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 2).abs()

        if ignore:
            # 过滤掉忽略的负样本
            xy = grid_torch(fh, fw)[None].contiguous().view(1, -1, 2)
            pred_boxes = xywh2x1y1x2y2(
                torch.cat((offset_pred + xy.to(offset_pred.device), wh_pred), -1)).detach()

            for i in range(bs):
                # 统一缩放到 heatmap尺度上
                pred_box = pred_boxes[i]
                gt_box = (targets[i]['boxes'] * torch.cat((ratio, ratio), -1)).to(self.device)
                # 计算IOU
                ious = box_iou(pred_box, gt_box)
                score, _ = ious.max(1)
                index = torch.arange(0, score.size(0))[score > 0.5]
                if len(index) > 0:
                    for idx in index:
                        if centerness_target[i][idx].sum() == 0: centerness_target[i][idx] = -1  # 忽略

        # loss
        keep = centerness_target != -1
        loss_center_heatmap = GaussianFocalLoss()(centerness_pred.sigmoid()[keep], centerness_target[keep],
                                                  avg_factor=avg_factor)
        # loss_center_heatmap = FocalLoss()(centerness_pred[keep].view(-1, 1),  # 会自动添加 sigmoid
        #                                   centerness_target[keep].view(-1, 1).argmax(1),
        #                                   avg_factor=avg_factor)

        loss_wh = L1Loss(loss_weight=0.8)(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor)

        loss_offset = L1Loss()(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor)

        loss_cls = CrossEntropyLoss(True)(cls_pred, cls_target, wh_offset_target_weight[..., [0]].expand_as(cls_target),
                                          avg_factor=avg_factor)  # 会自动添加 sigmoid

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    @torch.no_grad()
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        output = self.model(img)
        centerness_preds = output[:, [0]]
        offset_preds = output[:, 1:3]
        wh_preds = output[:, 3:5]
        cls_preds = output[:, 5:]
        centerness_preds = centerness_preds.sigmoid()
        cls_preds = cls_preds.sigmoid()
        wh_preds, offset_preds = wh_preds.abs(), offset_preds.abs()

        boxes, scores, labels = get_bboxesv2(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                             iou_threshold, conf_threshold, scale_factors, padding, in_shape)

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        output = self.model(imgs)
        centerness_preds = output[:, [0]]
        offset_preds = output[:, 1:3]
        wh_preds = output[:, 3:5]
        cls_preds = output[:, 5:]
        centerness_preds = centerness_preds.sigmoid()
        cls_preds = cls_preds.sigmoid()
        wh_preds, offset_preds = wh_preds.abs(), offset_preds.abs()

        boxes, scores, labels = get_bboxesv2(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                             iou_threshold, conf_threshold, scale_factors, padding, in_shape, False)

        return boxes, scores, labels


network = Yolov1(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                 gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                 gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                 train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
# network.evalute(mode='mmdet', conf_threshold=0.1)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.1, with_nms=True, method='pad')
