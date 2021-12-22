"""
将原本的 heatmap 分支 拆成 centerness 分支 + cls 分支
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.nn import functional as F

from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from mmdet.models.losses import GaussianFocalLoss, L1Loss, CrossEntropyLoss
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, CenterNetNeck, _initParmas, CBA  # ,CenterNetHead
from toolcv.api.define.utils.model.head import CenterNetHeadV2
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.data.transform import gaussian_radius, draw_umich_gaussian
# from toolcv.api.define.utils.model.mmdet import get_bboxes
from toolcv.api.define.utils.tools.tools import get_bboxesv2

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
h, w = 512, 512
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

backbone = Backbone('resnet18', True)
neck = CenterNetNeck(backbone.out_channels, backbone.out_channels // 2, False)
head = CenterNetHeadV2(backbone.out_channels // 8, num_classes)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)


class CenterNet(_BaseNetV2):
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

                """
                radius = gaussian_radius((scale_box_h, scale_box_w), 0.7)
                radius = max(0, int(radius))
                htmap = centerness_target[batch_id, 0].cpu().numpy()
                htmap = draw_umich_gaussian(htmap, (ctx_int, cty_int), radius)
                centerness_target[batch_id, 0] = torch.from_numpy(htmap).to(self.device)
                """
                radius = gaussian_radiusv1([scale_box_h, scale_box_w], 0.3)
                radius = max(0, int(radius))
                gen_gaussian_target(centerness_target[batch_id, 0], [ctx_int, cty_int], radius)
                # """
                # 只标记正样本
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

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            centerness_pred, cls_pred, wh_pred, offset_pred = self.model(imgs)

        # get_target
        # target
        target_result, avg_factor = self.get_targets(imgs, targets, centerness_pred.shape)
        centerness_target = target_result['centerness_target']
        cls_target = target_result['cls_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # loss
        loss_center_heatmap = GaussianFocalLoss()(centerness_pred.sigmoid(), centerness_target,
                                                  avg_factor=avg_factor)

        keep = wh_offset_target_weight[:, [0]]  # .expand_as(cls_target)

        loss_cls = F.binary_cross_entropy(cls_pred.sigmoid() * keep, cls_target * keep, reduction='none').sum() / (
                avg_factor * 2)

        loss_wh = L1Loss(loss_weight=0.1)(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = L1Loss()(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    @torch.no_grad()
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        centerness_preds, cls_preds, wh_preds, offset_preds = self.model(img)
        centerness_preds = centerness_preds.sigmoid()
        cls_preds = cls_preds.sigmoid()

        boxes, scores, labels = get_bboxesv2(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                             iou_threshold, conf_threshold, scale_factors, padding, in_shape)

        return boxes, scores, labels

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        centerness_preds, cls_preds, wh_preds, offset_preds = self.model(imgs)
        centerness_preds = centerness_preds.sigmoid()
        cls_preds = cls_preds.sigmoid()

        boxes, scores, labels = get_bboxesv2(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                             iou_threshold, conf_threshold, scale_factors, padding, in_shape, False)

        return boxes, scores, labels


network = CenterNet(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
                    gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                    gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                    train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(mode='mmdet', conf_threshold=0.1)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.1, with_nms=True, method=method)
