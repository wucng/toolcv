"""
类似 centernet—ms
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.ops import batched_nms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, CyclicLR
from fvcore.nn import giou_loss

from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from mmdet.models.losses import GaussianFocalLoss, L1Loss, CrossEntropyLoss, GIoULoss
# from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
#                                                 transpose_and_gather_feat)

from toolcv.api.define.utils.model.base import _BaseNetV2, get_params
from toolcv.api.define.utils.model.net import Backbone, _initParmas, FPN, FCOSHead, CBA
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.data.transform import gaussian_radius, draw_umich_gaussian
# from toolcv.api.define.utils.model.mmdet import get_bboxes
from toolcv.api.define.utils.tools.tools import get_bboxesv2

only_center = True  # 正样本时只使用中心点

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

backbone = Backbone('dla34', True, num_out=3, freeze_at=5)
# neck = FPN(backbone.out_channels, 256, Conv2d=nn.Conv2d)
neck = FPN(backbone.out_channels, 256, Conv2d=CBA)
head = FCOSHead(256, num_classes, use_gn=True)  # head 一般不使用BN 但可以使用GN
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)

INF = 1e8
# regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
# regress_ranges = ((-1, 32), (32, 64), (64, 128), (128, 256), (256, INF))
regress_ranges = ((-1, 32), (32, 96), (96, 224), (224, 480), (480, INF))


class Fcos(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def configure_optimizers(self):
        params = get_params(self.model.modules(), self.lr, self.weight_decay, self.gamma)
        self.optimizer = torch.optim.SGD(params, self.lr, momentum=0.9, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, epochs, 2, self.lr * self.lrf)
        # self.scheduler = CosineAnnealingLR(self.optimizer, epochs, self.lr * self.lrf)

    def get_targets(self, imgs, targets, feat_shape):
        bs, _, img_h, img_w = imgs.shape

        ratio_list = []
        target_list = []
        for feat_h, feat_w in feat_shape:
            width_ratio = float(feat_w / img_w)  # 1/stride
            height_ratio = float(feat_h / img_h)  # 1/stride
            ratio = torch.tensor([[width_ratio, height_ratio]], dtype=imgs.dtype)
            ratio_list.append(ratio)

            # 转换成最终的target
            cls_target = imgs[0].new_zeros([bs, self.num_classes, feat_h, feat_w])
            box_reg_target = imgs[0].new_zeros([bs, 4, feat_h, feat_w])
            centerness_target = imgs[0].new_zeros([bs, 1, feat_h, feat_w])
            target_list.append([cls_target, box_reg_target, centerness_target])

        # 分配正负样本
        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes']
            gt_label = targets[batch_id]['labels']

            for i, (x1, y1, x2, y2) in enumerate(gt_bbox):
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                label = gt_label[i].item()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                gt_w = x2 - x1
                gt_h = y2 - y1
                for j, ratio in enumerate(ratio_list):
                    [[width_ratio, height_ratio]] = ratio
                    # 恢复到原图上
                    stride = 1 / width_ratio.item()

                    # center
                    _cx = cx * width_ratio.item()
                    _cy = cy * height_ratio.item()
                    _cx_int = int(_cx)
                    _cy_int = int(_cy)
                    dcx = _cx - _cx_int
                    dcy = _cy - _cy_int

                    target_list[j][2][batch_id, :, _cy_int, _cx_int] = 1
                    target_list[j][0][batch_id, label, _cy_int, _cx_int] = 1
                    target_list[j][1][batch_id, :, _cy_int, _cx_int] = torch.tensor([dcx, dcy,
                                                                                     gt_w * width_ratio,
                                                                                     gt_h * height_ratio],
                                                                                    device=self.device)

                    # target_list[j][1][batch_id, :, _cy_int, _cx_int] = torch.tensor([x1 * width_ratio,
                    #                                                                  y1 * height_ratio,
                    #                                                                  x2 * width_ratio,
                    #                                                                  y2 * height_ratio],
                    #                                                                 device=self.device)

                    # 从输入图像映射到heatmap上
                    _x1 = (x1 * width_ratio).floor().int().item()
                    _x2 = (x2 * width_ratio).ceil().int().item()
                    _y1 = (y1 * height_ratio).floor().int().item()
                    _y2 = (y2 * height_ratio).ceil().int().item()

                    for _y in range(_y1, _y2 + 1):
                        y = int(stride / 2) + stride * _y  # 从heatmap 映射到 输入图像上
                        t = y - y1
                        b = y2 - y
                        if min(t, b) < 0: continue  # 落在原始框外面
                        max_val = max(t, b)
                        if max_val < regress_ranges[j][0] or max_val > regress_ranges[j][1]: continue

                        for _x in range(_x1, _x2 + 1):
                            x = int(stride / 2) + stride * _x
                            l = x - x1
                            r = x2 - x
                            # if min(l, r, t, b) < 0: continue  # 落在原始框外面
                            if min(l, r) < 0: continue  # 落在原始框外面
                            max_val = max(l, r, t, b)
                            if max_val < regress_ranges[j][0] or max_val > regress_ranges[j][1]: continue

                            # centerness = math.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                            centerness = min(l, r) / max(l, r) * min(t, b) / max(t, b)
                            centerness *= 0.6
                            if target_list[j][2][batch_id, :, _y, _x] >= centerness: continue
                            target_list[j][2][batch_id, :, _y, _x] = centerness

                            if not only_center:
                                dcx = _cx - _x
                                dcy = _cy - _y
                                target_list[j][0][batch_id, label, _y, _x] = 1
                                target_list[j][1][batch_id, :, _y, _x] = torch.tensor([dcx, dcy,
                                                                                       gt_w * width_ratio,
                                                                                       gt_h * height_ratio],
                                                                                      device=self.device)

        return target_list

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_preds, reg_preds, centerness_preds = self.model(imgs)

        # get_target
        # target
        feat_shape = [centerness_pred.shape[-2:] for centerness_pred in centerness_preds]
        target_result = self.get_targets(imgs, targets, feat_shape)

        # loss
        loss_center_heatmap = Variable(torch.zeros(1, device=self.device), requires_grad=True)
        loss_cls = Variable(torch.zeros(1, device=self.device), requires_grad=True)
        loss_reg = Variable(torch.zeros(1, device=self.device), requires_grad=True)

        for i in range(len(feat_shape)):
            cls_target, reg_target, centerness_target = target_result[i]
            avg_factor = max(1, (centerness_target == 1).sum())
            # GaussianFocalLoss 不会自动添加 sigmoid，需要手动添加
            loss_center_heatmap = loss_center_heatmap + GaussianFocalLoss()(centerness_preds[i].sigmoid(),
                                                                            centerness_target,
                                                                            avg_factor=avg_factor)

            if only_center:
                weight = (centerness_target == 1).float().expand_as(cls_target)
                if weight.sum() > 0:
                    # CrossEntropyLoss(True) 会自动添加 sigmoid，不需要手动添加
                    loss_cls = loss_cls + CrossEntropyLoss(True)(cls_preds[i], cls_target,  # .sigmoid()
                                                                 weight,
                                                                 avg_factor=avg_factor)

                    weight = (centerness_target == 1).float().expand_as(reg_target)
                    # loss_reg = loss_reg + GIoULoss()(reg_preds[i].exp().permute(0, 2, 3, 1).contiguous().view(-1, 4),
                    #                                  reg_target.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                    #                                  weight.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                    #                                  avg_factor=avg_factor)

                    loss_reg = loss_reg + L1Loss()(reg_preds[i], reg_target, weight, avg_factor=avg_factor)
            else:
                weight = (centerness_target > 0.1).float().expand_as(cls_target)
                if weight.sum() > 0:
                    # CrossEntropyLoss(True) 会自动添加 sigmoid，不需要手动添加
                    loss_cls = loss_cls + CrossEntropyLoss(True)(cls_preds[i], cls_target,  # .sigmoid()
                                                                 weight,
                                                                 avg_factor=avg_factor)

                    weight = (centerness_target > 0.1).float().expand_as(reg_target)
                    loss_reg = loss_reg + L1Loss()(reg_preds[i], reg_target, weight, avg_factor=avg_factor)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_box_reg=loss_reg)

    @torch.no_grad()
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        cls_preds, reg_preds, centerness_preds = self.model(img)
        boxes_list, scores_list, labels_list = [], [], []
        for i in range(len(centerness_preds)):
            centerness_pred = centerness_preds[i].sigmoid()
            cls_pred = cls_preds[i].sigmoid()
            reg_pred = reg_preds[i]  # .exp()
            wh_pred, offset_pred = reg_pred[:, 2:], reg_pred[:, :2]
            boxes, scores, labels = get_bboxesv2(centerness_pred, cls_pred, wh_pred, offset_pred, with_nms,
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

        cls_preds, reg_preds, centerness_preds = self.model(imgs)
        boxes_list, scores_list, labels_list = [], [], []
        for i in range(len(centerness_preds)):
            centerness_pred = centerness_preds[i].sigmoid()
            cls_pred = cls_preds[i].sigmoid()
            reg_pred = reg_preds[i]  # .exp()
            wh_pred, offset_pred = reg_pred[:, 2:], reg_pred[:, :2]
            boxes, scores, labels = get_bboxesv2(centerness_pred, cls_pred, wh_pred, offset_pred, with_nms,
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


network = Fcos(model, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters=1000,
               gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
               gradient_clip_val=gradient_clip_val, device=device, criterion=None,
               train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()
# exit(-1)
# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
# network.evalute(mode='mmdet', conf_threshold=0.1)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.1, with_nms=True, method=method)
