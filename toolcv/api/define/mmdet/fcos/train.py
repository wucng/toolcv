"""
https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py
https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth

效果很差？？？
"""

import torch
from torch import nn
from torch.nn import functional as Func
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import math

from fvcore.nn import sigmoid_focal_loss

from mmdet.core import multi_apply
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet.models.losses import FocalLoss, SmoothL1Loss, CIoULoss, CrossEntropyLoss, L1Loss, GaussianFocalLoss
from toolcv.api.define.utils.model.mmdet import _BaseNet, _initParmas, load_model, batched_nms
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.data.dataset import glob_format
from toolcv.api.define.utils.model.net import CBA
from toolcv.tools.utils import xywh2x1y1x2y2

INF = 1e8
regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))


class Fcos(_BaseNet):
    def __init__(self, model, config, checkpoint, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):

        super().__init__(num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters, gamma,
                         optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

        if model is None:
            self.model = self.load_model(config, checkpoint, num_classes, device)
        else:
            self.model = model

    def load_model(self, config, checkpoint, num_classes, device):
        if num_classes == 80:
            model = load_model(config, checkpoint, num_classes, device)  # 类别对不上会报错
        else:
            model = load_model(config, None, num_classes, device)
            if checkpoint is not None:
                state_dict = torch.load(checkpoint, device)
                self.load_weights(model.backbone, state_dict, 'backbone.')
                self.load_weights(model.neck, state_dict, 'neck.')
                self.load_weights(model.bbox_head, state_dict, 'bbox_head.')
                print("--------load weights successful-----------")

        # freeze
        self.freeze_model(model)
        self.unfreeze_model(model.bbox_head)
        # self.unfreeze_model(model.neck)
        self.freeze_bn(model)
        self.statistical_parameter(model)

        model = nn.Sequential(model.backbone, model.neck, model.bbox_head)

        """
        x=torch.rand([1,3,512,512]).to(self.device)
        a = model.backbone(x) # (tensor[1,256,128,128],tensor[1,512,64,64],tensor[1,1024,32,32],tensor[1,2048,16,16])
        b=model.neck(a) # (tensor[1,256,64,64],tensor[1,256,32,32],tensor[1,256,16,16],tensor[1,256,8,8],tensor[1,256,4,4])
        c=model.bbox_head(b) # ([tensor[1,3,64,64],tensor[1,3,32,32],tensor[1,3,16,16],tensor[1,3,8,8],tensor[1,3,4,4]], # classification
                                [tensor[1,4,64,64],tensor[1,4,32,32],tensor[1,4,16,16],tensor[1,4,8,8],tensor[1,4,4,4]], # box regression
                                [tensor[1,1,64,64],tensor[1,1,32,32],tensor[1,1,16,16],tensor[1,1,8,8],tensor[1,1,4,4]]) # centerness
        """

        return model

    def forward(self, x):
        return self.model(x)

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
                            centerness *= 0.8
                            if target_list[j][2][batch_id, :, _y, _x] >= centerness: continue
                            target_list[j][2][batch_id, :, _y, _x] = centerness

                            # if centerness == 1:
                            #     target_list[j][0][batch_id, label, _y, _x] = 1  # 1
                            #     target_list[j][1][batch_id, :, _y, _x] = torch.tensor([x1 * width_ratio,
                            #                                                            y1 * height_ratio,
                            #                                                            x2 * width_ratio,
                            #                                                            y2 * height_ratio],
                            #                                                           device=self.device)  # [l, r, t, b]

        return target_list

    def train_step(self, model, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # img_h,img_w = self.img_shape
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_preds, boxes_preds, centerness_preds = model(imgs)
            # boxes_preds 已经使用了 exp()
            # cls_preds,centerness_preds 没有使用sigmoid()

        feat_shape = [pred.shape[-2:] for pred in centerness_preds]
        target_list = self.get_targets(imgs, targets, feat_shape)

        # avg_factor = 0
        loss_cls = Variable(torch.zeros(1, requires_grad=True, device=self.device))
        loss_box = Variable(torch.zeros(1, requires_grad=True, device=self.device))
        loss_centerness = Variable(torch.zeros(1, requires_grad=True, device=self.device))

        for i in range(len(feat_shape)):
            cls_target, box_reg_target, centerness_target = target_list[i]
            # avg_factor = (centerness_target > 0).sum()
            # if avg_factor <= 0: continue

            keep = centerness_target == 1
            avg_factor = keep.sum()
            if avg_factor <= 0: continue

            # 所有正负样本
            loss_centerness += GaussianFocalLoss()(centerness_preds[i].sigmoid(), centerness_target,
                                                   avg_factor=avg_factor)

            # 满足条件的正样本
            loss_cls += Func.binary_cross_entropy(cls_preds[i].sigmoid() * keep, cls_target * keep,
                                                  reduction='none').sum() / avg_factor
            loss_box += L1Loss(reduction='none')(boxes_preds[i] * keep, box_reg_target * keep).sum() / avg_factor

            """
            # boxes_preds[i]  # l,t,r,b
            fh, fw = feat_shape[i]
            grid_x = torch.range(0, fw - 1)
            grid_y = torch.range(0, fh - 1)
            Y, X = torch.meshgrid(grid_y, grid_x)
            XY = torch.stack((X, Y), 0)[None].to(self.device)

            boxes_pred = torch.cat((0.5 + XY - boxes_preds[i][:, :2], 0.5 + XY + boxes_preds[i][:, 2:]), 1)
            

            loss_box += (CIoULoss(reduction='none')(boxes_pred, box_reg_target) * centerness_target.squeeze(
                1)).sum() / avg_factor  # 只更新正样本
            """

        return dict(
            loss_cls=loss_cls,
            loss_box=loss_box,
            loss_centerness=loss_centerness)

    @torch.no_grad()
    def pred_step(self, model, img, iou_threshold, conf_threshold, scale_factors, in_shape, with_nms):
        cls_preds, boxes_preds, centerness_preds = model(img)
        feat_shape = [pred.shape[-2:] for pred in centerness_preds]
        bs, _, img_h, img_w = img.shape

        scores_list, labels_list, boxes_list = [], [], []
        for i in range(len(feat_shape)):
            stride_w = img_w / feat_shape[i][0]
            stride_h = img_w / feat_shape[i][1]
            cls_pred = cls_preds[i].sigmoid()
            boxes_pred = boxes_preds[i]  # 自动加了 exp
            centerness_pred = centerness_preds[i].sigmoid()

            centerness_pred = get_local_maximum(centerness_pred, kernel=3)
            *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
                centerness_pred, k=min(50, centerness_pred.shape[-2:].numel()))
            batch_scores, batch_index, batch_topk_labels = batch_dets
            boxes_pred = transpose_and_gather_feat(boxes_pred, batch_index)
            boxes_pred[..., 0] += topk_xs
            boxes_pred[..., 1] += topk_ys
            boxes = boxes_pred * torch.tensor([[[stride_w, stride_h, stride_w, stride_h]]], device=self.device)
            # xywh to x1y1x2y2
            boxes = xywh2x1y1x2y2(boxes)
            cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
            scores, labels = cls_pred.max(-1)
            scores *= batch_scores
            # labels *= batch_topk_labels
            scores, labels, boxes = batch_scores[0], batch_topk_labels[0], boxes[0]

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_h - 1)

            keep = ((boxes[:, 2:] - boxes[:, :2]) > 0).sum(1) == 2
            if keep.sum() > 0:
                scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            scores_list.append(scores)
            labels_list.append(labels)
            boxes_list.append(boxes)

        # nms
        if len(scores_list) > 0:
            scores, labels, boxes = torch.cat(scores_list, 0), torch.cat(labels_list, 0), torch.cat(boxes_list, 0)
            keep = batched_nms(boxes, scores, labels, iou_threshold)
            scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            boxes /= torch.tensor(scale_factors, device=self.device)  # 缩放到原始图像上

            keep = scores > conf_threshold
            scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

            return boxes, scores, labels

        return torch.zeros([1, 4]), torch.zeros([1]), torch.zeros([1])


anchors = None
strides = 4
use_amp = False
accumulate = 1
gradient_clip_val = 0.0
lrf = 0.1
lr = 5e-4
weight_decay = 5e-5
epochs = 50
batch_size = 8
resize = (512, 512)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
config = r"D:\zyy\git\mmdetection\configs\fcos\fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py"
checkpoint = r'D:\zyy\git\mmdetection\checkpoints\fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth'
num_classes = len(classes)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.backends.cudnn.benchmark = True
# else:
#     device = torch.device('cpu')

# data
transforms = Compose([RandomHorizontalFlip(), Resize(*resize), ToTensor(), Normalize()])
dataset = FruitsNutsDataset(dir_data, classes, transforms, 0)
# dataset.show(mode='pil')
dataloader = LoadDataloader(dataset, None, 0.1, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# model
model = Fcos(None, config, checkpoint, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, 1000,
             0.5, None, None, use_amp,
             accumulate, gradient_clip_val, device, None, train_dataLoader, val_dataLoader)
model.model.to(device)

model.fit(model.model)

transforms = Compose([Resize(*resize), ToTensor(), Normalize()])
model.predict(model.model, glob_format(dir_data), transforms, device, visual=False, conf_threshold=0.1)
