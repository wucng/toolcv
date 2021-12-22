import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss

from toolcv.api.define.utils.model.base import _BaseNetV2
from toolcv.api.define.utils.model.net import Backbone, YOLOV3Neck, YOLOV3Head, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxes_yolov3 as get_bboxesv3, grid_torch
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

ignore = True

seed = 100
torch.manual_seed(seed)

anchors = [[[0.1, 0.1], [0.15, 0.15], [0.2, 0.2]],  # s =8
           [[0.25, 0.25], [0.3, 0.3], [0.35, 0.35]],  # s =16
           [[0.4, 0.4], [0.45, 0.45], [0.5, 0.5]]]  # s =32 # 缩放到0~1 [w,h]
anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)

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
backbone = Backbone('resnet18', True, num_out=3)
neck = YOLOV3Neck(backbone.out_channels, True)
head = YOLOV3Head(backbone.out_channels // 2, 3, num_classes, True)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, neck, head).to(device)


class Yolov3(_BaseNetV2):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):
        super().__init__(model, num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters,
                         gamma, optimizer, scheduler, use_amp, accumulate, gradient_clip_val, device, criterion,
                         train_dataloader, val_dataloader)

    def get_targets(self, imgs, targets, feat_shape):
        out_targets = []
        ratios = []
        bs, _, img_h, img_w = imgs.shape
        for shape in feat_shape:
            out_targets.append(imgs[0].new_zeros(shape))
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
        # get_target
        # target
        out_targets, avg_factor = self.get_targets(imgs, targets, feat_shape)

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
        loss_center_heatmap = GaussianFocalLoss()(centerness_pred.sigmoid()[keep], centerness_target[keep],
                                                  avg_factor=avg_factor)

        loss_wh = L1Loss(loss_weight=0.8)(
            wh_pred,
            wh_target,
            wh_offset_target_weight.expand_as(wh_target),
            avg_factor=avg_factor)

        loss_offset = L1Loss()(
            offset_pred,
            offset_target,
            wh_offset_target_weight.expand_as(offset_target),
            avg_factor=avg_factor)

        loss_cls = CrossEntropyLoss(True)(cls_pred, cls_target, wh_offset_target_weight.expand_as(cls_target),
                                          avg_factor=avg_factor)  # 会自动添加 sigmoid

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

            boxes, scores, labels = get_bboxesv3(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
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

            boxes, scores, labels = get_bboxesv3(centerness_preds, cls_preds, wh_preds, offset_preds, with_nms,
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
                 gamma=0.5, optimizer=None, scheduler=None, use_amp=use_amp, accumulate=accumulate,
                 gradient_clip_val=gradient_clip_val, device=device, criterion=None,
                 train_dataloader=train_dataLoader, val_dataloader=val_dataLoader)

# network.statistical_parameter()

# -----------------------fit ---------------------------
# network.fit()
# -----------------------eval ---------------------------
# network.evalute(mode='mmdet', conf_threshold=0.2)
# -----------------------predict ---------------------------
network.predict(glob_format(dir_data), test_transforms, device, visual=False,
                conf_threshold=0.2, with_nms=True, method='pad')
