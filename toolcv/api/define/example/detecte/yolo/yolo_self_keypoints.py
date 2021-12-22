"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点

使用 maskrcnn 的 roiAlign 方式 获取每个建议框对应的特征 来做 keypoints
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms, RoIAlign
from torch.nn.functional import interpolate, binary_cross_entropy_with_logits, cross_entropy

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, \
    _initParmas, MaskRoiHead, KeypointHeadHeatMap
from toolcv.api.define.utils.data.data import MinicocoKeypointsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, keypoints_to_heatmap
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

# ignore = True
surrounding = True

seed = 100
torch.manual_seed(seed)

anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
anchors = torch.tensor(anchors, dtype=torch.float32)

strides = 16
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
dir_data = r'D:\data\coco\minicoco'
classes = ['person']
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

dataset = MinicocoKeypointsDataset(dir_data, classes, transforms)
val_dataset = MinicocoKeypointsDataset(dir_data, classes, test_transforms)

# dataset.show(mode='cv2')
dataloader = LoadDataloader(dataset, val_dataset, 0.1, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
backbone = Backbone('resnet18', True, stride=strides)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelf(backbone.out_channels // 4, len(anchors), 1, num_classes)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')

roiHead = MaskRoiHead(strides)
keypointsHead = KeypointHeadHeatMap(backbone.out_channels // 4, 17)
_initParmas(keypointsHead.modules(), mode='normal')

model = nn.Sequential(backbone, neck, head, roiHead, keypointsHead).to(device)

model.feature = lambda x: model[:2](x)
# model.rpn = lambda x: model[2](x)
model.forward = lambda x: model[2](x)
model.roi = lambda x, proposal: model[3](x, proposal)
model.rcnn = lambda x: model[-1](x)


class YoloSelf(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    @torch.no_grad()
    def get_proposal(self, model_outputs, targets=None, **kwargs):
        """evalute_step()"""

        iou_threshold = kwargs['iou_threshold'] if 'iou_threshold' in kwargs else 0.3
        conf_threshold = kwargs['conf_threshold'] if 'conf_threshold' in kwargs else 0.2
        with_nms = kwargs['with_nms'] if 'with_nms' in kwargs else True

        in_shape = self.img_shape

        object_pred, cls_pred, bbox_offset, wh_cls = model_outputs
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()

        bs = object_pred.size(0)
        boxes_list, scores_list, labels_list, keypoints_list, keypoints_valid_list = [], [], [], [], []
        keep_bs = [1] * bs
        for i in range(bs):
            boxes, scores, labels = get_bboxesv3(self.anchors, object_pred[[i]], cls_pred[[i]], bbox_offset[[i]],
                                                 wh_cls[[i]], with_nms, iou_threshold, conf_threshold, None, None,
                                                 in_shape, False)
            if len(labels) == 0:
                keep_bs[i] = 0
            else:
                # 对齐到目标数据上
                if targets is not None:
                    gt_labels = targets[i]['labels'].to(self.device)
                    gt_boxes = targets[i]['boxes'].to(self.device)
                    gt_keypoints = targets[i]['keypoints'].float().to(self.device)
                    ious = box_iou(boxes, gt_boxes)
                    # 预测框对应的最大gt
                    values, indices = ious.max(1)
                    # 每个gt 对应的最大anchor
                    # values_gt, indices_gt = ious.max(0)
                    keep = values > 0.5
                    if keep.sum() == 0:
                        keep_bs[i] = 0
                    else:
                        indices = indices[keep]
                        boxes = boxes[keep]
                        scores = scores[keep]
                        gt_keypoints = gt_keypoints[indices]
                        labels = gt_labels[indices]

                        heatmaps, valid = keypoints_to_heatmap(gt_keypoints, boxes, 56)
                        keypoints_list.append(heatmaps)
                        keypoints_valid_list.append(valid)

                        boxes_list.append(boxes)
                        scores_list.append(scores)
                        labels_list.append(labels)

                else:
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)

        targets = dict(boxes=boxes_list, scores=scores_list, labels=labels_list,
                       kps_logit=keypoints_list, kps_valid=keypoints_valid_list,
                       keep_bs=keep_bs)

        return targets

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            feature = self.model.feature(imgs)
            outputs = self.model(feature)

        losses = {}

        # if do_bboxes:
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
        loss_heatmap = GaussianFocalLoss()(object_pred.sigmoid()[keep], object_target[keep],
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

        losses.update(dict(
            loss_heatmap=loss_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_boxes=loss_boxes))

        # if do_keypoints:
        # in_shape = imgs.shape[-2:]
        proposals = self.get_proposal(outputs, targets)
        keep_bs = torch.tensor(proposals['keep_bs'])
        if keep_bs.sum() > 0:
            proposal_boxes = proposals['boxes']
            proposal_labels = proposals['labels']
            proposal_labels = torch.cat(proposal_labels, 0)
            proposal_kps_logit = proposals['kps_logit']
            proposal_kps_logit = torch.cat(proposal_kps_logit, 0).squeeze(1)
            proposal_kps_valid = proposals['kps_valid']
            proposal_kps_valid = torch.cat(proposal_kps_valid, 0).squeeze(1)
            keep = proposal_kps_valid == 1

            if keep.sum() > 0:
                roi = self.model.roi(feature[keep_bs == 1], proposal_boxes)
                rcnn_out = self.model.rcnn(roi)
                loss_keypoints = cross_entropy(rcnn_out[keep].view(-1, 56 * 56), proposal_kps_logit[keep].long(),
                                               reduction='mean')
                losses.update(dict(loss_keypoints=loss_keypoints))

        return losses

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        return self.do_predict(imgs, True, **kwargs)

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):

        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        return self.do_predict(imgs, **kwargs)

    def do_predict(self, imgs, to_img=False, **kwargs):
        # iou_threshold = kwargs['iou_threshold']
        # conf_threshold = kwargs['conf_threshold']
        # with_nms = kwargs['with_nms']
        # in_shape = imgs.shape[-2:]

        feature = self.model.feature(imgs)
        outputs = self.model(feature)
        proposals = self.get_proposal(outputs, None, **kwargs)
        proposal_boxes = proposals['boxes']
        proposal_labels = proposals['labels']
        proposal_scores = proposals['scores']
        bs = feature.size(0)

        keypoints_list = []
        for i in range(bs):
            if len(proposal_boxes) > 0:
                rois = proposal_boxes[i]
                roi = self.model.roi(feature[[i]], [rois])
                rcnn_out = self.model.rcnn(roi)  # [-1,17,56,56]
                n, c, h, w = rcnn_out.shape
                rcnn_out = rcnn_out.view(n, c, h * w).softmax(-1)
                values, indices = rcnn_out.max(-1)  # [n,17]
                y = indices // h
                x = indices % h
                valid = (values > 0.1).float()
                # x *= valid
                # y *= valid

                heatmap_size = h

                offset_x = rois[:, 0]
                offset_y = rois[:, 1]
                scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
                scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

                offset_x = offset_x[:, None]
                offset_y = offset_y[:, None]
                scale_x = scale_x[:, None]
                scale_y = scale_y[:, None]

                x = x / scale_x + offset_x
                y = y / scale_y + offset_y

                keypoints = torch.stack((x, y, valid), -1)

                # 裁剪到对应的boxes内
                for i in range(len(rois)):
                    x1, y1, x2, y2 = rois[i]
                    keypoints[i, :, 0] = keypoints[i, :, 0].clamp(x1, x2)
                    keypoints[i, :, 1] = keypoints[i, :, 1].clamp(y1, y2)

                keypoints_list.append(keypoints)

        if len(keypoints_list) == 0:
            return dict(boxes=torch.zeros([1, 4]), scores=torch.zeros([1]),
                        labels=torch.zeros([1]), keypoints=torch.zeros([1, 17, 3]))

        target = dict(boxes=proposal_boxes[0], scores=proposal_scores[0],
                      labels=proposal_labels[0], keypoints=keypoints_list[0])

        if to_img:
            # 缩放到原始图像上
            scale_factors = kwargs['scale_factors']
            padding = kwargs['padding']

            target['boxes'] -= torch.tensor(padding, device=self.device)[None]
            target['boxes'] /= torch.tensor(scale_factors, device=self.device)[None]
            target['keypoints'][..., :2] -= torch.tensor(padding[:2], device=self.device)[None]
            target['keypoints'][..., :2] /= torch.tensor(scale_factors[:2], device=self.device)[None]

        return target


network = YoloSelf(**dict(model=model, num_classes=num_classes,
                          img_shape=resize, anchors=anchors,
                          strides=strides, epochs=epochs,
                          lr=lr, weight_decay=weight_decay, lrf=lrf,
                          warmup_iters=1000, gamma=0.5, optimizer=None,
                          scheduler=None, use_amp=use_amp, accumulate=accumulate,
                          gradient_clip_val=gradient_clip_val, device=device,
                          criterion=None, train_dataloader=train_dataLoader,
                          val_dataloader=val_dataLoader))

# network.statistical_parameter()

# -----------------------fit ---------------------------
# network.fit()
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path='weight.pth', iou_threshold=0.3,
                       conf_threshold=0.2, with_nms=False, mode='coco', iou_types='keypoints'))
# # -----------------------predict ---------------------------
# network.predict(**dict(img_paths=glob_format(dir_data)[:20],
#                        transform=test_transforms, device=device,
#                        weight_path='weight.pth',
#                        save_path='output', visual=False,
#                        with_nms=False, iou_threshold=0.3,
#                        conf_threshold=0.2, method='pad', draw='draw_keypoint'))
