"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点

使用 maskrcnn 的 roiAlign 方式 获取每个建议框对应的特征 来做 keypoints

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.773
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.948
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.895
IoU metric: keypoints
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.797
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.871
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms, RoIAlign
from torch.nn.functional import interpolate, binary_cross_entropy_with_logits, cross_entropy
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, ciou_loss, \
    diou_loss  # , giou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, \
    _initParmas, MaskRoiHead, KeypointHeadHeatMap
# from toolcv.api.define.utils.data.data import MinicocoKeypointsDataset, LoadDataloader
# from toolcv.api.define.utils.data import augment as aug
# from toolcv.data.dataset import glob_format
from toolcv.utils.data.data import DataDefine, MinicocoKeypointsDataset
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, get_bboxes_DA as get_bboxes
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

from toolcv.api.define.utils.tools.tools import keypoints_to_heatmap  # get_bboxesv3
from toolcv.tools.utils import box_iou  # , wh_iou, xywh2x1y1x2y2

from toolcv.tools.anchor import gen_anchorv2  # , kmean_gen_anchor

do_bboxes = True
do_keypoints = True

# ignore = True
surrounding = True

seed = 100
set_seed(seed)

# anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
# anchors = torch.tensor(anchors, dtype=torch.float32)
# anchors = torch.stack((anchors, anchors), -1)
# 通过聚类生成先验框
wsize = [0.06956193, 0.22290888, 0.37818182, 0.5574343, 0.86061215]
hsize = [0.06077686, 0.28725243, 0.5210408, 0.6929334, 0.88046366]
anchors_w = torch.tensor(wsize, dtype=torch.float32)
anchors_h = torch.tensor(hsize, dtype=torch.float32)
anchors = torch.stack((anchors_w, anchors_h), -1)  # 5x2

n_anchors = len(anchors)
strides = 8
use_amp = True  # 推荐 先 True 再 False
accumulate = 2  # 推荐 先 >1 再 1
gradient_clip_val = 0.0 if use_amp else 1.0
lrf = 0.1
lr = 5e-4 if use_amp else 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 8
h, w = 416, 416
resize = (h, w)
dir_data = r'D:\data\coco\minicoco'
classes = ['person']
num_classes = len(classes)
device = get_device()
weight_path = 'weight.pth'
each_batch_scheduler = False  # 推荐 先 False 再 True
use_iouloss = False  # 推荐 先 False 再 True
save_path = 'output'
summary = SummaryWriter(save_path)

# 通过聚类生成先验框
# gen_anchorv2(MinicocoKeypointsDataset(dir_data, classes), 15)
# exit(0)
# w [0.06956193 0.22290888 0.37818182 0.5574343  0.86061215]
# h [0.06077686 0.28725243 0.5210408  0.6929334  0.88046366]

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
train_dataset = MinicocoKeypointsDataset(dir_data, classes, data.train_transforms)
val_dataset = MinicocoKeypointsDataset(dir_data, classes, data.val_transforms)
data.get_dataloader(train_dataset, val_dataset)
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# train_dataset.show(20, "plt", draw='draw_keypoint')
# ----------------model --------------------------
backbone = Backbone('resnet34', True, stride=strides)
spp = SPPv2(backbone.out_channels, backbone.out_channels // 4)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelf(backbone.out_channels // 4, len(anchors), 1, num_classes)
_initParmas(spp.modules(), mode='kaiming')
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')

if not do_keypoints:
    model = nn.Sequential(backbone, spp, neck, head).to(device)
    model.feature = lambda x: model[:3](x)
    model.forward = lambda x: model[3](x)
else:
    roiHead = MaskRoiHead(strides)
    keypointsHead = KeypointHeadHeatMap(backbone.out_channels // 4, 17)
    _initParmas(keypointsHead.modules(), mode='normal')

    model = nn.Sequential(backbone, spp, neck, head, roiHead, keypointsHead).to(device)

    model.feature = lambda x: model[:3](x)
    # model.rpn = lambda x: model[3](x)
    model.forward = lambda x: model[3](x)
    model.roi = lambda x, proposal: model[4](x, proposal)
    model.rcnn = lambda x: model[-1](x)

load_weight(model, weight_path, "", device=device)
model.train()

if each_batch_scheduler:
    optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
                                               weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
else:
    optimizer, scheduler = None, None


class YoloSelf(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                a_w, a_h = self.anchors[iw, 0], self.anchors[ih, 1]

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
            boxes, scores, labels = get_bboxes(self.anchors, object_pred[[i]], cls_pred[[i]], bbox_offset[[i]],
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

        losses = {}

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            feature = self.model.feature(imgs)
            outputs = self.model(feature)

            if do_bboxes:
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
                loss_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep], object_target[keep]).sum() / avg_factor

                loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                              wh_offset_target_weight.expand_as(bbox_offset_target)).sum() / avg_factor

                loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                            wh_offset_target_weight.expand_as(cls_target)).sum() / avg_factor

                loss_wh = (F.cross_entropy(wh_cls, wh_cls_target, reduction="none") *
                           wh_offset_target_weight.expand_as(wh_cls_target)).sum() / avg_factor

                losses.update(dict(
                    loss_heatmap=loss_heatmap,
                    loss_cls=loss_cls,
                    loss_wh=loss_wh,
                    loss_boxes=loss_boxes))

            if do_keypoints:
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
                        loss_keypoints = cross_entropy(rcnn_out[keep].view(-1, 56 * 56),
                                                       proposal_kps_logit[keep].long(),
                                                       reduction='mean')

                        # loss_keypoints = (cross_entropy(rcnn_out.contiguous().view(-1, 56 * 56),
                        #                                 proposal_kps_logit.contiguous().view(-1).long(),
                        #                                 reduction='none') *
                        #                   proposal_kps_valid.contiguous().view(-1)).sum() / proposal_kps_valid.sum()

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

        if len(proposal_boxes) == 0:
            proposal_boxes = torch.zeros([1, 1, 4], device=device)
            proposal_scores = torch.zeros([1, 1])
            proposal_labels = torch.zeros([1, 1])

        target = dict(boxes=proposal_boxes[0], scores=proposal_scores[0],
                      labels=proposal_labels[0])

        if do_keypoints:
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
                keypoints_list = torch.zeros([1, 1, 17, 3], device=device)

            target.update(dict(keypoints=keypoints_list[0]))

        if to_img:
            # 缩放到原始图像上
            scale_factors = kwargs['scale_factors']
            padding = kwargs['padding']

            target['boxes'] -= torch.tensor(padding, device=self.device)[None]
            target['boxes'] /= torch.tensor(scale_factors, device=self.device)[None]
            if "keypoints" in target:
                target['keypoints'][..., :2] -= torch.tensor(padding[:2], device=self.device)[None]
                target['keypoints'][..., :2] /= torch.tensor(scale_factors[:2], device=self.device)[None]

        return target


network = YoloSelf(**dict(model=model, num_classes=num_classes,
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
                       conf_threshold=0.2, with_nms=False, mode='coco',
                       iou_types='keypoints' if do_keypoints else "none"))
# -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data)[:20],
                       transform=test_transforms, device=device,
                       weight_path=weight_path,
                       save_path=save_path, visual=False,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.2, method='pad',
                       draw='draw_keypoint' if do_keypoints else "draw_rect"))
