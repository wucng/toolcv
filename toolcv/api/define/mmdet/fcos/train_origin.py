"""
https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py
https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth
效果差？？？？？
"""

import torch
from torch import nn
from torch.nn import functional as Func
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import math

# from fvcore.nn import sigmoid_focal_loss

# from mmdet.core import multi_apply
from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmdet.models.losses import FocalLoss, SmoothL1Loss, CIoULoss, CrossEntropyLoss

from toolcv.api.define.utils.model.mmdet import _BaseNet, _initParmas, load_model, batched_nms
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.data.dataset import glob_format
from toolcv.api.define.utils.model.net import CBA

INF = 1e8
regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))


class Fcos(_BaseNet):
    def __init__(self, model, config, checkpoint, num_classes, img_shape=(), anchors=None, strides=4,
                 epochs=10, lr=5e-4, weight_decay=5e-5, lrf=0.1,
                 warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):

        super().__init__(num_classes, img_shape, anchors, strides, epochs, lr, weight_decay, lrf, warmup_iters, gamma,
                         optimizer,
                         scheduler,
                         use_amp, accumulate, gradient_clip_val, device, criterion, train_dataloader, val_dataloader)

        if model is None:
            self.model = self.load_model(config, checkpoint, num_classes, device)
        else:
            self.model = model

        self.regress_ranges = regress_ranges
        self.center_sampling = False
        self.center_sample_radius = 1.5
        self.norm_on_bbox = False
        self.centerness_on_reg = False

        self.loss_cls = FocalLoss()
        self.loss_bbox = CIoULoss()
        self.loss_centerness = CrossEntropyLoss(True)

        self.strides = [8, 16, 32, 64, 128]

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
        self.unfreeze_model(model.neck)
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

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                                         left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        x_range = torch.arange(w, device=device).to(dtype)
        y_range = torch.arange(h, device=device).to(dtype)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        # return y, x
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points

    def train_step(self, model, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        gt_bboxes = [target['boxes'].to(self.device) for target in targets]
        gt_labels = [target['labels'].to(self.device) for target in targets]

        # img_h,img_w = self.img_shape
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_scores, bbox_preds, centernesses = model(imgs)
            # bbox_preds 已经使用了 exp()
            # cls_scores,centernesses 没有使用sigmoid()

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @torch.no_grad()
    def pred_step(self, model, img, iou_threshold, conf_threshold, scale_factors, in_shape, with_nms):
        cls_scores, bbox_preds, centernesses = model(img)

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]

        img_shapes = in_shape

        bboxes, scores, labels = self._get_bboxes(cls_score_list, bbox_pred_list,
                                                  centerness_pred_list, mlvl_points,
                                                  img_shapes, scale_factors, with_nms,
                                                  iou_threshold, conf_threshold)

        return bboxes, scores, labels

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    with_nms=True, iou_threshold=0.3, conf_threshold=0.3):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            100, device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()

                points = points[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
            scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        # if torch.onnx.is_in_onnx_export() and with_nms:
        #     from mmdet.core.export import add_dummy_nms_for_onnx
        #     batch_mlvl_scores = batch_mlvl_scores * (
        #         batch_mlvl_centerness.unsqueeze(2))
        #     max_output_boxes_per_class = 200 #cfg.nms.get(
        #         # 'max_output_boxes_per_class', 200)
        #     iou_threshold = 0.5 #cfg.nms.get('iou_threshold', 0.5)
        #     score_threshold = 0.3 #cfg.score_thr
        #     nms_pre = -1 #cfg.get('deploy_nms_pre', -1)
        #     return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
        #                                   max_output_boxes_per_class,
        #                                   iou_threshold, score_threshold,
        #                                   nms_pre, 100)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        # padding = batch_mlvl_scores.new_zeros(batch_size,
        #                                       batch_mlvl_scores.shape[1], 1)
        # batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        bboxes, cls_score, centerness = batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_centerness
        scores, labels = cls_score.max(-1)
        scores *= centerness

        keep = scores > conf_threshold
        bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

        if with_nms:
            keep = batched_nms(bboxes, scores, labels, iou_threshold)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

        return bboxes, scores, labels


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

# model.fit(model.model)

transforms = Compose([Resize(*resize), ToTensor(), Normalize()])
model.predict(model.model, glob_format(dir_data), transforms, device, visual=False, conf_threshold=0.01)
