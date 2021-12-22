"""
https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth

test_cfg:

{'nms_pre': 1000,
 'nms': {'type': 'nms', 'iou_threshold': 0.45},
 'min_bbox_size': 0,
 'score_thr': 0.02,
 'max_per_img': 200}
"""

import torch
from torch import nn
from torch.nn import functional as Func
from torch.utils.data import Dataset, DataLoader, random_split
import math

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models.builder import build_loss

from toolcv.api.define.utils.model.mmdet import _BaseNet, load_model
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.api.define.utils.model.net import CBA, _initParmas
from toolcv.data.dataset import glob_format

anchor_generator = dict(
    type='SSDAnchorGenerator',
    scale_major=False,
    input_size=300,
    strides=[8, 16, 32, 64, 100, 300],
    ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
    basesize_ratio_range=(0.1, 0.9))
bbox_coder = dict(
    type='DeltaXYWHBBoxCoder',
    clip_border=True,
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
)

loss_cls = dict(
    type='CrossEntropyLoss',
    use_sigmoid=True,
    loss_weight=1.0)
loss_bbox = dict(
    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)


class SSD(_BaseNet):
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

        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def load_model(self, config, checkpoint, num_classes, device):
        if num_classes == 81:
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
        x=torch.rand([1,3,300,300]).to(self.device)
        a = model.backbone(x) # (tensor[1,512,38,38],tensor[1,1024,19,19])
        b=model.neck(a) # (tensor[1,512,38,38],tensor[1,1024,19,19],tensor[1,512,10,10],tensor[1,256,5,5],tensor[1,256,3,3],tensor[1,256,1,1])
        c=model.bbox_head(b) # ([
                                    t[1,324,38,38],t[1,486,19,19],t[1,486,10,10],t[1,486,5,5],t[1,324,3,3],t[1,324,1,1]
                                ],
                                [
                                    t[1,16,38,38],t[1,24,19,19],t[1,24,10,10],t[1,24,5,5],t[1,16,3,3],t[1,16,1,1]
                                ])
        """

        return model

    def forward(self, x):
        return self.model(x)

    def train_step(self, model, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        # targets = torch.stack(targets, 0).to(self.device)
        gt_bboxes = [target['boxes'].to(self.device) for target in targets]
        gt_labels = [target['labels'].to(self.device) for target in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            cls_scores, bbox_preds = model(imgs)  # heatmap 没有使用 sigmoid

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = self.device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    @torch.no_grad()
    def pred_step(self, model, img, iou_threshold, conf_threshold, scale_factors, in_shape, with_nms):
        center_heatmap_preds, cls_preds, wh_preds, offset_preds = model(img)
        center_heatmap_preds, cls_preds = center_heatmap_preds.sigmoid(), cls_preds.sigmoid()

        det_results = get_bboxes(center_heatmap_preds, cls_preds, wh_preds, offset_preds, with_nms,
                                 iou_threshold, scale_factors, in_shape)

        boxes = det_results[0][0][..., :4]
        scores = det_results[0][0][..., 4]
        labels = det_results[0][1]

        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        return boxes, scores, labels


    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


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
classes = ['date', 'fig', 'hazelnut', '__background__']
config = r"D:\zyy\git\mmdetection\configs\ssd\ssd300_coco.py"
checkpoint = r'D:\zyy\git\mmdetection\checkpoints\ssd300_coco_20210803_015428-d231a06e.pth'
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
model = SSD(None, config, checkpoint, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, 1000,
            0.5, None, None, use_amp, accumulate, gradient_clip_val, device, None, train_dataLoader,
            val_dataLoader)
model.model.to(device)

model.fit(model.model)

transforms = Compose([Resize(*resize), ToTensor(), Normalize()])
model.predict(model.model, glob_format(dir_data), transforms, device, visual=False, conf_threshold=0.1)
