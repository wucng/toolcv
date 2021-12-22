import torch
from torch import nn
from torch.nn import functional as Func
from torch.utils.data import Dataset, DataLoader, random_split
import math

from mmdet.models.losses import GaussianFocalLoss, L1Loss, CrossEntropyLoss
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)

from toolcv.api.define.utils.model.mmdet import _BaseNet, load_model, gaussian_radius, gen_gaussian_target, batched_nms
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data.augment import *
from toolcv.data.dataset import glob_format
from toolcv.api.define.utils.model.net import CBA, _initParmas


def get_bboxes(
        center_heatmap_preds,
        cls_preds,
        wh_preds,
        offset_preds,
        with_nms=False, iou_threshold=0.3,
        scale_factors=(), img_shape=()):
    """Transform network output for a batch into bbox predictions.

    Args:
        center_heatmap_preds (list[Tensor]): center predict heatmaps for
            all levels with shape (B, num_classes, H, W).
        wh_preds (list[Tensor]): wh predicts for all levels with
            shape (B, 2, H, W).
        offset_preds (list[Tensor]): offset predicts for all levels
            with shape (B, 2, H, W).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        rescale (bool): If True, return boxes in original image space.
            Default: True.
        with_nms (bool): If True, do nms before return boxes.
            Default: False.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.
    """
    assert len(center_heatmap_preds) == len(wh_preds) == len(
        offset_preds) == 1
    batch_det_bboxes, batch_labels = decode_heatmap(
        center_heatmap_preds,
        cls_preds,
        wh_preds,
        offset_preds,
        img_shape)

    # batch_border = batch_det_bboxes.new_tensor(
    #     border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
    # batch_det_bboxes[..., :4] -= batch_border

    batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
        scale_factors).unsqueeze(1)

    if with_nms:
        det_results = []
        for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                            batch_labels):
            # det_bbox, det_label = _bboxes_nms(det_bboxes, det_labels)

            boxes = det_bboxes[..., :4]
            scores = det_bboxes[..., 4]
            keep = batched_nms(boxes, scores, det_labels, iou_threshold)
            det_bbox, det_label = det_bboxes[keep], det_labels[keep]

            det_results.append(tuple([det_bbox, det_label]))
    else:
        det_results = [
            tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
        ]
    return det_results


def decode_heatmap(
        center_heatmap_pred,
        cls_pred,
        wh_pred,
        offset_pred,
        img_shape,
        k=100,
        kernel=3):
    """Transform outputs into detections raw bbox prediction.

    Args:
        center_heatmap_pred (Tensor): center predict heatmap,
           shape (B, num_classes, H, W).
        wh_pred (Tensor): wh predict, shape (B, 2, H, W).
        offset_pred (Tensor): offset predict, shape (B, 2, H, W).
        img_shape (list[int]): image shape in [h, w] format.
        k (int): Get top k center keypoints from heatmap. Default 100.
        kernel (int): Max pooling kernel for extract local maximum pixels.
           Default 3.

    Returns:
        tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
           the following Tensors:

          - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
          - batch_topk_labels (Tensor): Categories of each box with \
              shape (B, k)
    """
    height, width = center_heatmap_pred.shape[2:]
    inp_h, inp_w = img_shape

    center_heatmap_pred = get_local_maximum(
        center_heatmap_pred, kernel=kernel)

    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
        center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    cls = transpose_and_gather_feat(cls_pred, batch_index)
    _batch_scores, batch_topk_labels = cls.max(-1)
    batch_scores *= _batch_scores

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

    batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
    batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                             dim=-1)
    return batch_bboxes, batch_topk_labels


class BboxHead(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.heatmap_head = nn.Sequential(CBA(in_c, in_c, 3, 1, use_bn=False), nn.Conv2d(in_c, 1, 1))
        self.cls_head = nn.Sequential(CBA(in_c, in_c, 3, 1, use_bn=False), nn.Conv2d(in_c, num_classes, 1))
        self.wh_head = nn.Sequential(CBA(in_c, in_c, 3, 1, use_bn=False), nn.Conv2d(in_c, 2, 1))
        self.offset_head = nn.Sequential(CBA(in_c, in_c, 3, 1, use_bn=False), nn.Conv2d(in_c, 2, 1))

        _initParmas(self.modules())

    def forward(self, x):
        return self.heatmap_head(x[0]), self.cls_head(x[0]), self.wh_head(x[0]), self.offset_head(x[0])


class CenterNet(_BaseNet):
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
        try:
            model = load_model(config, checkpoint, num_classes, device)  # 类别对不上会报错
        except Exception as e:
            print(e)
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

        bbox_head = BboxHead(64, self.num_classes)

        # model = nn.Sequential(model.backbone, model.neck, model.bbox_head)
        model = nn.Sequential(model.backbone, model.neck, bbox_head)

        """
        x=torch.rand([1,3,512,512]).to(self.device)
        a = model.backbone(x) # (tensor[1,64,128,128],tensor[1,128,64,64],tensor[1,256,32,32],tensor[1,512,16,16])
        b=model.neck(a) # (tensor[1,64,128,128],)
        c=model.bbox_head(b) # ([tensor[1,1,128,128]], # headmap
                                [tensor[1,3,128,128]], # classification 
                                [tensor[1,2,128,128]], # wh 
                                [tensor[1,2,128,128]]) # offset
        """

        return model

    def forward(self, x):
        return self.model(x)

    def get_targets(self, imgs, targets, feat_shape):
        bs, _, feat_h, feat_w = feat_shape
        _, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride

        ratio = torch.tensor([[width_ratio, height_ratio]], dtype=imgs.dtype)

        # 转换成最终的target
        center_heatmap_target = imgs[0].new_zeros([bs, 1, feat_h, feat_w])
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
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]

                gen_gaussian_target(center_heatmap_target[batch_id, 0], [ctx_int, cty_int], radius)
                cls_target[batch_id, ind, cty_int, ctx_int] = 1

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            cls_target=cls_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def train_step(self, model, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        # targets = torch.stack(targets, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            center_heatmap_pred, cls_pred, wh_pred, offset_pred = model(imgs)  # heatmap 没有使用 sigmoid

        # target
        target_result, avg_factor = self.get_targets(imgs, targets, center_heatmap_pred.shape)
        center_heatmap_target = target_result['center_heatmap_target']
        cls_target = target_result['cls_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # loss
        loss_center_heatmap = GaussianFocalLoss()(center_heatmap_pred.sigmoid(), center_heatmap_target,
                                                  avg_factor=avg_factor)
        # loss_cls = CrossEntropyLoss(use_mask=True).forward(cls_pred, cls_target, wh_offset_target_weight,
        #                                                    avg_factor=avg_factor * 2)  # 会自动加 sigmoid

        keep = center_heatmap_target.expand_as(cls_target) == 1
        loss_cls = Func.binary_cross_entropy_with_logits(cls_pred[keep], cls_target[keep], reduction='sum') / (
                avg_factor * 2)

        """
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
        """

        keep = wh_offset_target_weight == 1
        loss_wh = 0.1 * Func.l1_loss(wh_pred[keep], wh_target[keep], reduction='sum') / (avg_factor * 2)
        loss_offset = Func.l1_loss(offset_pred[keep], offset_target[keep], reduction='sum') / (avg_factor * 2)

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
# config = r"D:\zyy\git\mmdetection\configs\centernet\centernet_resnet18_dcnv2_140e_coco.py"
# checkpoint = r'D:\zyy\git\mmdetection\checkpoints\centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'
config = r"D:\zyy\git\mmdetection\configs\centernet\centernet_resnet18_140e_coco.py"
checkpoint = r'D:\zyy\git\mmdetection\checkpoints\centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth'
num_classes = len(classes)
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# data
transforms = Compose([RandomHorizontalFlip(), Resize(*resize), ToTensor(), Normalize()])
dataset = FruitsNutsDataset(dir_data, classes, transforms, 0)
# dataset.show(mode='pil')
dataloader = LoadDataloader(dataset, None, 0.1, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# model
model = CenterNet(None, config, checkpoint, num_classes, resize, anchors, strides, epochs, lr, weight_decay, lrf, 1000,
                  0.5, None, None, use_amp, accumulate, gradient_clip_val, device, None, train_dataLoader,
                  val_dataLoader)
model.model.to(device)

model.fit(model.model)

transforms = Compose([Resize(*resize), ToTensor(), Normalize()])
model.predict(model.model, glob_format(dir_data), transforms, device, visual=False, conf_threshold=0.1)
