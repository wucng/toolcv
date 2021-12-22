"""
效果差 舍弃

使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms, RoIAlign
from torch.nn.functional import interpolate, binary_cross_entropy_with_logits, l1_loss

from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.api.define.utils.model.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, \
    _initParmas  # , MaskRoiHead, KeypointHead, KeypointHeadHeatMap, YOLOFHeadSelfAndKeypoints
from toolcv.api.define.utils.data.data import MinicocoKeypointsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxesv3, grid_torch, keypoints_to_heatmap, keypoints_to_heatmapv2
from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2

from toolcv.api.define.utils.tools.tools import _nms, top_k, transpose_and_gather_feat
from toolcv.api.define.utils.model.net import CBA


def get_bboxesv3_2(anchors, object_pred, cls_pred, bbox_offset, wh_cls, keypoints_logit, keypoints_offset, with_nms,
                   iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    """动态选择 anchor"""

    fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    wh_cls = transpose_and_gather_feat(wh_cls, batch_index)
    bs, h, c = wh_cls.shape
    wh_cls = wh_cls.contiguous().view(bs, h, c // 2, 2).softmax(2)
    wh_cls_score, wh_cls_index = wh_cls.max(2)
    keypoints_logit = transpose_and_gather_feat(keypoints_logit, batch_index)
    keypoints_offset = transpose_and_gather_feat(keypoints_offset, batch_index)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            iw, ih = wh_cls_index[i, j]
            _anchors.append([anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    offset_preds = bbox_offset[..., :2]
    wh_preds = bbox_offset[..., 2:]
    wh_preds = wh_preds.exp() * anchors
    # 恢复到heatmap上
    wh_preds[..., 0] *= fw
    wh_preds[..., 1] *= fh

    img_h, img_w = in_shape
    bs, c, h, w = object_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score * wh_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keypoints_logit = keypoints_logit[keep]
    keypoints_v = (keypoints_logit > 0.5).float()
    keypoints_offset = keypoints_offset[keep] * keypoints_v

    keypoints_offset *= fh
    y = keypoints_offset // fh * (img_h / h)
    x = keypoints_offset % fh * (img_w / w)

    # 裁剪到对应的boxes内
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x[i, :] = x[i, :].clamp(x1, x2)
        y[i, :] = y[i, :].clamp(y1, y2)

    keypoints = torch.stack((x, y, keypoints_v, keypoints_logit), -1)

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keypoints = keypoints[keep]
    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]
        keypoints[..., :2] -= torch.tensor(padding[:2], device=boxes.device)[None]
        keypoints[..., :2] /= torch.tensor(scale_factors[:2], device=boxes.device)[None]

    return boxes, scores, labels, keypoints


class YOLOFHeadSelfAndKeypoints(nn.Module):
    """动态选择 anchor
        同时做 keypoints
    """

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80, num_keypoints=17):
        super().__init__()
        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        self.wh_cls = nn.Conv2d(in_c, num_anchor * 2 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor
        self.keypoint_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.keypoints_logit = nn.Conv2d(in_c, num_keypoints, 3, 1, 1)
        self.keypoints_offset = nn.Conv2d(in_c, num_keypoints, 3, 1, 1)

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_offset = self.bbox_offset(bbox_subnet)
        wh_cls = self.wh_cls(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)
        #         keypoints_pred = self.keypoints_pred(bbox_subnet)
        keypoint_subnet = self.keypoint_subnet(x)
        keypoints_logit = self.keypoints_logit(keypoint_subnet)
        keypoints_offset = self.keypoints_offset(keypoint_subnet)

        return object_pred, cls_score, bbox_offset, wh_cls, keypoints_logit, keypoints_offset


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
epochs = 100
batch_size = 4
h, w = 416, 416
resize = (h, w)
dir_data = r'../input/minicoco-keypoints/minicoco'
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
backbone = Backbone('dla34', True, stride=strides, freeze_at=4)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelfAndKeypoints(backbone.out_channels // 4, len(anchors), 1, num_classes, 17)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')

model = nn.Sequential(backbone, neck, head).to(device)


class YoloSelf(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_targets(self, imgs, targets, feat_shape):
        out_targets = [imgs[0].new_zeros(shape) for shape in feat_shape]
        object_target, cls_target, bbox_offset_target, wh_cls_target, keyps_logit_target, keyps_offset_target = out_targets
        bs, c, feat_h, feat_w = wh_cls_target.shape
        wh_cls_target = wh_cls_target.contiguous().view(bs, c // 2, 2, feat_h, feat_w)

        bs, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride

        ratio = torch.tensor([width_ratio, height_ratio], dtype=imgs.dtype)

        for batch_id in range(bs):
            gt_bbox = targets[batch_id]['boxes']
            gt_label = targets[batch_id]['labels']
            gt_keypoint = targets[batch_id]['keypoints']
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

                #                 keypoints_target[batch_id, :, cty_int, ctx_int] = \
                #                     keypoints_to_heatmapv2(gt_keypoint[j], gt_bbox[[j]], 56)[0][0].float() / 56
                keypoint = gt_keypoint[j]
                keyps_logit_target[batch_id, :, cty_int, ctx_int] = (keypoint[..., -1] > 0).float()
                keypoint[..., 0] *= width_ratio
                keypoint[..., 1] *= height_ratio
                keyps_offset = (keypoint[..., 1] * feat_h + keypoint[..., 0]) / feat_h
                keyps_offset_target[batch_id, :, cty_int, ctx_int] = keyps_offset

        avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            object_target=object_target,
            cls_target=cls_target,
            bbox_offset_target=bbox_offset_target,
            wh_cls_target=wh_cls_target,
            keyps_logit_target=keyps_logit_target,
            keyps_offset_target=keyps_offset_target
        )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)

        losses = {}

        feat_shape = [out.shape for out in outputs]
        object_pred, cls_pred, bbox_offset, wh_cls, keypoints_logit, keypoints_offset = outputs
        bs, c, feat_h, feat_w = wh_cls.shape
        wh_cls = wh_cls.contiguous().view(bs, c // 2, 2, feat_h, feat_w)

        # get_target
        # target
        target_result, avg_factor = self.get_targets(imgs, targets, feat_shape)

        object_target = target_result['object_target']
        cls_target = target_result['cls_target']
        bbox_offset_target = target_result['bbox_offset_target']
        wh_cls_target = target_result['wh_cls_target']
        keyps_logit_target = target_result['keyps_logit_target']
        keyps_offset_target = target_result['keyps_offset_target']
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

        #         loss_keys_logit = CrossEntropyLoss(True)(keypoints_logit, keyps_logit_target,
        #                                   wh_offset_target_weight.expand_as(keyps_logit_target),
        #                                   avg_factor=avg_factor)
        keep = wh_offset_target_weight.expand_as(keyps_logit_target) == 1
        loss_keys_logit = binary_cross_entropy_with_logits(keypoints_logit[keep], keyps_logit_target[keep],
                                                           reduction="sum") / avg_factor

        weight = wh_offset_target_weight.expand_as(keyps_offset_target) * keyps_logit_target
        #         loss_keys_offset = L1Loss()(keypoints_offset, keyps_offset_target,
        #                                   weight,
        #                                   avg_factor=avg_factor)
        #         keep = weight==1
        loss_keys_offset = l1_loss(keypoints_offset * weight, keyps_offset_target * weight,
                                   reduction="sum") / avg_factor

        losses.update(dict(
            loss_heatmap=loss_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_boxes=loss_boxes,
            loss_keys_logit=loss_keys_logit,
            loss_keys_offset=loss_keys_offset
        ))

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
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        with_nms = kwargs['with_nms']
        if to_img:
            scale_factors = kwargs['scale_factors']
            padding = kwargs['padding']

        in_shape = imgs.shape[-2:]

        object_pred, cls_pred, bbox_offset, wh_cls, keypoints_logit, keypoints_offset = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        keypoints_logit = keypoints_logit.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        boxes, scores, labels, keypoints = get_bboxesv3_2(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls,
                                                          keypoints_logit, keypoints_offset, with_nms, iou_threshold,
                                                          conf_threshold, None,
                                                          None, in_shape, False)

        if to_img:
            # 缩放到原始图像上
            boxes -= torch.tensor(padding, device=boxes.device)[None]
            boxes /= torch.tensor(scale_factors, device=boxes.device)[None]
            keypoints[..., :2] -= torch.tensor(padding[:2], device=boxes.device)[None]
            keypoints[..., :2] /= torch.tensor(scale_factors[:2], device=boxes.device)[None]

        return dict(boxes=boxes, scores=scores, labels=labels, keypoints=keypoints)


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
network.fit()
# -----------------------eval ---------------------------
# network.evalute(**dict(weight_path='weight.pth', iou_threshold=0.3,
#                        conf_threshold=0.2, with_nms=False, mode='coco', iou_types='keypoints'))
# # # -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data)[:30],
                       transform=test_transforms, device=device,
                       weight_path='weight.pth',
                       save_path='output', visual=True,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.2, method='pad', draw='draw_keypoint'))
