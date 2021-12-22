"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点

使用 centernet的方式直接做 keypoints回归（没有使用 maskrcnn 的 roiAlign 方式）

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
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, \
    _initParmas, YOLOFHeadSelfAndKeypointsv2
from toolcv.api.define.utils.data.data import MinicocoKeypointsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.data.dataset import glob_format

from toolcv.api.define.utils.tools.tools import get_bboxes_keypoints, grid_torch, keypoints_to_heatmap
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
head = YOLOFHeadSelfAndKeypointsv2(backbone.out_channels // 4, len(anchors), 1, num_classes, 17)
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')

model = nn.Sequential(backbone, neck, head).to(device)


class YoloSelf(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_targets(self, imgs, targets, feat_shape):
        out_targets = [imgs[0].new_zeros(shape) for shape in feat_shape]
        object_target, cls_target, bbox_offset_target, wh_cls_target, keypoints_logit_target, \
        keypoints_offset_target = out_targets
        bs, c, feat_h, feat_w = wh_cls_target.shape
        wh_cls_target = wh_cls_target.contiguous().view(bs, c // 2, 2, feat_h, feat_w)
        keypoints_offset_target = keypoints_offset_target.contiguous().view(bs, 17, 2, feat_h, feat_w)

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

                keypoint = gt_keypoint[j]
                keypoint[..., 2] = (keypoint[..., 2] > 0).float()
                keypoints_logit_target[batch_id, :, cty_int, ctx_int] = keypoint[..., 2]
                keypoint[..., 0] *= width_ratio
                keypoint[..., 1] *= height_ratio
                keypoint[..., 0] -= ctx_int
                keypoint[..., 1] -= cty_int
                keypoint[..., 0] *= keypoint[..., 2]
                keypoint[..., 1] *= keypoint[..., 2]
                # keypoint[..., 0] /= feat_w
                # keypoint[..., 1] /= feat_h

                keypoints_offset_target[batch_id, :, 0, cty_int, ctx_int] = keypoint[..., 0]
                keypoints_offset_target[batch_id, :, 1, cty_int, ctx_int] = keypoint[..., 1]

        avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            object_target=object_target,
            cls_target=cls_target,
            bbox_offset_target=bbox_offset_target,
            wh_cls_target=wh_cls_target,
            keypoints_logit_target=keypoints_logit_target,
            keypoints_offset_target=keypoints_offset_target
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
        keypoints_offset = keypoints_offset.contiguous().view(bs, 17, 2, feat_h, feat_w)

        # get_target
        # target
        target_result, avg_factor = self.get_targets(imgs, targets, feat_shape)

        object_target = target_result['object_target']
        cls_target = target_result['cls_target']
        bbox_offset_target = target_result['bbox_offset_target']
        wh_cls_target = target_result['wh_cls_target']
        keypoints_logit_target = target_result['keypoints_logit_target']
        keypoints_offset_target = target_result['keypoints_offset_target']
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
            loss_boxes=loss_boxes
        ))

        # loss_keys_logit = CrossEntropyLoss(True)(keypoints_logit, keypoints_logit_target,
        #                                          wh_offset_target_weight.expand_as(keypoints_logit_target),
        #                                          avg_factor=avg_factor)
        #
        # weight = wh_offset_target_weight.unsqueeze(2).expand_as(
        #     keypoints_offset_target) * keypoints_logit_target.unsqueeze(2)
        # loss_keys_offset = L1Loss()(keypoints_offset, keypoints_offset_target,
        #                             weight, avg_factor=avg_factor)

        keep = wh_offset_target_weight.expand_as(keypoints_logit_target) == 1
        loss_keys_logit = binary_cross_entropy_with_logits(keypoints_logit[keep], keypoints_logit_target[keep])
        losses.update(dict(
            loss_keys_logit=loss_keys_logit
        ))

        keep = keypoints_logit_target.unsqueeze(2).expand_as(keypoints_offset_target) == 1
        if keep.sum() > 0:  # 只对正的做
            loss_keys_offset = l1_loss(keypoints_offset[keep], keypoints_offset_target[keep])

            losses.update(dict(
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
        boxes, scores, labels, keypoints = get_bboxes_keypoints(self.anchors, object_pred, cls_pred, bbox_offset,
                                                                wh_cls, keypoints_logit, keypoints_offset, with_nms,
                                                                iou_threshold, conf_threshold, None, None, in_shape,
                                                                False)

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
network.evalute(**dict(weight_path='weight.pth', iou_threshold=0.3,
                       conf_threshold=0.1, with_nms=False, mode='coco', iou_types='keypoints'))
# # -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data)[:20],
                       transform=test_transforms, device=device,
                       weight_path='weight.pth',
                       save_path='output', visual=False,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.1, method='pad', draw='draw_keypoint'))
