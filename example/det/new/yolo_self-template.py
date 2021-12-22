"""
使用centernet方式 同时考虑中心附近的点
网络结构 使用的是yolof
动态选择anchor
    从备选的anchor中选择合适的anchor
    固定使用中心点所在网格的左上角 作为中心点

epoch = 50;strides = 8;'resnet18',freeze_at=5
use_iouloss = False;multiscale=False

from toolcv.utils.net.basev3 import BaseNet
ema = True
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.805
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.934
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.934

ema = False
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.786
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.931
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.913

--------------------------------------------------------------------------------
from toolcv.utils.net.basev4 import BaseNet
ema = True
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.756
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.933
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.911

ema = False
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.774
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.928
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.autograd import Variable
from torchvision.ops import batched_nms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from math import ceil, floor, exp
import numpy as np

# from mmdet.models.losses import GaussianFocalLoss, L1Loss, FocalLoss, CrossEntropyLoss
# from mmdet.models.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target
from fvcore.nn import giou_loss, sigmoid_focal_loss, smooth_l1_loss
from toolcv.utils.tools.general import bbox_iou
from toolcv.utils.loss.loss import gaussian_focal_loss, labelsmooth_focal, labelsmooth, giou_loss, ciou_loss
from toolcv.utils.tools.utils import gaussian_radius as gaussian_radiusv1, gen_gaussian_target

from toolcv.utils.net.net import SPPv2
from toolcv.utils.net.basev3 import BaseNet
# from toolcv.utils.net.basev4 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, YOLOFNeck, YOLOFHeadSelf, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
from toolcv.utils.data.data import DataDefine
from toolcv.utils.data import transform as T
from toolcv.utils.tools.tools import glob_format, get_bboxes_DA as get_bboxes
from toolcv.utils.tools.tools2 import set_seed, get_device, get_params, get_optim_scheduler, load_weight

# from toolcv.api.define.utils.tools.tools import get_bboxesv3
# from toolcv.tools.utils import box_iou, wh_iou, xywh2x1y1x2y2
from toolcv.tools.anchor import gen_anchorv2  # ,kmean_gen_anchorv2,kmean_gen_anchor,gen_anchor

# from toolcv.api.define.utils.data.data import FruitsNutsDataset
from toolcv.utils.tools.confusionMatrix import ConfusionMatrix
from toolcv.utils.other.yolov5.metrics import ConfusionMatrix as ConfusionMatrix2, ap_per_class, process_batch
from toolcv.utils.other.custom import get_optim_scheduler as get_optim_schedulerv2

multiscale = False
# ignore = True
surrounding = True
radius = 1  # 3
sigma = 1
threds = 0.3

seed = 100
set_seed(seed)

# anchors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 缩放到0~1 [w,h]
# anchors = torch.tensor(anchors, dtype=torch.float32)
# anchors = torch.stack((anchors, anchors), -1)

# 通过聚类生成先验框
wsize = [0.09784091, 0.13189815, 0.16025642, 0.1969853, 0.2532143]
hsize = [0.12607142, 0.17412879, 0.22333333, 0.27459598, 0.3354762]
anchors_w = torch.tensor(wsize, dtype=torch.float32)
anchors_h = torch.tensor(hsize, dtype=torch.float32)
anchors = torch.stack((anchors_w, anchors_h), -1)  # 5x2
n_anchors = len(anchors)

ema = True
strides = 8
use_amp = True  # 推荐 先 True 再 False
accumulate = 2  # 推荐 先 >1 再 1
gradient_clip_val = 0.0 if use_amp else 1.0
lrf = 0.1
lr = 5e-4 if use_amp else 1e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
h, w = 416, 416
resize = (h, w)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
num_classes = len(classes)
device = get_device()
weight_path = 'weight.pth'
each_batch_scheduler = False  # 推荐 先 False 再 True
use_iouloss = False  # 推荐 先 False 再 True
save_path = 'output'
summary = SummaryWriter(save_path)
iou_threshold = 0.3
conf_threshold = 0.25
do_evaluate = True

if do_evaluate:
    cmtx_05 = ConfusionMatrix(0.5, classes)
    cmtx_075 = ConfusionMatrix(0.75, classes)

    cmatrix = ConfusionMatrix2(num_classes, conf_threshold, iou_threshold)
    stats = []
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

# 通过聚类生成先验框
# gen_anchorv2(FruitsNutsDataset(dir_data,classes),5)
# exit(0)
# w [0.09784091 0.13189815 0.16025642 0.1969853  0.2532143 ]
# h [0.12607142 0.17412879 0.22333333 0.27459598 0.3354762 ]

# -------------------data-------------------
data = DataDefine(dir_data, classes, batch_size, resize, 0)
data.set_transform()
data.get_dataloader()
train_dataloader = data.train_dataloader
val_dataloader = data.val_dataloader
test_transforms = data.val_transforms

# ----------------model --------------------------
backbone = Backbone('resnet18', True, stride=strides)
spp = SPPv2(backbone.out_channels, backbone.out_channels // 4)
neck = YOLOFNeck(backbone.out_channels)
head = YOLOFHeadSelf(backbone.out_channels // 4, n_anchors, 1, num_classes)
_initParmas(spp.modules(), mode='kaiming')
_initParmas(neck.modules(), mode='kaiming')
_initParmas(head.modules(), mode='normal')
model = nn.Sequential(backbone, spp, neck, head).to(device)
# load_weight(model, weight_path, "", device=device)
model.train()

# if each_batch_scheduler:
#     optimizer, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr,
#                                                weight_decay, "radam", "SineAnnealingLROnecev2", lrf, 0.6)
# else:
#     optimizer, scheduler = None, None

optimizer, scheduler = get_optim_schedulerv2(model, epochs, {"lr0": lr, "lrf": lrf, "weight_decay": weight_decay},
                                             "adamw", "one_cycle")


class YoloSelf(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.load_ckpt(weight_path)

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
                x1, y1 = gt_bbox[j, :2] * ratio
                x2, y2 = gt_bbox[j, 2:] * ratio

                gw, gh = gt_wh[j]
                iw = (self.anchors[:, 0] - gw).abs().argmin()
                ih = (self.anchors[:, 1] - gh).abs().argmin()

                # 缩放到heatmap
                ct = gt_centers[j] * ratio
                # ctx_int, cty_int = ct.int() # 默认为左上角
                ctx_int, cty_int = ct.round().int()  # 从 左上角，右上角，右下角，左下角 中选取 最合适的作为中心点
                ctx, cty = ct
                scale_box_w, scale_box_h = gt_wh[j]
                ind = gt_label[j]

                # 对应的先验anchor
                a_w, a_h = self.anchors[iw, 0], self.anchors[ih, 1]

                # if surrounding:
                #     radius = gaussian_radiusv1([scale_box_h * feat_h, scale_box_w * feat_w], 0.3)
                #     radius = max(0, int(radius))
                #     gen_gaussian_target(object_target[batch_id, 0], [ctx_int, cty_int], radius)
                # else:
                #     object_target[batch_id, 0, cty_int, ctx_int] = 1

                x1, x2 = max(ctx_int - radius, ceil(x1)), 1 + min(ctx_int + radius, floor(x2), feat_w - 1)
                y1, y2 = max(cty_int - radius, ceil(y1)), 1 + min(cty_int + radius, floor(y2), feat_h - 1)
                if x1 >= x2 or y1 >= y2: continue
                for _cy in range(y1, y2):
                    for _cx in range(x1, x2):
                        # object_target[batch_id, 0, _cy, _cx] = 1 / exp(
                        #     ((_cx - ctx_int) ** 2 + (_cy - cty_int) ** 2) / (2 * sigma ** 2))
                        value = 1 / exp(((_cx - ctx_int) ** 2 + (_cy - cty_int) ** 2) / (2 * sigma ** 2))
                        if value <= object_target[batch_id, 0, _cy, _cx]: continue
                        object_target[batch_id, 0, _cy, _cx] = value

                        cls_target[batch_id, ind, _cy, _cx] = 1
                        bbox_offset_target[batch_id, 0, _cy, _cx] = ctx - _cx
                        bbox_offset_target[batch_id, 1, _cy, _cx] = cty - _cy
                        bbox_offset_target[batch_id, 2, _cy, _cx] = (scale_box_w / a_w).log()
                        bbox_offset_target[batch_id, 3, _cy, _cx] = (scale_box_h / a_h).log()

                        wh_cls_target[batch_id, iw, 0, _cy, _cx] = 1
                        wh_cls_target[batch_id, ih, 1, _cy, _cx] = 1

        # avg_factor = max(1, object_target.eq(1).sum())  # 正样本总个数
        avg_factor = object_target.gt(0).sum()

        target_result = dict(
            object_target=object_target,
            cls_target=cls_target,
            bbox_offset_target=bbox_offset_target,
            wh_cls_target=wh_cls_target
        )

        return target_result, avg_factor

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        if multiscale:
            # 多尺寸训练
            # new_size = 32 * np.random.choice([9, 11, 13, 15, 17], p=[0.05, 0.1, 0.5, 0.2, 0.15])
            new_size = 32 * np.random.choice([13, 15, 17, 19], p=[0.4, 0.25, 0.2, 0.15])
            ratio = new_size / resize[0]
            if ratio != 1:
                imgs = F.interpolate(imgs, scale_factor=ratio, mode='bilinear')
                bs = len(targets)
                for i in range(bs):
                    targets[i]["boxes"] = targets[i]["boxes"] * ratio

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)
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
            # wh_offset_target_weight = (object_target > 0.5).float()
            # wh_offset_target_weight = object_target
            wh_offset_target_weight = object_target * (object_target > threds).float()
            """
            object_target * (object_target > 0.4).float()  # 5个点
            object_target * (object_target > 0.3).float()  # 9个点
            """

            # loss
            keep = object_target != -1
            loss_center_heatmap = gaussian_focal_loss(object_pred.sigmoid()[keep],
                                                      object_target[keep]).sum() / avg_factor

            if not use_iouloss:
                loss_boxes = (F.l1_loss(bbox_offset, bbox_offset_target, reduction="none") *
                              wh_offset_target_weight.expand_as(bbox_offset_target)).sum() / avg_factor
            else:
                keep = (object_target == 1).flatten(0)
                bbox_offset = bbox_offset.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
                bbox_offset[:, 2:] = bbox_offset[:, 2:].exp() + 2
                bbox_offset_target = bbox_offset_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)[keep]
                bbox_offset_target[:, 2:] = bbox_offset_target[:, 2:].exp() + 2
                # loss_boxes = giou_loss(bbox_offset, bbox_offset_target, "mean")
                loss_boxes = ciou_loss(bbox_offset, bbox_offset_target).mean()

            loss_cls = (F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none") *
                        wh_offset_target_weight.expand_as(cls_target)).sum() / avg_factor

            loss_wh = (F.cross_entropy(wh_cls, wh_cls_target, reduction="none") *
                       wh_offset_target_weight.expand_as(wh_cls_target)).sum() / avg_factor

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_cls=loss_cls,
            loss_wh=loss_wh,
            loss_boxes=loss_boxes)

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        in_shape = imgs.shape[-2:]

        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        scale_factors = kwargs['scale_factors']
        padding = kwargs['padding']
        with_nms = kwargs['with_nms']

        object_pred, cls_pred, bbox_offset, wh_cls = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        if use_iouloss:
            bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2
        boxes, scores, labels = get_bboxes(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                                           iou_threshold, conf_threshold, scale_factors, padding, in_shape)

        return dict(boxes=boxes, scores=scores, labels=labels)

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        iou_threshold = kwargs['iou_threshold']
        conf_threshold = kwargs['conf_threshold']
        with_nms = kwargs['with_nms']

        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        in_shape = imgs.shape[-2:]

        object_pred, cls_pred, bbox_offset, wh_cls = self.model(imgs)
        object_pred = object_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        # wh_cls = wh_cls.softmax(1)
        if use_iouloss:
            bbox_offset[:, :2] = bbox_offset[:, :2].sigmoid() * 2

        boxes, scores, labels = get_bboxes(self.anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                                           iou_threshold, conf_threshold, None, None, in_shape, False)

        pred = dict(boxes=boxes, scores=scores, labels=labels)

        if do_evaluate:
            cmtx_05.add_batch(pred, targets[0])
            cmtx_075.add_batch(pred, targets[0])

            predn = torch.cat((boxes, scores[..., None], labels[..., None]), -1).cpu()
            labelsn = torch.cat((targets[0]["labels"][..., None], targets[0]["boxes"]), -1).cpu()
            cmatrix.process_batch(predn, labelsn)

            tcls = targets[0]["labels"].cpu()
            if boxes.sum() == 0:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            else:
                if len(tcls) == 0:
                    correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)
                else:
                    correct = process_batch(predn.to(device), labelsn.to(device), iouv)

                stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        return pred


network = YoloSelf(**dict(model=model, num_classes=num_classes,
                          img_shape=resize, anchors=anchors,
                          strides=strides, epochs=epochs,
                          lr=lr, weight_decay=weight_decay, lrf=lrf,
                          warmup_iters=1000, gamma=0.5, optimizer=optimizer,
                          scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                          gradient_clip_val=gradient_clip_val, device=device,
                          criterion=None, train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader, each_batch_scheduler=each_batch_scheduler,
                          summary=summary, ema=ema))

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit(weight_path)
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path=weight_path, iou_threshold=iou_threshold,
                       conf_threshold=conf_threshold, with_nms=False, mode='coco'))

if do_evaluate:
    cmtx_05.plot_confusion_matrix(percent=True)
    cmtx_075.plot_confusion_matrix(percent=True)
    cmtx_075.save_confusion_matrix()
    cmatrix.plot(save_dir="./", names=classes)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=save_path,
                                                  names=dict(zip(range(num_classes), classes)))
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

# # -----------------------predict ---------------------------
network.predict(**dict(img_paths=glob_format(dir_data),
                       transform=test_transforms, device=device,
                       weight_path=weight_path,
                       save_path=save_path, visual=False,
                       with_nms=False, iou_threshold=0.3,
                       conf_threshold=0.1, method='pad'))
