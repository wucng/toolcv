"""
pip install mmcv-full
pip install openmim
mim install mmdet
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import os
import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from torchvision.ops import batched_nms
# from mmcv.ops import batched_nms
from mmdet.apis import init_detector
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.losses import GaussianFocalLoss, L1Loss
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)

from toolcv.api.pytorch_lightning.net import get_params, _initParmas
from toolcv.api.pytorch_lightning.utils import warmup_lr_scheduler  # , drawImg
from toolcv.tools.utils import drawImg


def load_weights(model, state_dict, name='', device='cpu'):
    if os.path.isfile(state_dict): state_dict = torch.load(state_dict, map_location=device)
    new_state_dict = {}
    for k, v in model.state_dict().items():
        new_k = name + k
        if new_k in state_dict and state_dict[new_k].numel() == v.numel():
            new_state_dict.update({k: state_dict[new_k]})
        else:
            new_state_dict.update({k: v})

    model.load_state_dict(new_state_dict)


def load_model(config, checkpoint=None, num_classes=20, device='cpu'):
    """
    Example:
        Args:
            config = 'configs/centernet/centernet_resnet18_140e_coco.py'
            checkpoint = 'checkpoints/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth'
            model = init_detector(config, checkpoint)
    """
    if num_classes > 0:
        cfg_options = dict(model=dict(bbox_head=dict(num_classes=num_classes)))
    else:
        cfg_options = None

    model = init_detector(config, checkpoint, device, cfg_options)

    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def freeze_bn(model):
    # 默认冻结 BN中的参数 不更新
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            for parameter in m.parameters():
                parameter.requires_grad_(False)


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad_(True)


def statistical_parameter(model):
    train_param = 0
    notrain_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param += param.numel()
        else:
            notrain_param += param.numel()
    print("train params:%d \nfreeze params:%d" % (train_param, notrain_param))


def get_bboxes(
        center_heatmap_preds,
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
        center_heatmap_preds[0],
        wh_preds[0],
        offset_preds[0],
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


def _bboxes_nms(bboxes, labels, cfg={'nms': 0.3, 'max_per_img': 100}):
    if labels.numel() == 0:
        return bboxes, labels

    out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                   cfg['nms'])
    out_labels = labels[keep]

    if len(out_bboxes) > 0:
        idx = torch.argsort(out_bboxes[:, -1], descending=True)
        idx = idx[:cfg['max_per_img']]
        out_bboxes = out_bboxes[idx]
        out_labels = out_labels[idx]

    return out_bboxes, out_labels


class _BaseNet(nn.Module):
    def __init__(self, num_classes, img_shape=(), anchors=None, strides=4, epochs=10, lr=5e-4, weight_decay=5e-5,
                 lrf=0.1, warmup_iters=1000, gamma=0.5, optimizer=None, scheduler=None,
                 use_amp=True, accumulate=4, gradient_clip_val=0.1,
                 device='cpu', criterion=None, train_dataloader=None, val_dataloader=None):

        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.anchors = anchors
        self.strides = strides
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lrf = lrf
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        self.use_amp = use_amp
        self.accumulate = accumulate
        self.gradient_clip_val = gradient_clip_val
        self.device = device
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.optimizer, self.scheduler = optimizer, scheduler

    def configure_optimizers(self, model):
        if self.optimizer is None:
            # optim
            if self.gamma > 0:
                params = get_params(model.modules(), self.lr, self.weight_decay, self.gamma)
            else:
                params = [param for param in model.parameters() if param.requires_grad]
            optim = torch.optim.AdamW(params, self.lr, weight_decay=self.weight_decay)

            self.optimizer = optim

        if self.scheduler is None:
            lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (
                    1 - self.lrf) + self.lrf  # cosine  last lr=lr*lrf
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

            self.scheduler = scheduler

        # return optim, scheduler

    def get_targets(self, imgs, targets, feat_shape):
        bs, _, feat_h, feat_w = feat_shape
        _, _, img_h, img_w = imgs.shape

        width_ratio = float(feat_w / img_w)  # 1/stride
        height_ratio = float(feat_h / img_h)  # 1/stride

        ratio = torch.tensor([[width_ratio, height_ratio]], dtype=imgs.dtype)

        # 转换成最终的target
        center_heatmap_target = imgs[0].new_zeros([bs, self.num_classes, feat_h, feat_w])
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

                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())  # 正样本总个数

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
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
            center_heatmap_preds, wh_preds, offset_preds = model(imgs)  # heatmap 已经使用过 sigmoid

        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        # target
        target_result, avg_factor = self.get_targets(imgs, targets, center_heatmap_pred.shape)
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # loss
        loss_center_heatmap = GaussianFocalLoss()(center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

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

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def train_one_epoch(self, model, epoch):
        model.train()
        start = time.time()
        nums = len(self.train_dataloader.dataset)
        total_loss = 0
        nb = len(self.train_dataloader)
        if epoch == 0 and self.warmup_iters > 0:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
            warmup_factor = 1.0 / self.warmup_iters
            warmup_iters = min(self.warmup_iters, nb - 1)
            lr_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        pbar = tqdm(enumerate(self.train_dataloader))
        for step, (imgs, targets) in pbar:
            # ni 统计从epoch0开始的所有batch数
            ni = step + nb * epoch  # number integrated batches (since train start)

            losses_dict = self.train_step(model, (imgs, targets), step)
            losses = sum(losses_dict.values())
            if losses.isnan() or losses.isinf(): continue

            if (ni + 1) % self.accumulate == 0:
                self.scaler.scale(losses).backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                                   self.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if epoch == 0 and warmup_iters > 0:
                lr_scheduler.step()

            total_loss += losses.item()

            desc = "epoch:%d step:%d loss:%.3f" % (epoch, step, losses.item())
            pbar.set_description(desc)

        mean_loss = total_loss / nums
        learning_rate = self.get_current_lr()
        end = time.time()

        print("-" * 60)
        print("| epoch:%d train_loss:%.3f cost_time:%.3f lr:%.5f |" % (
            epoch, mean_loss, end - start, learning_rate))
        print("-" * 60)

    @torch.no_grad()
    def test_one_epoch(self, model):
        model.eval()
        pass

    @torch.no_grad()
    def pred_step(self, model, img, iou_threshold, conf_threshold, scale_factors, in_shape, with_nms):
        center_heatmap_preds, wh_preds, offset_preds = model(img)

        det_results = get_bboxes(center_heatmap_preds, wh_preds, offset_preds, with_nms,
                                 iou_threshold, scale_factors, in_shape)

        boxes = det_results[0][0][..., :4]
        scores = det_results[0][0][..., 4]
        labels = det_results[0][1]

        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        return boxes, scores, labels

    @torch.no_grad()
    def predict(self, model, img_paths, transform, device, weight_path='weight.pth', save_path='output', visual=True,
                with_nms=False, iou_threshold=0.3, conf_threshold=0.2):
        self.load_weight(model, weight_path)
        model.eval()

        resize_h, resize_w = self.img_shape
        in_shape = (resize_h, resize_w)

        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            scale_w = resize_w / w
            scale_h = resize_h / h
            img, _ = transform(img, None)
            img = img[None].to(device)

            scale_factors = [[scale_w, scale_h, scale_w, scale_h]]

            boxes, scores, labels = self.pred_step(model, img, iou_threshold, conf_threshold, scale_factors, in_shape,
                                                   with_nms)

            img = cv2.imread(img_path)
            img = drawImg(img, boxes, labels, scores)
            if visual:
                # cv2.imshow('test',img)
                # cv2.waitKey(0)
                plt.imshow(img[..., ::-1])
                plt.show()
            else:
                if not os.path.exists(save_path): os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    def fit(self, model, weight_path='weight.pth'):
        self.configure_optimizers(model)
        self.load_weight(model, weight_path)
        for epoch in range(self.epochs):
            self.train_one_epoch(model, epoch)
            self.scheduler.step()

            self.save_weight(model, weight_path)

    def get_current_lr(self):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        return learning_rate

    def load_weight(self, model, weight_path):
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, self.device))
            print(" ----------load weight successful ------------")

    def save_weight(self, model, weight_path):
        torch.save(model.state_dict(), weight_path)

    def save_onnx(self, model, onnx_path, args):
        torch.onnx.export(model, args, onnx_path, verbose=True, opset_version=11)

    def load_weights(self, model, state_dict, name):
        new_state_dict = {}
        for k, v in model.state_dict().items():
            new_k = name + k
            if new_k in state_dict and state_dict[new_k].numel() == v.numel():
                new_state_dict.update({k: state_dict[new_k]})
            else:
                new_state_dict.update({k: v})

        model.load_state_dict(new_state_dict)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def freeze_bn(self, model):
        # 默认冻结 BN中的参数 不更新
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in m.parameters():
                    parameter.requires_grad_(False)

    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad_(True)

    def statistical_parameter(self, model):
        train_param = 0
        notrain_param = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                train_param += param.numel()
            else:
                notrain_param += param.numel()
        print("train params:%d \nfreeze params:%d" % (train_param, notrain_param))


class CenterNet(_BaseNet):
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

    def load_model(self, config, checkpoint, num_classes, device):
        try:
            model = load_model(config, checkpoint, num_classes, device)  # 类别对不上会报错
        except Exception as e:
            print(e)
            model = load_model(config, None, num_classes, device)
            if checkpoint is not None:
                state_dict = torch.load(checkpoint, device)
                load_weights(model.backbone, state_dict, 'backbone.')
                load_weights(model.neck, state_dict, 'neck.')
                load_weights(model.bbox_head, state_dict, 'bbox_head.')
                print("--------load weights successful-----------")

        # freeze
        freeze_model(model)
        unfreeze_model(model.bbox_head)
        unfreeze_model(model.neck)
        freeze_bn(model)
        statistical_parameter(model)

        model = nn.Sequential(model.backbone, model.neck, model.bbox_head)

        """
        x=torch.rand([1,3,512,512]).to(self.device)
        a = model.backbone(x) # (tensor[1,64,128,128],tensor[1,128,64,64],tensor[1,256,32,32],tensor[1,512,16,16])
        b=model.neck(a) # (tensor[1,64,128,128],)
        c=model.bbox_head(b) # ([tensor[1,3,128,128]], # headmap
                                [tensor[1,2,128,128]], # wh 
                                [tensor[1,2,128,128]]) # offset
        """

        return model

    def forward(self, x):
        return self.model(x)


def test_centernet(config, checkpoint=None, num_classes=20, device='cpu'):
    x = torch.rand([1, 3, 512, 512]).to('cuda:0')
    model = load_model(config, checkpoint, num_classes, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters())

    backbone = model.backbone
    neck = model.neck
    bbox_head = model.bbox_head

    # backbone.layer4.training = True
    # neck.training = True
    # bbox_head.training = True

    freeze_model(model.backbone)
    unfreeze_model(model.backbone.layer4)
    freeze_bn(model)
    statistical_parameter(model)

    out_b = backbone(x)
    out_n = neck(out_b)
    out_h = bbox_head(out_n)

    loss = F.binary_cross_entropy(out_h[0][0], torch.zeros_like(out_h[0][0]))
    loss.backward()
    opt.step()
    opt.zero_grad()


"""
torch.onnx.export(model,
                  (torch.rand([1, 2, 3, 224, 224]),
                  [[{'img_shape_for_onnx': torch.tensor((2, 3, 224, 224)),
                  'scale_factor': torch.tensor(1),
                  'border': torch.rand([2,5]),
                  'batch_input_shape': torch.tensor((224, 224))
                  }]]),
                  'model.onnx', verbose=True, opset_version=11)
"""

if __name__ == "__main__":
    config = r"D:\zyy\git\mmdetection\configs\centernet\centernet_resnet18_140e_coco.py"
    test_centernet(config, device='cuda:0')
