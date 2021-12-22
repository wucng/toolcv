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
import numpy as np

from toolcv.api.pytorch_lightning.net import get_params  # , _initParmas
from toolcv.api.pytorch_lightning.utils import warmup_lr_scheduler  # , drawImg
from toolcv.tools.utils import drawImg

# from mmdet.core.evaluation.mean_ap import eval_map, print_map_summary
from toolcv.api.define.utils.tools.mean_ap import eval_map
from toolcv.api.define.utils.tools.coco_evalute.engine import evaluate_do


class _BaseNetV2(nn.Module):
    def __init__(self, model, num_classes, img_shape=(), anchors=None, strides=4, epochs=10, lr=5e-4, weight_decay=5e-5,
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

        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer is None:
            # optim
            if self.gamma > 0:
                params = get_params(self.model.modules(), self.lr, self.weight_decay, self.gamma)
            else:
                params = [param for param in self.model.parameters() if param.requires_grad]
            optim = torch.optim.AdamW(params, self.lr, weight_decay=self.weight_decay)

            self.optimizer = optim

        if self.scheduler is None:
            lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (
                    1 - self.lrf) + self.lrf  # cosine  last lr=lr*lrf
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

            self.scheduler = scheduler

        # return optim, scheduler

    def train_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        # targets = torch.stack(targets, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            center_heatmap_preds, wh_preds, offset_preds = self.model(imgs)  # heatmap 已经使用过 sigmoid

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

    def train_one_epoch(self, epoch):
        self.model.train()
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

            losses_dict = self.train_step((imgs, targets), step)
            losses = sum(losses_dict.values())
            if losses.isnan() or losses.isinf():
                print({k: v.item() for k, v in losses_dict.items()})
                exit(-1)

            # ----------使用梯度累加---------------------------
            losses = losses / self.accumulate
            self.scaler.scale(losses).backward()
            if (ni + 1) % self.accumulate == 0:
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
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
    def test_step(self, batch, step):
        pass

    @torch.no_grad()
    def test_one_epoch(self):
        self.model.eval()
        pass

    @torch.no_grad()
    def evalute_step(self, batch, step, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)

        pass

    @torch.no_grad()
    def evalute(self, weight_path='weight.pth', iou_threshold=0.3, conf_threshold=0.2, with_nms=False,
                mode='coco'):
        self.load_weight(weight_path)
        self.model.eval()

        if mode == 'coco':
            coco_evaluator = evaluate_do(self, iou_threshold, conf_threshold, with_nms)
        else:
            pbar = tqdm(enumerate(self.val_dataloader))
            annotations = []
            det_results = []
            for step, (imgs, targets) in pbar:
                annotations.append({'bboxes': targets[0]['boxes'].numpy(), 'labels': targets[0]['labels'].numpy()})
                boxes, scores, labels = self.evalute_step((imgs, targets), step, iou_threshold, conf_threshold, None,
                                                          None,
                                                          None, with_nms)
                tmp = [[]] * self.num_classes
                for box, label in zip(boxes, labels):
                    tmp[int(label.cpu().numpy())].append(box.cpu().numpy())
                for i, item in enumerate(tmp):
                    tmp[i] = np.stack(item, 0)
                det_results.append(tmp)

            mean_ap, eval_results = eval_map(det_results, annotations, nproc=1)

    @torch.no_grad()
    def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, in_shape, with_nms):
        center_heatmap_preds, wh_preds, offset_preds = self.model(img)

        det_results = get_bboxes(center_heatmap_preds, wh_preds, offset_preds, with_nms,
                                 iou_threshold, scale_factors, in_shape)

        boxes = det_results[0][0][..., :4]
        scores = det_results[0][0][..., 4]
        labels = det_results[0][1]

        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        return boxes, scores, labels

    @torch.no_grad()
    def predict(self, img_paths, transform, device, weight_path='weight.pth', save_path='output', visual=True,
                with_nms=False, iou_threshold=0.3, conf_threshold=0.2, method='resize'):
        self.load_weight(weight_path)
        self.model.eval()

        resize_h, resize_w = self.img_shape
        in_shape = (resize_h, resize_w)
        p_bar = tqdm(img_paths)
        for img_path in p_bar:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            img, _ = transform(img, None)
            img = img[None].to(device)

            if method == "resize":  # 直接resize 没有等比例缩放
                scale_w = resize_w / w
                scale_h = resize_h / h
                scale_factors = [scale_w, scale_h, scale_w, scale_h]
                padding = [0, 0, 0, 0]
            elif method == "low_right":  # 右边或者下边填充
                scale = min(in_shape) / max(w, h)
                scale_factors = [scale, scale, scale, scale]
                padding = [0, 0, 0, 0]
            else:  # 等比例缩放 + padding (两边padding)
                scale = resize_w / max(w, h)
                scale_factors = [scale, scale, scale, scale]
                pad_w = (resize_w - scale * w) // 2
                pad_h = (resize_h - scale * h) // 2
                padding = [pad_w, pad_h, pad_w, pad_h]

            boxes, scores, labels = self.pred_step(img, iou_threshold, conf_threshold, scale_factors, padding, in_shape,
                                                   with_nms)

            boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

            p_bar.set_description('detecte %d object' % (len(labels)))

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

    def fit(self, weight_path='weight.pth'):
        self.configure_optimizers()
        self.load_weight(weight_path)
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.scheduler.step()

            self.save_weight(weight_path)

    def get_current_lr(self):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        return learning_rate

    def load_weight(self, weight_path):
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, self.device))
            print(" ----------load weight successful ------------")

    def save_weight(self, weight_path):
        torch.save(self.model.state_dict(), weight_path)

    def save_onnx(self, onnx_path, args):
        torch.onnx.export(self.model, args, onnx_path, verbose=True, opset_version=11)

    def load_weights(self, state_dict, name):
        new_state_dict = {}
        for k, v in self.model.state_dict().items():
            new_k = name + k
            if new_k in state_dict and state_dict[new_k].numel() == v.numel():
                new_state_dict.update({k: state_dict[new_k]})
            else:
                new_state_dict.update({k: v})

        self.model.load_state_dict(new_state_dict)

        print(" ----------load weight successful ------------")

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

    def freeze_bn(self):
        # 默认冻结 BN中的参数 不更新
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in m.parameters():
                    parameter.requires_grad_(False)

    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad_(True)

    def statistical_parameter(self):
        train_param = 0
        notrain_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                train_param += param.numel()
            else:
                notrain_param += param.numel()
        print("train params:%d \nfreeze params:%d" % (train_param, notrain_param))
