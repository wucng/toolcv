"""
参考：https://github.com/ultralytics/yolov5/blob/master/train.py

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
from copy import deepcopy

# from toolcv.api.pytorch_lightning.net import get_params  # , _initParmas
# from toolcv.api.pytorch_lightning.utils import warmup_lr_scheduler  # , drawImg
# from toolcv.tools.utils import drawImg
# from toolcv.tools.vis import draw_rect, draw_mask, draw_segms, draw_keypoint

# from mmdet.core.evaluation.mean_ap import eval_map, print_map_summary
# from toolcv.api.define.utils.tools.mean_ap import eval_map
# from toolcv.api.define.utils.tools.coco_evalute.engine import evaluate_dov2 as evaluate_do

from toolcv.utils.tools.tools2 import get_params, warmup_lr_scheduler
from toolcv.utils.tools.vis import draw_rect, draw_mask, draw_segms, draw_keypoint
from toolcv.utils.coco_evalute.engine import evaluate_dov2 as evaluate_do

from toolcv.utils.other.yolov5.torch_utils import EarlyStopping, ModelEMA


class Common(nn.Module):
    def __init__(self):
        super().__init__()

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

    def train_step(self, batch, step):
        """
        imgs, targets = batch
        imgs = torch.stack(imgs, 0).to(self.device)
        targets = torch.stack(targets, 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)
        """

        raise ("error!!")

    def train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        nums = len(self.train_dataloader.dataset)
        total_loss = 0
        nb = len(self.train_dataloader)
        if not self.each_batch_scheduler:
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
            losses = losses / self.accumulate  # yolov5 并未 除以
            self.scaler.scale(losses).backward()
            if (ni + 1) % self.accumulate == 0:
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
                                                   self.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.ema:
                    self.ema.update(self.model)

            if not self.each_batch_scheduler:
                if epoch == 0 and warmup_iters > 0:
                    lr_scheduler.step()
            else:
                self.scheduler.step()

            total_loss += losses.item()

            desc = "epoch:%d step:%d loss:%.3f" % (epoch, step, losses.item())
            for k, v in losses_dict.items():
                desc += " %s:%.3f" % (k, v.item())
            pbar.set_description(desc)

            if hasattr(self, "summary"):
                self.summary.add_scalar("loss", losses.item(), ni)

        mean_loss = total_loss / nums
        learning_rate = self.get_current_lr()
        end = time.time()

        print("-" * 60)
        print("| epoch:%d train_loss:%.3f cost_time:%.3f lr:%.5f mem:%s|" % (
            epoch, mean_loss, end - start, learning_rate, self.get_model_mem()))
        print("-" * 60)

    @torch.no_grad()
    def test_step(self, batch, step):
        raise ('error')

    @torch.no_grad()
    def test_one_epoch(self, epoch):
        # self.model.eval()
        raise ('error')

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        """
        def evalute_step(self, batch, step, iou_threshold, conf_threshold,with_nms):

            imgs, targets = batch
            outputs = self.model(imgs)
            ....
            return dict(boxes=boxes, scores=scores, labels=labels)
        """

        raise ('error')

    @torch.no_grad()
    def evalute(self, **kwargs):
        """
        def evalute(self, weight_path='weight.pth', iou_threshold=0.3, conf_threshold=0.2, with_nms=False,
                mode='coco'):
        """
        weight_path = kwargs['weight_path']
        # iou_threshold = kwargs['iou_threshold']
        # conf_threshold = kwargs['conf_threshold']
        # with_nms = kwargs['with_nms']
        mode = kwargs['mode']

        # self.load_weight(weight_path)
        # self.load_weights(weight_path)
        self.model.eval()

        if mode == 'coco':
            coco_evaluator = evaluate_do(self, **kwargs)
        else:
            pbar = tqdm(enumerate(self.val_dataloader))
            annotations = []
            det_results = []
            for step, (imgs, targets) in pbar:
                annotations.append({'bboxes': targets[0]['boxes'].numpy(), 'labels': targets[0]['labels'].numpy()})
                preds = self.evalute_step((imgs, targets), step, **kwargs)
                boxes, scores, labels = preds['boxes'], preds['scores'], preds['labels']
                tmp = [[]] * self.num_classes
                for box, label in zip(boxes, labels):
                    tmp[int(label.cpu().numpy())].append(box.cpu().numpy())
                for i, item in enumerate(tmp):
                    tmp[i] = np.stack(item, 0)
                det_results.append(tmp)

            mean_ap, eval_results = eval_map(det_results, annotations, nproc=1)

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        """
        def pred_step(self, img, iou_threshold, conf_threshold, scale_factors, padding, with_nms):

            outputs = self.model(imgs)
            ....
            return dict(boxes=boxes, scores=scores, labels=labels)
        """

        raise ('error!!')

    @torch.no_grad()
    def predict(self, **kwargs):
        """
        def predict(self, img_paths, transform, device,
            weight_path='weight.pth',
            save_path='output', visual=True,
            with_nms=False, iou_threshold=0.3,
            conf_threshold=0.2, method='resize'):
        """
        img_paths = kwargs['img_paths']
        transform = kwargs['transform']
        device = kwargs['device']
        weight_path = kwargs['weight_path']
        save_path = kwargs['save_path']
        visual = kwargs['visual']
        method = kwargs['method']
        if 'draw' not in kwargs:
            draw = 'draw_rect'
        else:
            draw = kwargs['draw']

        if draw == 'draw_rect':
            draw = draw_rect
        elif draw == 'draw_mask':
            draw = draw_mask
        elif draw == 'draw_segms':
            draw = draw_segms
        elif draw == 'draw_keypoint':
            draw = draw_keypoint
        else:
            raise ('error!!')

        new_kwargs = kwargs.copy()
        # self.load_weight(weight_path)
        # self.load_weights(weight_path)
        self.model.eval()

        resize_h, resize_w = self.img_shape
        in_shape = (resize_h, resize_w)
        p_bar = tqdm(img_paths)
        for img_path in p_bar:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            # img, _ = transform(img, None)
            img = transform(img, None)
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

            new_kwargs.update(dict(scale_factors=scale_factors, padding=padding))
            target = self.pred_step(img, **new_kwargs)

            if isinstance(target, dict):
                boxes = target['boxes']
                boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
                boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)
                target['boxes'] = boxes

                p_bar.set_description('detecte %d object' % (len(boxes)))

            img = cv2.imread(img_path)
            img = draw(img, target)
            # img = drawImg(img, boxes, labels, scores)

            if visual:
                plt.imshow(img[..., ::-1])
                plt.show()
            else:
                if not os.path.exists(save_path): os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    def fit(self, weight_path='weight.pth', **kwargs):
        self.configure_optimizers()
        # self.load_weight(weight_path)
        # self.load_weights(weight_path)
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            if "exec_test" in kwargs:
                self.test_one_epoch(epoch)
            if not self.each_batch_scheduler:
                # 每个 epoch 使用 scheduler
                self.scheduler.step()

            # self.save_weight(weight_path)
            self.save_ckpt(weight_path, epoch)

        torch.cuda.empty_cache()

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

    def load_weights(self, state_dict, name=""):
        if isinstance(state_dict, str) and os.path.exists(state_dict):
            state_dict = torch.load(state_dict, map_location=self.device)

        if isinstance(state_dict, dict):
            new_state_dict = {}
            for k, v in self.model.state_dict().items():
                new_k = name + k
                if new_k in state_dict and state_dict[new_k].numel() == v.numel():
                    new_state_dict.update({k: state_dict[new_k]})
                else:
                    new_state_dict.update({k: v})

            self.model.load_state_dict(new_state_dict)

            print(" ----------load weight successful ------------")
        else:
            print(" ----------load weight fail ------------")

    def load_ckpt(self, ckpt_path):
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            # Optimizer
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                # best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            self.epochs += ckpt['epoch']
            # if self.epochs < self.start_epoch:
            #     # LOGGER.info(
            #     #     f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            #     self.epochs += ckpt['epoch']  # finetune additional epochs

            # model weight
            self.load_weights(ckpt['weight'].float().state_dict())

    def save_ckpt(self, ckpt_path="model.ckpt", epoch=0):
        ckpt = {'epoch': epoch,
                # 'best_fitness': best_fitness,
                # 'model': deepcopy(de_parallel(model)).half(),
                # "weight": self.model.state_dict(),
                "weight": deepcopy(self.model).half(),
                # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                # 'date': datetime.now().isoformat()
                }
        if self.ema:
            ckpt.update({
                # 'optimizer': self.optimizer.state_dict(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates})
        torch.save(ckpt, ckpt_path)

    def freeze_model(self, model=None):
        if model is None: model = self.model

        # for param in self.model.parameters():
        for param in model.parameters():
            param.requires_grad_(False)

    def freeze_bn(self, model=None):
        if model is None: model = self.model

        # 默认冻结 BN中的参数 不更新
        # for m in self.model.modules():
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in m.parameters():
                    parameter.requires_grad_(False)

    def unfreeze_model(self, model=None):
        if model is None: model = self.model

        # for param in self.model.parameters():
        for param in model.parameters():
            param.requires_grad_(True)

    def statistical_parameter(self, model=None):
        if model is None: model = self.model

        train_param = 0
        notrain_param = 0
        # for name, param in self.model.named_parameters():
        for name, param in model.named_parameters():
            if param.requires_grad:
                train_param += param.numel()
            else:
                notrain_param += param.numel()
        print("train params:%d \nfreeze params:%d" % (train_param, notrain_param))

    def flops(self, model=None, input=torch.randn(1, 3, 224, 224)):
        """
        FLOPS（即“每秒浮点运算次数”，“每秒峰值速度”）是“每秒所执行的浮点运算次数”
        （floating-point operations per second）的缩写。

        !pip install thop
        https://github.com/Lyken17/pytorch-OpCounter
        https://github.com/Swall0w/torchstat
        """
        from thop import profile
        from thop import clever_format

        if model is None: model = self.model
        flops, params = profile(model, inputs=(input.to(self.device),))
        flops, params = clever_format([flops, params], "%.3f")

        print('flops: ', flops, 'params: ', params, 'memory(about):', str(float(params[:-1]) * 4) + params[-1])

    def summary(self, model=None, input_size=(3, 224, 224)):
        """
        !pip install torchsummary
        from torchsummary import summary
        summary(model.cuda(), input_size=(3, 512, 512))
        """
        from torchsummary import summary

        if model is None: model = self.model
        summary(model, input_size=input_size, device=self.device)

    def print_size_of_model(self, model=None):
        if model is None: model = self.model
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        print('Size (MB):', size)
        os.remove('temp.p')

        return size

    def get_model_mem(self):
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        return mem


class BaseNet(Common):
    """
    model, num_classes, img_shape=(), anchors=None,
    strides=4, epochs=10, lr=5e-4, weight_decay=5e-5,
    lrf=0.1, warmup_iters=1000, gamma=0.5, optimizer=None,
    scheduler=None,use_amp=True, accumulate=4, gradient_clip_val=0.1,
    device='cpu', criterion=None, train_dataloader=None, val_dataloader=None
    """

    def __init__(self, **kwargs):
        super().__init__()
        assert 'use_amp' in kwargs
        # assert 'enable_gradscaler' in kwargs
        if 'enable_gradscaler' not in kwargs: kwargs['enable_gradscaler'] = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs['use_amp'] & kwargs['enable_gradscaler'])
        self.__dict__.update(kwargs)
        if "ema" in kwargs and kwargs["ema"]:
            self.ema = ModelEMA(self.model)
        else:
            self.ema = None

        self.start_epoch = 0


if __name__ == "__main__":
    network = BaseNet(**dict(model=model, num_classes=num_classes,
                             img_shape=resize, anchors=anchors,
                             strides=strides, epochs=epochs,
                             lr=lr, weight_decay=weight_decay, lrf=lrf,
                             warmup_iters=1000, gamma=0.5, optimizer=None,
                             scheduler=None, use_amp=use_amp, accumulate=accumulate,
                             gradient_clip_val=gradient_clip_val, device=device,
                             criterion=None, train_dataloader=train_dataLoader,
                             val_dataloader=val_dataLoader))
