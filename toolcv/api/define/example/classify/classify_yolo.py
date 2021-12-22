"""
自定义 backbone 网络
    使用目标检测数据 训练分类器 作为 后续 目标检测网络的 backbone
"""
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import time

from toolcv.api.pytorch_lightning.net import get_params
from toolcv.api.define.utils.model.basev2 import BaseNet
from toolcv.api.define.utils.model.net import Backbone, _initParmas
from toolcv.api.define.utils.data.data import FruitsNutsDataset, LoadDataloader
from toolcv.api.define.utils.data import augment as aug
import toolcv.data.augment.bboxAugv2 as baug
from toolcv.data.dataset import glob_format
from toolcv.api.define.utils.loss.loss import LabelSmoothingCrossEntropy
from toolcv.api.mobileNeXt.codebase.optim.radam import RAdam, PlainRAdam
from toolcv.api.mobileNeXt.codebase.scheduler.plateau_lr import PlateauLRScheduler
from toolcv.api.toolsmall.cls.tool import confusion_matrix, History
from toolcv.api.pytorch_lightning.utils import warmup_lr_scheduler

from toolcv.api.define.utils.model.detecte.backbone.backbone import RandomModel, Resnet50x

# from timm.optim.radam import RAdam,PlainRAdam
# from timm.loss.cross_entropy import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy
from timm.loss.jsd import JsdCrossEntropy

seed = 100
torch.manual_seed(seed)

anchors = None
strides = 32
use_amp = False
accumulate = 1
gradient_clip_val = 1.0
lrf = 0.1
lr = 3e-3
weight_decay = 5e-6
epochs = 50
batch_size = 4
h, w = 416, 416
# h, w = 224, 224
resize = (h, w)
dir_data = r"D:/data/fruitsNuts/"
classes = ['date', 'fig', 'hazelnut']
num_classes = len(classes)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.backends.cudnn.benchmark = True
# else:
#     device = torch.device('cpu')

# -------------------data-------------------
transforms = aug.Compose([
    aug.RandomHorizontalFlip(),
    aug.RandomChoice([baug.RandomBlur(), baug.RandomNoise(), aug.RandomColorJitter()]),
    baug.ResizeFixMinAndRandomCropV2AndPatch([300, 400, 500, 600], [0.6, 0.7, 0.8, 0.9]),
    # baug.ResizeFixMinAndRandomCropV2AndPatch([200, 256, 300], [0.7, 0.8, 0.9]),
    aug.Resize(*resize),
    aug.ToTensor(), aug.Normalize()])

test_transforms = aug.Compose([
    aug.ResizeMax(*resize), aug.Padding(),
    aug.ToTensor(), aug.Normalize()])

dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, False, None, 416, 416, 0, True)

val_dataset = FruitsNutsDataset(dir_data, classes, test_transforms, 0, False, None, 416, 416, 0, True)

# dataset.show(mode='cv2')
dataloader = LoadDataloader(dataset, val_dataset, 0.0, batch_size, {})
train_dataLoader, val_dataLoader = dataloader.train_dataloader(), dataloader.val_dataloader()

# ----------------model --------------------------
# model = Backbone('resnet18', True, stride=strides, freeze_at=0, num_classes=num_classes, do_cls=True).to(device)
model = Resnet50x(3, num_classes, freeze_at=0, stride=strides, do_cls=True).to(device)
# model = RandomModel(3, num_classes, freeze_at=0, stride=strides, do_cls=True).to(device)

"""
optimizer = None
scheduler=None
"""
optimizer = RAdam(get_params(model.modules(), lr, weight_decay, gamma=0.5), lr, weight_decay=weight_decay)
scheduler = PlateauLRScheduler(optimizer)
# scheduler=None
# """

history = History()


class Classify(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, batch, step):
        imgs, targets = batch
        # imgs = torch.stack(imgs, 0).to(self.device)
        imgs = torch.cat(imgs, 0).to(self.device)
        targets = torch.cat([target['labels'] for target in targets], 0).to(self.device)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(self.use_amp):
            outputs = self.model(imgs)

        # 使用 smooth label
        loss = LabelSmoothingCrossEntropy()(outputs, targets)
        preds = outputs.argmax(-1)

        return dict(preds=preds, targets=targets, loss=loss)

    def train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        nums = len(self.train_dataloader.dataset)
        total_loss = 0
        preds_list = []
        targets_list = []
        nb = len(self.train_dataloader)
        if epoch == 0 and self.warmup_iters > 0:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
            warmup_factor = 1.0 / self.warmup_iters
            warmup_iters = min(self.warmup_iters, nb - 1)
            lr_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        pbar = tqdm(enumerate(self.train_dataloader))
        for step, (imgs, targets) in pbar:
            # ni 统计从epoch0开始的所有batch数
            ni = step + nb * epoch  # number integrated batches (since train start)

            outputs = self.train_step((imgs, targets), step)
            preds_list.extend(outputs['preds'])
            targets_list.extend(outputs['targets'])
            losses = outputs['loss']
            if losses.isnan() or losses.isinf():
                print({k: v.item() for k, v in losses_dict.items()})
                exit(-1)

            if (ni + 1) % self.accumulate == 0:
                self.scaler.scale(losses).backward()
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

        acc = sum(np.array(preds_list) == np.array(targets_list)) / len(preds_list)
        loss = total_loss / nums
        learning_rate = self.get_current_lr()
        end = time.time()

        print("-" * 60)
        print("| epoch:%d train_loss:%.3f  train_acc:%.3f cost_time:%.3f lr:%.5f |" % (
            epoch, loss, acc, end - start, learning_rate))
        print("-" * 60)

        return dict(acc=acc, loss=loss)

    def fit(self, weight_path='weight.pth', **kwargs):
        self.configure_optimizers()
        self.load_weight(weight_path)
        for epoch in range(self.epochs):
            train_outputs = self.train_one_epoch(epoch)
            test_outputs = self.test_one_epoch(epoch)

            history.epoch.append(epoch)
            history.history['acc'].append(train_outputs['acc'])
            history.history['loss'].append(train_outputs['loss'])
            history.history['val_acc'].append(test_outputs['acc'])
            history.history['val_loss'].append(test_outputs['loss'])

            self.scheduler.step()
            self.save_weight(weight_path)

        history.show_final_history(history)

    @torch.no_grad()
    def test_step(self, batch, step):
        imgs, targets = batch
        imgs = torch.cat(imgs, 0).to(self.device)
        targets = torch.cat([target['labels'] for target in targets], 0).to(self.device)
        outputs = self.model(imgs)
        preds = outputs.argmax(-1)

        loss = LabelSmoothingCrossEntropy()(outputs, targets)

        return dict(preds=preds, targets=targets, loss=loss)

    @torch.no_grad()
    def test_one_epoch(self, epoch):
        self.model.eval()
        start = time.time()
        preds_list = []
        targets_list = []
        total_loss = 0
        nums = len(self.val_dataloader.dataset)
        pbar = tqdm(enumerate(self.val_dataloader))
        for step, (imgs, targets) in pbar:
            outputs = self.test_step((imgs, targets), step)
            preds = outputs['preds']
            targets = outputs['targets']
            loss = outputs['loss']
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            total_loss += loss.item()

        acc = sum(np.array(preds_list) == np.array(targets_list)) / len(preds_list)
        loss = total_loss / nums

        end = time.time()

        print("-" * 60)
        print("| epoch:%d val_loss:%.3f val_acc:%.3f cost_time:%.3f |" % (epoch, loss, acc, end - start))
        print("-" * 60)

        return dict(acc=acc, loss=loss)

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        imgs, targets = batch
        # imgs = torch.stack(imgs, 0).to(self.device)
        imgs = torch.cat(imgs, 0).to(self.device)
        targets = torch.cat([target['labels'] for target in targets], 0).to(self.device)
        outputs = self.model(imgs)
        preds = outputs.argmax(-1)

        return preds, targets

    @torch.no_grad()
    def evalute(self, **kwargs):
        weight_path = kwargs['weight_path']
        self.load_weight(weight_path)
        self.model.eval()

        preds_list = []
        targets_list = []
        pbar = tqdm(enumerate(self.val_dataloader))
        for step, (imgs, targets) in pbar:
            preds, targets = self.evalute_step((imgs, targets), step)
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

        confusion_matrix(targets_list, preds_list, classes)


network = Classify(**dict(model=model, num_classes=num_classes,
                          img_shape=resize, anchors=anchors,
                          strides=strides, epochs=epochs,
                          lr=lr, weight_decay=weight_decay, lrf=lrf,
                          warmup_iters=1000, gamma=0.5, optimizer=optimizer,
                          scheduler=scheduler, use_amp=use_amp, accumulate=accumulate,
                          gradient_clip_val=gradient_clip_val, device=device,
                          criterion=None, train_dataloader=train_dataLoader,
                          val_dataloader=val_dataLoader))

# network.statistical_parameter()

# -----------------------fit ---------------------------
network.fit()
# -----------------------eval ---------------------------
network.evalute(**dict(weight_path='weight.pth'))
# -----------------------predict ---------------------------
# network.predict(**dict(img_paths=glob_format(dir_data),
#                        transform=test_transforms, device=device,
#                        weight_path='weight.pth',
#                        save_path='output', visual=False,
#                        with_nms=False, iou_threshold=0.3,
#                        conf_threshold=0.2, method='pad'))
