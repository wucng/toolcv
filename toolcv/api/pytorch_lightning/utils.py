import torch
from torch import nn
import torch.nn.functional as F
import time
import math
import numpy as np
import os
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from fvcore.nn import sigmoid_focal_loss, giou_loss, smooth_l1_loss
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import loggers as pl_loggers

from toolcv.cls.tools import History
from toolcv.tools.utils import drawImg, _nms, xywh2x1y1x2y2, x1y1x2y22xywh, batched_nms, \
    x1y1x2y22xywh_np, xywh2x1y1x2y2_np, box_iou_np, box_iou


def set_seed(seed=100):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)


def load_callbacks(earlystopping=True, monitor='val_acc', lr_scheduler=True):
    """monitor='val_acc' 'val_loss' """
    callbacks = []
    if earlystopping:
        callbacks.append(plc.EarlyStopping(
            monitor=monitor,
            mode='max',
            patience=10,
            min_delta=0.001
        ))

    callbacks.append(plc.ModelCheckpoint(
        monitor=monitor,
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    return callbacks


def load_logger(save_dir='logs/', mode="tensorboard"):
    if mode == "tensorboard":
        # Or use the same format as others
        logger = pl_loggers.TensorBoardLogger(save_dir=save_dir)
    else:
        # One Logger
        logger = pl_loggers.CometLogger(save_dir=save_dir)

    return logger


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(1.0, x / warmup_iters))
    # """


# ------------------- 以下是 pytorch 原本的方法------------------------------------

def trainV0(model, optimizer, criterion, dataloader, device, epoch, print_step=50, gradient_clip_val=0.1):
    start = time.time()
    model.train()
    total_loss = 0
    total_acc = 0
    nums = len(dataloader.dataset)
    for step, (x, y) in enumerate(dataloader):
        # x = x.view(x.size(0), -1)
        out = model(x.to(device))
        if criterion is None:
            loss = F.cross_entropy(out, y.to(device), reduction='sum')
        else:
            loss = criterion(out, y.to(device))
        total_loss += loss.item()
        bs = out.size(0)
        loss = loss / bs

        loss.backward()
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                           gradient_clip_val)
        optimizer.step()
        optimizer.zero_grad()

        trues = (out.argmax(1).cpu() == y).sum().item()
        total_acc += trues

        if step % print_step == 0:
            print("epoch:%d step:%d acc:%.3f loss:%.3f" % (epoch, step, trues / bs, loss.item()))

    mean_acc, mean_loss = total_acc / nums, total_loss / nums

    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    end = time.time()
    print("-" * 60)
    print(
        "| epoch:%d train_acc:%.3f train_loss:%.3f cost_time:%.3f lr:%.5f |" % (
            epoch, mean_acc, mean_loss, end - start, learning_rate))
    print("-" * 60)

    return mean_acc, mean_loss


def trainV1(model, optimizer, criterion, dataloader, device, epoch, print_step=50, accumulate=4, gradient_clip_val=0.1):
    start = time.time()
    model.train()
    total_loss = 0
    total_acc = 0
    losses = 0
    nums = len(dataloader.dataset)

    if epoch == 0:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for step, (x, y) in enumerate(dataloader):
        # x = x.view(x.size(0), -1)
        out = model(x.to(device))
        if criterion is None:
            loss = F.cross_entropy(out, y.to(device), reduction='sum')
        else:
            loss = criterion(out, y.to(device))
        total_loss += loss.item()
        bs = out.size(0)
        loss = loss / bs

        losses = losses + loss
        if (step + 1) % accumulate == 0:
            losses = losses / accumulate
            losses.backward()
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                               gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            losses = 0

        trues = (out.argmax(1).cpu() == y).sum().item()
        total_acc += trues

        if step % print_step == 0:
            print("epoch:%d step:%d acc:%.3f loss:%.3f" % (epoch, step, trues / bs, loss.item()))

        if epoch == 0:
            lr_scheduler.step()

    mean_acc, mean_loss = total_acc / nums, total_loss / nums

    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    end = time.time()
    print("-" * 60)
    print(
        "| epoch:%d train_acc:%.3f train_loss:%.3f cost_time:%.3f lr:%.5f |" % (
            epoch, mean_acc, mean_loss, end - start, learning_rate))
    print("-" * 60)

    return mean_acc, mean_loss


class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0, reduction='none'):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction=reduction)
            # self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


default_scaler = torch.cuda.amp.GradScaler(enabled=True)


def trainV2(model, optimizer, criterion, dataloader, device, epoch, print_step=50, scaler=None, use_amp=True,
            accumulate=4, accumulate_mode=0, gradient_clip_val=0.1, reduction='none', filter_loss=True,
            warmup_iters=1000, label_smoothing=True):
    """
    1、启用warmup训练方式，可理解为热身训练
    2、使用apex混合精度训练（加速）
    3、使用 accumulate加速 原本每个batch更新一次梯度，现在 每 batch*accumulate 更新一次梯度
    """
    # accumulate = max(round(64 / batch_size), 1)
    if scaler is None and use_amp:
        scaler = default_scaler

    start = time.time()
    model.train()
    total_loss = 0
    total_acc = 0
    losses = 0
    nums = len(dataloader.dataset)
    nb = len(dataloader)
    if epoch == 0 and warmup_iters > 0:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / warmup_iters
        warmup_iters = min(warmup_iters, len(dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for step, (x, y) in enumerate(dataloader):
        # ni 统计从epoch0开始的所有batch数
        ni = step + nb * epoch  # number integrated batches (since train start)

        # x = x.view(x.size(0), -1)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(use_amp):
            out = model(x.to(device))
            if criterion is None:
                if label_smoothing:
                    loss = NMTCritierion(np.random.uniform(0.1, 0.5), reduction)(out, y.to(device))
                else:
                    # loss = F.cross_entropy(out, y.to(device), reduction='sum')
                    loss = F.cross_entropy(out, y.to(device), reduction=reduction)
            else:
                loss = criterion(out, y.to(device))

        if filter_loss and reduction == 'none' and step % 5 == 0:
            # 过滤loss (过滤掉 loss异常大的 估计是异常值)
            keep = torch.bitwise_and(loss > 0, loss < 0.95 * loss.max().item()).detach().float()
            loss = loss * keep
            loss = loss.sum()
            bs = keep.sum()
            # loss = (loss-2).clamp(min=0.0)
            # bs = out.size(0)
            # loss = loss.sum()
        else:
            if reduction == 'none': loss = loss.sum()
            bs = out.size(0)

        total_loss += loss.item()
        if reduction == 'none' or reduction == 'sum': loss = loss / bs

        if accumulate_mode == 0:
            losses = losses + loss
            if scaler is None or not use_amp:
                if (step + 1) % accumulate == 0:
                    losses = losses / accumulate
                    losses.backward()
                    if gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                                       gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()

                    losses = 0
            elif scaler is not None and use_amp:
                if (step + 1) % accumulate == 0:
                    losses = losses / accumulate
                    scaler.scale(losses).backward()
                    # 使用 混合精度训练 ；再使用梯度裁剪 效果很差
                    # if gradient_clip_val > 0:
                    #     torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                    #                                    gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    losses = 0
        else:
            if scaler is None or not use_amp:
                loss.backward()
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                                   gradient_clip_val)
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            elif scaler is not None and use_amp:
                scaler.scale(loss).backward()
                # 使用 混合精度训练 ；再使用梯度裁剪 效果很差
                # if gradient_clip_val > 0:
                #     torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                #                                    gradient_clip_val)
                # optimize
                # 每训练64张图片更新一次权重
                if ni % accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        trues = (out.argmax(1).cpu() == y).sum().item()
        total_acc += trues

        if step % print_step == 0:
            print("epoch:%d step:%d acc:%.3f loss:%.3f" % (epoch, step, trues / len(y), loss.item()))

        if epoch == 0 and warmup_iters > 0:
            lr_scheduler.step()

    mean_acc, mean_loss = total_acc / nums, total_loss / nums

    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    end = time.time()
    print("-" * 60)
    print(
        "| epoch:%d train_acc:%.3f train_loss:%.3f cost_time:%.3f lr:%.5f |" % (
            epoch, mean_acc, mean_loss, end - start, learning_rate))
    print("-" * 60)

    return mean_acc, mean_loss


@torch.no_grad()
def evalute(model, criterion, dataloader, device, epoch):
    model.eval()
    preds = []
    trues = []
    total_loss = 0
    for (x, y) in dataloader:
        # x = x.view(x.size(0), -1)
        out = model(x.to(device))
        if criterion is not None:
            loss = criterion(out, y.to(device))
        else:
            loss = F.cross_entropy(out, y.to(device), reduction='sum')
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    mean_loss = total_loss / len(preds)
    mean_acc = sum(np.array(preds) == np.array(trues)) / len(preds)

    print("-" * 60)
    print("| epoch:%d val_acc:%.3f val_loss:%.3f |" % (epoch, mean_acc, mean_loss))
    print("-" * 60)
    return mean_acc, mean_loss


@torch.no_grad()
def test(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    for (x, y) in dataloader:
        # x = x.view(x.size(0), -1)
        out = model(x.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    mean_acc = sum(np.array(preds) == np.array(trues)) / len(preds)

    print("-" * 60)
    print("| test_acc:%.3f |" % (mean_acc))
    print("-" * 60)
    return mean_acc


def fit(model, optimizer, trainer=None, evaluter=None, criterion=None, checkpoint_path=None, scheduler=None,
        train_dataloader=None,
        val_dataloader=None, epochs=5, print_step=100, batch_size=32, device="cpu", use_amp=True, accumulate=None,
        accumulate_mode=0, gradient_clip_val=0.1, reduction='none', filter_loss=True, warmup_iters=1000, mode='v2',
        draw=True, label_smoothing=True):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, device))
        print("-" * 60)
        print("| load weight successful |")
        print("-" * 60)

    if draw:
        history = History()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None

    if accumulate is None:
        accumulate = max(round(64 / batch_size), 1)

    continue_epoch = 0
    continue_epoch2 = 0
    tmp_acc = 1.0
    best_acc = 0.0
    save_path = ""
    for epoch in range(epochs):
        if trainer is not None:
            train_acc, train_loss = trainer
        else:
            if mode == 'v0':
                train_acc, train_loss = trainV0(model, optimizer, criterion, train_dataloader, device, epoch,
                                                print_step,
                                                gradient_clip_val)
            elif mode == 'v1':
                train_acc, train_loss = trainV1(model, optimizer, criterion, train_dataloader, device, epoch,
                                                print_step,
                                                accumulate, gradient_clip_val)
            else:
                train_acc, train_loss = trainV2(model, optimizer, criterion, train_dataloader, device, epoch,
                                                print_step,
                                                scaler, use_amp, accumulate, accumulate_mode, gradient_clip_val,
                                                reduction,
                                                filter_loss, warmup_iters, label_smoothing)

        if evaluter is not None:
            val_acc, val_loss = evaluter
        else:
            val_acc, val_loss = evalute(model, criterion, val_dataloader, device, epoch)

        if scheduler is not None:
            if val_acc < tmp_acc:
                tmp_acc = val_acc
                scheduler.step()

        if val_acc < tmp_acc:
            continue_epoch += 1
            continue_epoch2 += 1
        else:
            continue_epoch = 0
            continue_epoch2 = 0
        if continue_epoch > 3:
            continue_epoch = 0
            model.load_state_dict(torch.load(save_path, device))
            print("-" * 60)
            print("| load weight successful |")
            print("-" * 60)

        if continue_epoch2 > 5:
            break

        if val_acc > best_acc:
            best_acc = val_acc
            if os.path.exists(save_path): os.remove(save_path)
            save_path = "best-epoch-{:d}-acc-{:.3f}-loss-{:.3f}.pth".format(epoch, val_acc, val_loss)
            torch.save(model.state_dict(), save_path)

            # learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            # torch.save({"model":model.state_dict(),'optimizer':optimizer.state_dict(),
            #             "learning_rate":learning_rate,"epoch":epoch}, save_path)

        if draw:
            history.epoch.append(epoch)
            history.history["loss"].append(train_loss)
            history.history["acc"].append(train_acc)
            history.history["val_loss"].append(val_loss)
            history.history["val_acc"].append(val_acc)

            history.show_dynamic_history()

    torch.save(model.state_dict(), 'last.pth')

    if draw:
        history.show_final_history()


# ------------------detecte-----------------------------------------

def training_step_fcos(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    if self.multiscale:
        img = img[0]
        target = target[0]
    output = self(img)

    # conf
    conf_keep = target[..., 0] > self.thres
    output_conf = output[conf_keep]
    target_conf = target[conf_keep]

    # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
    conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
    if self.reduction == "sum":
        conf_loss = conf_loss.sum()
    elif self.reduction == "mean":
        conf_loss = conf_loss.sum() / conf_keep.sum()

    # box
    if self.mode == "sigmoid":
        boxes_loss = self.boxes_weight*F.mse_loss(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],
                                reduction=self.reduction)
    elif self.mode == "exp":
        boxes_loss = self.boxes_weight*F.mse_loss(output_conf[..., 1:5].exp(), target_conf[..., 1:5],
                                reduction=self.reduction)

    # class
    # cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],reduction=self.reduction)
    cls_loss = F.binary_cross_entropy_with_logits(output_conf[..., 5:], target_conf[..., 5:],
                                                  reduction=self.reduction)
    """
    else:
        # box
        if self.mode == "sigmoid":
            boxes_loss = self.boxes_weight * F.mse_loss(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],
                                                        reduction="none")
        elif self.mode == "exp":
            boxes_loss = self.boxes_weight * F.mse_loss(output_conf[..., 1:5].exp(), target_conf[..., 1:5],
                                                        reduction="none")

        boxes_loss = boxes_loss * target_conf[..., [0]].detach()
        if self.reduction == "sum":
            boxes_loss = boxes_loss.sum()
        elif self.reduction == "mean":
            boxes_loss = boxes_loss.sum() / target_conf.size(0)

        # class
        # cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],reduction=self.reduction)
        cls_loss = F.binary_cross_entropy_with_logits(output_conf[..., 5:], target_conf[..., 5:], reduction='none')
        cls_loss = cls_loss * target_conf[..., [0]].detach()
        if self.reduction == "sum":
            cls_loss = cls_loss.sum()
        elif self.reduction == "mean":
            cls_loss = cls_loss.sum() / target_conf.size(0)
    """

    loss = conf_loss + boxes_loss + cls_loss

    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_fcos(model, img_paths, transform, resize, device, visual=True, strides=416,
                 fix_resize=False, mode='exp', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    fh, fw = int(np.ceil(resize / strides)), int(np.ceil(resize / strides))
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img, _ = transform(img, {})

        output = model(img[None].to(device))

        conf = torch.sigmoid(output[..., 0])
        keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

        fwhwh = torch.tensor([[fw, fh, fw, fh]], dtype=torch.float32, device=device)
        if mode == "sigmoid":
            boxes = output[..., 1:5].sigmoid() * fwhwh
        elif mode == "exp":
            boxes = output[..., 1:5].exp()

        shift_x = np.arange(0, fw)
        shift_y = np.arange(0, fh)
        X, Y = np.meshgrid(shift_x, shift_y)
        xy = np.stack((X, Y), -1)
        # boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        # boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

        # x1y1x2y2
        tmp = torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        boxes[..., :2] = tmp - boxes[..., :2]
        boxes[..., 2:] += tmp
        boxes /= fwhwh  # 0~1

        scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

        conf = conf * scores
        # keep = conf > conf_threshold
        keep = torch.bitwise_and(conf > conf_threshold, keep)

        if keep.sum() == 0:
            print("no object detecte")
            continue
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]
        conf = conf[keep]

        # nms
        # boxes = xywh2x1y1x2y2(boxes)
        # keep = batched_nms(boxes, scores, labels, iou_threshold)
        keep = batched_nms(boxes, conf, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))


def training_step_yolov1(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    if self.multiscale:
        img = img[0]
        target = target[0]
    output = self(img)

    # conf
    if self.mode in ['fcosv2', 'centernetv2']:
        conf_keep = target[..., 0] > self.thres
    else:
        conf_keep = target[..., 0] == 1

    output_conf = output[conf_keep]
    target_conf = target[conf_keep]

    conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
    if self.reduction == "sum":
        conf_loss = conf_loss.sum()
    elif self.reduction == "mean":
        conf_loss = conf_loss.sum() / conf_keep.sum()

    if 'log' not in self.box_norm and self.mode not in ['fcosv2', 'centernetv2']:
        # boxes_loss = self.boxes_weight * F.binary_cross_entropy(torch.sigmoid(output_conf[..., 1:5]),
        #                                                         target_conf[..., 1:5], reduction=self.reduction)
        boxes_loss = self.boxes_weight * F.binary_cross_entropy_with_logits(output_conf[..., 1:5],
                                                                            target_conf[..., 1:5],
                                                                            reduction=self.reduction)
    else:
        boxes_loss = self.boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5],
                                                          reduction=self.reduction)

    # class
    # cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
    #                                   reduction=self.reduction)
    cls_loss = F.binary_cross_entropy_with_logits(output_conf[..., 5:], target_conf[..., 5:],
                                                  reduction=self.reduction)

    loss = conf_loss + boxes_loss + cls_loss
    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_yolov1(model, img_paths, transform, resize, device, visual=True, strides=16,
                   fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3,
                   box_norm="log"):
    fh, fw = int(np.ceil(resize / strides)), int(np.ceil(resize / strides))
    model.eval()
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        img, _ = transform(img, {})

        output = model(img[None].to(device))

        conf = torch.sigmoid(output[..., 0])
        keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

        if 'log' not in box_norm and mode not in ['fcosv2', 'centernetv2']:
            boxes = torch.sigmoid(output[..., 1:5])
        else:
            boxes = output[..., 1:5]

        shift_x = np.arange(0, fw)
        shift_y = np.arange(0, fh)
        X, Y = np.meshgrid(shift_x, shift_y)
        xy = np.stack((X, Y), -1)
        boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

        if box_norm == "log":
            boxes[..., 2:] = boxes[..., 2:].exp()
        elif box_norm == "logv2":
            boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor((1 / fw, 1 / fh), dtype=torch.float32, device=device)
        elif box_norm == "sqrt":
            boxes[..., 2:] = boxes[..., 2:] ** 2

        scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

        conf = conf * scores
        # keep = conf > conf_threshold
        keep = torch.bitwise_and(conf > conf_threshold, keep)

        if keep.sum() == 0:
            print("no object detecte")
            continue
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]
        conf = conf[keep]

        # nms
        boxes = xywh2x1y1x2y2(boxes)
        # keep = batched_nms(boxes, scores, labels, iou_threshold)
        keep = batched_nms(boxes, conf, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))


def training_step_yolov2(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    if self.multiscale:
        img = img[0]
        target = target[0]
    output = self(img)

    # conf
    if self.mode in ['fcosv2', 'centernetv2']:
        conf_keep = target[..., 0] > thres
    else:
        conf_keep = target[..., 0] == 1
    # output_conf = torch.sigmoid(output[conf_keep])
    output_conf = output[conf_keep]
    target_conf = target[conf_keep]

    no_ignore = target[..., 0] != -1

    # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
    conf_loss = sigmoid_focal_loss(output[..., 0][no_ignore], target[..., 0][no_ignore], 0.4, 2, "none")
    if self.reduction == "sum":
        conf_loss = conf_loss.sum()
    elif self.reduction == "mean":
        conf_loss = conf_loss.sum() / conf_keep.sum()

    # box
    boxes_loss = self.boxes_weight * F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=self.reduction)
    # boxes_loss = boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=self.reduction)

    # class
    # cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:], reduction=self.reduction)
    cls_loss = F.binary_cross_entropy_with_logits(output_conf[..., 5:], target_conf[..., 5:], reduction=self.reduction)

    loss = conf_loss + boxes_loss + cls_loss

    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_yolov2(anchor, model, img_paths, transform, resize, device, visual=True, strides=16,
                   fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    # anchor = np.array(anchor)
    fh, fw = int(np.ceil(resize / strides)), int(np.ceil(resize / strides))
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img, _ = transform(img, {})

        output = model(img[None].to(device))

        conf = torch.sigmoid(output[..., 0])
        keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

        boxes = output[..., 1:5]

        boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor(anchor, dtype=torch.float32, device=device)[None, None]

        shift_x = np.arange(0, fw)
        shift_y = np.arange(0, fh)
        X, Y = np.meshgrid(shift_x, shift_y)
        xy = np.stack((X, Y), -1)
        boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

        scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

        conf = conf * scores
        # keep = conf > conf_threshold
        keep = torch.bitwise_and(conf > conf_threshold, keep)

        if keep.sum() == 0:
            print("no object detecte")
            continue
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]
        conf = conf[keep]

        # nms
        boxes = xywh2x1y1x2y2(boxes)
        # keep = batched_nms(boxes, scores, labels, iou_threshold)
        keep = batched_nms(boxes, conf, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))


def training_step_yolov3(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    if self.multiscale:
        img = img[0]
        target = target[0]
    output = self(img)

    x8, x16, x32 = output
    bs = x8.size(0)
    c = x8.size(-1)
    output = torch.cat((x8.contiguous().view(bs, -1, c), x16.contiguous().view(bs, -1, c),
                        x32.contiguous().view(bs, -1, c)), 1)

    # conf
    if self.mode in ['fcosv2', 'centernetv2']:
        conf_keep = target[..., 0] > thres
    else:
        conf_keep = target[..., 0] == 1
    # output_conf = torch.sigmoid(output[conf_keep])
    output_conf = output[conf_keep]
    target_conf = target[conf_keep]

    no_ignore = target[..., 0] != -1

    # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
    conf_loss = sigmoid_focal_loss(output[..., 0][no_ignore], target[..., 0][no_ignore], 0.4, 2, "none")
    if self.reduction == "sum":
        conf_loss = conf_loss.sum()
    elif self.reduction == "mean":
        conf_loss = conf_loss.sum() / conf_keep.sum()

    # box
    boxes_loss = self.boxes_weight * F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=self.reduction)
    # boxes_loss = boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=self.reduction)

    # class
    # cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:], reduction=self.reduction)
    cls_loss = F.binary_cross_entropy_with_logits(output_conf[..., 5:], target_conf[..., 5:], reduction=self.reduction)

    loss = conf_loss + boxes_loss + cls_loss

    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_yolov3(anchors, model, img_paths, transform, resize, device, visual=True, strides=[8, 16, 32],
                   fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    # anchors = np.array(anchors)
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img, _ = transform(img, {})

        outputs = model(img[None].to(device))

        _boxes = []
        _conf = []
        _scores = []
        _labels = []
        for i, stride in enumerate(strides):
            output = outputs[i]
            anchor = anchors[i]
            fw, fh = resize // stride, resize // stride

            conf = torch.sigmoid(output[..., 0])
            keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

            boxes = output[..., 1:5]

            boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor(anchor, dtype=torch.float32, device=device)[None, None]

            shift_x = np.arange(0, fw)
            shift_y = np.arange(0, fh)
            X, Y = np.meshgrid(shift_x, shift_y)
            xy = np.stack((X, Y), -1)
            boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
            boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

            scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

            conf = conf * scores
            # keep = conf > conf_threshold
            keep = torch.bitwise_and(conf > conf_threshold, keep)

            if keep.sum() == 0:
                print("no object detecte")
                continue
            boxes = boxes[keep]
            scores, labels = scores[keep], labels[keep]
            conf = conf[keep]

            _boxes.append(boxes)
            _scores.append(scores)
            _labels.append(labels)
            _conf.append(conf)

        if len(_conf)==0:
            print("no object detecte")
            continue

        boxes = torch.cat(_boxes, 0)
        scores = torch.cat(_scores, 0)
        labels = torch.cat(_labels, 0)
        conf = torch.cat(_conf, 0)

        # nms
        boxes = xywh2x1y1x2y2(boxes)
        # keep = batched_nms(boxes, scores, labels, iou_threshold)
        keep = batched_nms(boxes, conf, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))


def training_step_ssd(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    # if self.multiscale:
    #     img = img[0]
    #     target = target[0]
    target = target.view(-1, 5)
    output = self(img)

    boxes = output[..., :4]
    cls = output[..., 4:]

    gt_boxes = target[..., :4]
    gt_cls = target[..., 4]
    positive = gt_cls > 0
    nums_positive = positive.sum()
    if self.focal_loss:
        keep = gt_cls >= 0
        loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
                                      reduction="none")
        if self.reduction == "sum":
            loss_cls = loss_cls.sum()
        elif self.reduction == "mean":
            loss_cls = loss_cls.sum() / nums_positive

    else:
        with torch.no_grad():
            loss = -F.log_softmax(cls, -1)[..., 0]  # 对应 softmax ,第0列对应背景
            loss[positive] = -np.inf
        # 从大到小排序
        negindex = loss.sort(descending=True)[1]
        gt_cls[~positive] = -1
        gt_cls[negindex[:3 * nums_positive]] = 0  # 负样本

        # loss_cls = F.cross_entropy(cls, gt_cls.long(), ignore_index=-1, reduction=reduction)

        keep = gt_cls >= 0
        loss_cls = F.cross_entropy(cls[keep], gt_cls[keep].long(), reduction=reduction)

    loss_boxes = self.boxes_weight * F.smooth_l1_loss(boxes[positive], gt_boxes[positive], reduction=self.reduction)
    # loss_boxes = boxes_weight * F.mse_loss(boxes[positive], gt_boxes[positive], reduction=self.reduction)

    loss = loss_cls + loss_boxes

    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_ssd(anchor, model, img_paths, transform, resize, device, visual=True, strides=416,
                fix_resize=False, save_path="./output", iou_threshold=0.3, conf_threshold=0.3, focal_loss=False):
    # anchor = np.array(anchor)
    # fh, fw = int(np.ceil(resize / strides)), int(np.ceil(resize / strides))
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img, _ = transform(img, {})

        output = model(img[None].to(device))
        boxes = output[..., :4]
        if focal_loss:
            cls = output[..., 4:].sigmoid()
        else:
            cls = output[..., 4:].softmax(-1)

        anchor_xywh = torch.tensor(x1y1x2y22xywh_np(anchor), dtype=torch.float32, device=device)

        boxes[..., 2:] = boxes[..., 2:].exp() * anchor_xywh[..., 2:]
        boxes[..., :2] = boxes[..., :2] * anchor_xywh[..., 2:] + anchor_xywh[..., :2]

        scores, labels = cls.max(-1)

        keep = torch.bitwise_and(scores > conf_threshold, labels > 0)  # 0 为背景
        if keep.sum() == 0:
            print("no object detecte")
            continue
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]

        # nms
        boxes = xywh2x1y1x2y2(boxes)
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))


def training_step_ssdMS(self, batch, batch_idx, log_name='train_loss'):
    img, target = batch
    # if self.multiscale:
    #     img = img[0]
    #     target = target[0]
    target = target.view(-1, 5)
    output = self(img)

    boxes = output[..., :4]
    cls = output[..., 4:]

    gt_boxes = target[..., :4]
    gt_cls = target[..., 4]
    positive = gt_cls > 0
    nums_positive = positive.sum()
    if self.focal_loss:
        keep = gt_cls >= 0
        # loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
        #                               reduction=self.reduction)
        loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
                                      reduction="none")
        if self.reduction == "sum":
            loss_cls = loss_cls.sum()
        elif self.reduction == "mean":
            loss_cls = loss_cls.sum() / nums_positive

    else:
        with torch.no_grad():
            loss = -F.log_softmax(cls, -1)[..., 0]  # 对应 softmax ,第0列对应背景
            loss[positive] = -np.inf
        # 从大到小排序
        negindex = loss.sort(descending=True)[1]
        gt_cls[~positive] = -1
        gt_cls[negindex[:3 * nums_positive]] = 0  # 负样本

        # loss_cls = F.cross_entropy(cls, gt_cls.long(), ignore_index=-1, reduction=self.reduction)

        keep = gt_cls >= 0
        loss_cls = F.cross_entropy(cls[keep], gt_cls[keep].long(), reduction=self.reduction)

    loss_boxes = self.boxes_weight * F.smooth_l1_loss(boxes[positive], gt_boxes[positive], reduction=self.reduction)
    # loss_boxes = boxes_weight * F.mse_loss(boxes[positive], gt_boxes[positive], reduction=self.reduction)

    loss = loss_cls + loss_boxes

    # self.log(log_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(log_name, loss)

    if log_name == "train_loss":
        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

    return loss


@torch.no_grad()
def predict_ssdMS(anchors, model, img_paths, transform, resize, device, visual=True, strides=[8, 16, 32],
                  fix_resize=False, save_path="./output", iou_threshold=0.3, conf_threshold=0.3, focal_loss=False):
    # anchor = np.array(anchor)
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img, _ = transform(img, {})

        output = model(img[None].to(device))
        boxes = output[..., :4]
        if focal_loss:
            cls = output[..., 4:].sigmoid()
        else:
            cls = output[..., 4:].softmax(-1)

        anchor = np.concatenate(anchors, 0)
        anchor_xywh = torch.tensor(x1y1x2y22xywh_np(anchor), dtype=torch.float32, device=device)

        boxes[..., 2:] = boxes[..., 2:].exp() * anchor_xywh[..., 2:]
        boxes[..., :2] = boxes[..., :2] * anchor_xywh[..., 2:] + anchor_xywh[..., :2]

        scores, labels = cls.max(-1)

        keep = torch.bitwise_and(scores > conf_threshold, labels > 0)  # 0 为背景
        if keep.sum() == 0:
            print("no object detecte")
            continue
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]

        # nms
        boxes = xywh2x1y1x2y2(boxes)
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        if len(keep) == 0:
            print("no object detecte")
            continue
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

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

        print("%d object detecte" % (len(labels)))
