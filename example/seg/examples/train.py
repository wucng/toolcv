"""
# pip install prefetch_generator
# pip install medpy

# 1、
epochs = 100
batch_size = 4
lr = 5e-4
weight_decay = 5e-5
amp = False
accumulation_steps = 1
loss = F.cross_entropy(pred, mask, weight=weight)

max mIOU:0.66549 mean_time:0.889

# 2、（在 #1 基础上修改）
loss = focalloss(pred, mask, weight)
max mIOU:0.65956

# 3、（在 #1 基础上修改）
loss = F.cross_entropy(pred, mask, weight=weight)
loss_d = dice_loss(F.softmax(pred, dim=1).float(),
    F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float(), True)
loss = 0.8 * loss + 0.2 * loss_d

max mIOU:0.66591

# 4、（在 #1 基础上修改）
accumulation_steps = 2 # 相当于 batch_size = batch_size*2
max mIOU:0.61820 mean_time:0.900

# 5、（在 #1 基础上修改）
batch_size=8
max mIOU:0.62707 mean_time:0.929

# 6、（在 #1 基础上修改）
amp=True
max mIOU:0.64408 mean_time:0.903

# 7、（在 #1 基础上修改）
amp=True
accumulation_steps = 2

max mIOU:0.60614 mean_time:0.894
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR

from toolcv.tools.segment.net.YOLOP.net import YOLOPV2
from toolcv.tools.segment.net.espnet import ESPNet
from toolcv.tools.segment.net.pspnet import PSPNet
from toolcv.tools.segment.net.deeplabv3 import deeplabv3plus_resnet50
from toolcv.tools.segment.net.Unet.unet import UNet
from toolcv.tools.segment.net.segnet import SegNet
from toolcv.tools.segment.net.unet import unet
# from net import unet
from toolcv.tools.cls.utils.tools import get_device, set_seed, TrainerV2, load_model_weight, \
    get_optim_scheduler, create_dataset, get_train_val_dataset, DataLoaderX, get_criterion, load_weight
from toolcv.tools.segment.loss.dice_score import dice_loss
from toolcv.tools.segment.evaluation.metrics import ConfusionMatrix, dice, fscore
from toolcv.tools.segment.evaluation.metricsv1 import iou_score, dice_coef
from toolcv.tools.segment.evaluation.IOUEval import iouEval
from toolcv.tools.cls.net.net import _initParmas

from dataset import WaferDataset, FruitsNutsDataset, train_transform, val_transform

save_path = "./output"
summary = SummaryWriter(save_path)
root = r"D:\data\fruitsNuts"
# root = r"../input/fruitsnuts"
classes = ["__background__", 'date', 'fig', 'hazelnut']
color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
model_name = "resnet34"
out_channles = 256
num_classes = len(classes)
device = get_device()
set_seed(100)
epochs = 100
batch_size = 4

resize = (256, 256)
lr = 5e-4
weight_decay = 5e-5
weight_path = 'weight.pth'
best_path = "best.pth"
log_interval = 1
amp = False
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
accumulation_steps = 1 #if amp else 2

"""
# max mIOU:0.52950 mean_time:0.765
model = YOLOPV2(num_classes - 1, mode=["backbone", "fpn", "seg_driving"])
load_weight(model, "", url=model.url)
for parma in model.parameters():
    parma.requires_grad_(False)
for parma in model.model[-9:].parameters():
    parma.requires_grad_(True)
_initParmas(model.model[-9:].modules())
model.to(device)

# 微调 max mIOU:0.60958 mean_time:0.864
model = YOLOPV2(num_classes - 1, mode=["backbone", "fpn", "seg_driving"])
model.to(device)
# """
"""
# max mIOU:0.60152 mean_time:1.219
model = PSPNet(num_classes,backend=model_name,psp_size=512,deep_features_size=256)
model.to(device)
"""

# """
# max mIOU:0.58881 mean_time:1.362
# model = ESPNet(num_classes,p=3, q=5, num_hide=384)
# max mIOU:0.64220 mean_time:1.910
# model = ESPNet(num_classes,p=3, q=5, num_hide=512)
# max mIOU:0.62748 mean_time:1.690
# model = ESPNet(num_classes,p=2, q=2, num_hide=512)
# max mIOU:0.56572 mean_time:1.712
model = ESPNet(num_classes,p=2, q=2, num_hide=256)
model.to(device)
# """

"""
# max mIOU:0.65430 mean_time:1.333
model = deeplabv3plus_resnet50(num_classes)
model.to(device)
"""
"""
# max mIOU:0.60837 mean_time:0.873
model = SegNet(3,num_classes,True)
model.to(device)
"""
"""
# max mIOU:0.51025 mean_time:0.929
model = UNet(3,num_classes)
model.to(device)
load_weight(model,"",model.url,device)
"""

"""
# max mIOU:0.54861 mean_time:0.938
model = unet(model_name, out_channles, num_classes)
model.to(device)
"""
# load_model_weight(model, device, weight_path)
load_model_weight(model, device, best_path)


def weight_category(val_dataLoader=None, mode=1):
    """统计每个类别的像素总数"""
    if val_dataLoader is None:
        val_dataset = FruitsNutsDataset(root, classes, val_transform)
        val_dataLoader = DataLoaderX(val_dataset, 1)
    nums_piexl = [0] * num_classes
    for _, (_, mask) in tqdm(enumerate(val_dataLoader)):
        for j in range(num_classes):
            nums_piexl[j] += (mask == j).sum().item()

    # 设置权重
    nums_piexl = np.array(nums_piexl)
    if mode == 0:
        # 1、参考 Esnet
        class_weights = 1 / np.log(1.02 + nums_piexl / np.sum(nums_piexl))
    else:
        # 2、参考 deeplab
        class_weights = np.median(nums_piexl) / nums_piexl

    return class_weights


def focalloss(pred, mask, weight, alpha=1, gamma=2, reduction="mean"):
    # focal loss
    ce_loss = F.cross_entropy(pred, mask, weight=weight, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def train(model, optim, train_dataLoader, criterion=None, scheduler=None, device="cpu", epoch=0, log_interval=10,
          weight=None):
    model.train()
    ioueval = iouEval(num_classes)
    num_dataLoader = len(train_dataLoader)
    for step, (img, mask) in enumerate(train_dataLoader):
        img = img.to(device)
        mask = mask.long().to(device)

        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(img)
            # loss_d = dice_loss(F.softmax(pred, dim=1).float(),
            #                    F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float(), True)

            loss = F.cross_entropy(pred, mask, weight=weight)
            # focal loss
            # loss = focalloss(pred, mask, weight)
            # loss = 0.8 * loss + 0.2 * loss_d

        # loss.backward()
        # optim.step()
        # optim.zero_grad()

        # optim.zero_grad(set_to_none=True)
        # grad_scaler.scale(loss).backward()
        # if not amp: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # grad_scaler.step(optim)
        # grad_scaler.update()

        # ----------使用梯度累加---------------------------
        # 2.1 loss regularization
        loss = loss / accumulation_steps
        # 2.2 back propagation
        # loss.backward()
        grad_scaler.scale(loss).backward()

        # 3. update parameters of net
        if ((step + 1) % accumulation_steps) == 0:
            # optimizer the net
            # optim.step()  # update parameters of net
            # optim.zero_grad()  # reset gradient

            # 使用 grad_scaler 会自动缩放梯度 不需要手动裁剪梯度
            if not amp: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            grad_scaler.step(optim)
            grad_scaler.update()
            optim.zero_grad()

        # iou
        # ioueval.addBatch(torch.argmax(pred.detach(), 1), mask)
        pred = torch.argmax(pred.detach(), 1)
        [ioueval.addBatch(pred[i], mask[i]) for i in range(mask.size(0))]

        if step % log_interval == 0:
            print("epoch:%d step:%d loss:%.5f" % (epoch, step, loss.item()))

        if scheduler is not None:
            scheduler.step()
            summary.add_scalar("lr", scheduler.get_last_lr()[0], epoch * num_dataLoader + step)

        summary.add_scalar("loss", loss.item(), epoch * num_dataLoader + step)

    overall_acc, per_class_acc, per_class_iu, mIOU = ioueval.getMetric()
    # print("overall_acc:%.5f  mIOU:%.5f" % (overall_acc, mIOU))
    # print(per_class_acc, per_class_iu)

    return overall_acc, mIOU


def mask2rgb(mask, color, num_classes):
    h, w = mask.shape
    color = np.array(color)
    rgb_img = np.ones([h, w, 3])
    for i in range(num_classes):
        rgb_img[mask == i] *= color[i]
    return rgb_img.astype(np.uint8)


@torch.no_grad()
def test(model, val_dataLoader, criterion=None, device="cpu"):
    model.eval()
    # cmatrix = ConfusionMatrix()
    ioueval = iouEval(num_classes)
    for i, (img, mask) in enumerate(val_dataLoader):
        mask = mask.cpu().numpy()[0]
        pred = model(img.to(device))[0]
        pred = torch.argmax(pred, 0).cpu().numpy().astype(np.uint8)

        # ConfusionMatrix
        # cmatrix.set_test(pred)
        # cmatrix.set_reference(mask)

        summary.add_image("test", mask2rgb(pred, color, num_classes), i, dataformats="HWC")

        # iou
        ioueval.addBatch(torch.from_numpy(pred), torch.from_numpy(mask))

        # acc
        # acc = ((mask - pred) == 0).sum() / mask.size
        # iou = iou_score(torch.from_numpy(pred), torch.from_numpy(mask))
        # d_coef = dice_coef(torch.from_numpy(pred), torch.from_numpy(mask))
        # print("acc:%.3f iou:%.3f dice:%.3f" % (acc, iou.item(), d_coef.item()))

        # resize
        # save_path = "./output"
        # if not os.path.exists(save_path): os.makedirs(save_path)
        # if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        # Image.fromarray(pred).resize(shape, 0).save(save_path)
        # Image.fromarray(pred).save(save_path)

        # plt.subplot(1, 2, 1)
        # # plt.imshow(mask * 30)
        # plt.imshow(mask2rgb(mask, color,num_classes))
        # plt.subplot(1, 2, 2)
        # # plt.imshow(pred * 30)
        # plt.imshow(mask2rgb(pred, color,num_classes))
        # plt.savefig(os.path.join(save_path, "%d.jpg" % i))
        # plt.show()

    # cmatrix.compute()

    overall_acc, per_class_acc, per_class_iu, mIOU = ioueval.getMetric()
    # print("overall_acc:%.5f  mIOU:%.5f" % (overall_acc, mIOU))
    # print(per_class_acc, per_class_iu)
    # print("fscore:%.5f dice:%.5f" % (fscore(confusion_matrix=cmatrix), dice(confusion_matrix=cmatrix)))

    return overall_acc, mIOU


def fit(epochs=100):
    train_dataset = FruitsNutsDataset(root, classes, train_transform)
    val_dataset = FruitsNutsDataset(root, classes, val_transform)
    train_dataset, val_dataset = get_train_val_dataset(train_dataset, val_dataset, 0.8)

    train_dataLoader = DataLoaderX(train_dataset, batch_size, True, drop_last=False)
    val_dataLoader = DataLoaderX(val_dataset, 1)

    optim, scheduler = get_optim_scheduler(model, len(train_dataLoader) * 4, lr, weight_decay)
    criterion = get_criterion(mode="labelsmoothfocal")
    # scheduler = OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_dataLoader), epochs=epochs)
    # scheduler = MultiStepLR(optim, milestones=[30, 80], gamma=0.1)

    # weight = torch.ones([num_classes], device=device)
    # weight[0] = 0.1  # 背景 权重降低
    weight = torch.tensor(weight_category(mode=0), device=device, dtype=torch.float32)
    print(weight)

    t_miou = []
    cost_time = []
    _mIOU = 0.0
    for epoch in range(epochs):
        start = time.time()
        train_mAcc, train_mIOU = train(model, optim, train_dataLoader, criterion, scheduler, device, epoch,
                                       log_interval, weight)
        test_mAcc, test_mIOU = test(model, val_dataLoader, criterion, device)
        end = time.time()

        # scheduler.step()
        # summary.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        torch.save(model.state_dict(), weight_path)

        print("train_mAcc:%.5f train_mIOU:%.5f test_mAcc:%.5f test_mIOU:%.5f" % (
            train_mAcc, train_mIOU, test_mAcc, test_mIOU
        ))

        if test_mIOU > _mIOU:
            _mIOU = test_mIOU
            # torch.save({"state_dict": model.state_dict(), "mIOU": _mIOU}, best_path)
            torch.save(model.state_dict(), best_path)

        t_miou.append(test_mIOU)
        cost_time.append(end - start)

        summary.add_scalars("mAcc", {"train_mAcc": train_mAcc, "test_mAcc": test_mAcc}, epoch)
        summary.add_scalars("mIOU", {"train_mIOU": train_mIOU, "test_mIOU": test_mIOU}, epoch)

    print("max mIOU:%.5f mean_time:%.3f" % (np.max(t_miou), np.mean(cost_time)))


@torch.no_grad()
def predict(img_paths=None, save_path="./output", show=True):
    if not os.path.exists(save_path): os.makedirs(save_path)
    if img_paths is None: img_paths = glob(os.path.join(root, "images", "*"))

    model.eval()
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        shape = img.size
        # resize
        img = img.resize(resize[::-1])
        img = np.array(img, np.float32)
        # to tensor
        img = torch.tensor(img.copy() / 255.).permute(2, 0, 1)
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]

        pred = model(img[None].to(device))[0]
        pred = torch.argmax(pred, 0).cpu().numpy().astype(np.uint8)

        # resize
        mask = Image.fromarray(pred).resize(shape, 0)
        # mask.save(os.path.join(save_path, os.path.basename(img_path)))

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(Image.open(img_path).convert('RGB')))
        plt.subplot(1, 2, 2)
        plt.imshow(mask2rgb(np.array(mask), color, num_classes))
        plt.savefig(os.path.join(save_path, os.path.basename(img_path)))
        if show:
            plt.show()


if __name__ == "__main__":
    fit(epochs)
    # predict(save_path=save_path)
