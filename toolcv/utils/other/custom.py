"""
lr0: 0.01
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from toolcv.utils.other.yolov5.general import one_cycle
from toolcv.utils.other.yolov5.plots import plot_lr_scheduler


def get_params(model):
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            if v.bias.requires_grad:
                g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            if v.weight.requires_grad:
                g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            if v.weight.requires_grad:
                g1.append(v.weight)

    return g0, g1, g2


def get_optim_scheduler(model, epochs, chyp={}, mode="adam", lr_mode="linear_lr"):
    """
    https://github.com/ultralytics/yolov5
    """
    # Optimizer
    hyp = dict(lr0=0.01,
               lrf=0.1,
               momentum=0.937,
               weight_decay=0.0005,
               warmup_epochs=3.0,
               warmup_momentum=0.8,
               warmup_bias_lr=0.1)

    hyp.update(chyp)

    g0, g1, g2 = get_params(model)
    if mode == "adam":
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif mode == "adamw":
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2

    # Scheduler
    if lr_mode == "linear_lr":
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    return optimizer, scheduler


class EMA():
    """
    - https://zhuanlan.zhihu.com/p/68748778

    Example:
        # 初始化

        >>>ema = EMA(model, 0.999)
        >>>ema.register()

        # 训练过程中，更新完参数后，同步update shadow weights
        >>>def train():
        >>>    optimizer.step()
        >>>    ema.update()

        # eval前，apply shadow weights；eval之后，恢复原来模型的参数
        >>>def evaluate():
        >>>    ema.apply_shadow()
        >>>    # evaluate
        >>>    ema.restore()

    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def static_class(dataset, classes=[], mode="cls"):
    """统计每个类别的总数"""
    assert mode in ["cls", "det", "seg"]
    num_classes = len(classes)
    nums_of_per_classes = torch.zeros([num_classes])
    for _, target in tqdm(dataset):
        if mode == "cls":
            nums_of_per_classes[target] += 1
        elif mode == "det":
            labels = target['labels']
            for label in labels:
                nums_of_per_classes[label] += 1
        else:
            for j in range(num_classes):
                nums_of_per_classes[j] += (target == j).sum().item()

    return nums_of_per_classes.int().cpu().numpy()


def plot_class(nums_of_per_classes, classes, save_dir="./", mode='pie'):
    plt.style.use("ggplot")
    # plt.style.available
    if mode == "pie":
        plt.axes(aspect=1)
        plt.pie(nums_of_per_classes, labels=classes, autopct="%.0f%%")
    else:
        plt.barh(classes, nums_of_per_classes, 0.5, left=0)
        for x, y in zip(nums_of_per_classes, classes):
            plt.text(x, y, str(x))
    plt.savefig(os.path.join(save_dir, 'labels.jpg'), dpi=200)
    plt.close()

def labels_to_class_weights(nums_of_per_classes, mode=0):
    if mode == 0:
        # 参考 yolov5
        # weights = np.bincount(classes, minlength=nc) # 统计频率
        weights = nums_of_per_classes
        weights[weights == 0] = 1  # replace empty bins with 1
        weights = 1 / weights  # number of targets per class
        weights /= weights.sum()  # normalize
        weights *= len(nums_of_per_classes)

    elif mode == 1:
        # 1、参考 Esnet
        weights = 1 / np.log(1.02 + nums_of_per_classes / np.sum(nums_of_per_classes))
    else:
        # 2、参考 deeplab
        weights = np.median(nums_of_per_classes) / nums_of_per_classes

    return weights

def labels_to_image_weights(dataset, nc=80, class_weights=np.ones(80),mode="det"):
    # Produces image weights based on class_weights and image contents
    assert mode in ["cls", "det", "seg"]
    class_counts = []
    for _,target in dataset:
        if mode == "cls":
            class_counts.append(np.bincount(target.cpu().numpy().astype(np.int), minlength=nc))
        elif mode == "det":
            labels = target['labels']
            class_counts.append(np.bincount(labels.cpu().numpy().astype(np.int), minlength=nc))
        else:
            class_counts.append(np.bincount(target.cpu().numpy().astype(np.int), minlength=nc))

    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


if __name__ == "__main__":
    # from torchvision.models.resnet import resnet18
    #
    # model = resnet18(True)
    # epochs = 100
    #
    # optimizer, scheduler = get_optim_scheduler(model, epochs, mode="adamw", lr_mode="one_cycle")
    #
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # det
    from toolcv.utils.data.data import FruitsNutsDataset

    dataset = FruitsNutsDataset(r"D:/data/fruitsNuts/")
    nums_of_per_classes = static_class(dataset, dataset.classes, 'det')

    print(nums_of_per_classes)

    # plot_class(nums_of_per_classes, dataset.classes, mode="barh")
    # print(labels_to_class_weights(nums_of_per_classes,mode=0))

    nc = len(dataset.classes)
    maps = np.zeros(nc)  # mAP per class
    cw = labels_to_class_weights(nums_of_per_classes, mode=0) * (1 - maps) ** 2 / nc  # class weights
    iw = labels_to_image_weights(dataset, nc=nc, class_weights=cw)  # image weights
    # dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

    print(iw)

    pbar = enumerate(train_loader)
    nb = len(train_loader)  # number of batches
    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for i, (imgs, targets, paths, _) in pbar:
        pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
            f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

    # 多尺度训练
    imgsz = 416
    gs = 32
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Matches
    # t.shape [3, 58, 7] ; anchors.shape [3,2] ; r.shape [3,58,2]; j.shape [3,58]
    r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare  self.hyp['anchor_t'] =4.0
    # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
    t = t[j]  # filter
    # t.shape [138,7]

    # Regression
    pxy = ps[:, :2].sigmoid() * 2 - 0.5
    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
    pbox = torch.cat((pxy, pwh), 1)  # predicted box
    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
    lbox += (1.0 - iou).mean()  # iou loss

    # Objectness
    score_iou = iou.detach().clamp(0).type(tobj.dtype)
    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    obji = self.BCEobj(pi[..., 4], tobj)
    lobj += obji * self.balance[i]  # obj loss