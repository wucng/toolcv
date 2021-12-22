"""
!pip install einops thop timm

5类鲜花数据集
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader
import numpy as np

from torch.hub import load_state_dict_from_url

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms as T
import time
import os
import csv
from tqdm import tqdm
import math
import csv
import random

from timm.optim import RAdam, RMSpropTF
from timm.data import create_transform, RandomResizedCropAndInterpolation, \
    RandAugment, AutoAugment, rand_augment_ops, auto_augment_policy

from toolcv.tools.cls.optim.lr_scheduler import SineAnnealingLROnecev2
from toolcv.tools.tools_summary import model_profile
from toolcv.tools.cls.net.net import _initParmas
from toolcv.api.pytorch_lightning.net import get_params
from toolcv.tools.cls.loss.lossv2 import LabelSmooth, LabelSmoothFocal, OneShotLoss, TripletLoss, DistilLoss
from toolcv.tools.cls.augment.mixup import mixup_cls
from toolcv.tools.cls.augment.mosaic import mosaictwo_cls, mosaicfour_cls

# from torch.utils.data import DataLoader
# pip install prefetch_generator
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """然后用DataLoaderX替换原本的DataLoader
    prefetch_generator 加速 Pytorch 数据读取
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcherv2():
    """
    https://zhuanlan.zhihu.com/p/97190313

    注意dataloader必须设置pin_memory=True

    # ----改造前----
    for iter_id, batch in enumerate(data_loader):
        if iter_id >= num_iters:
            break
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        run_step()

    # ----改造后----
    prefetcher = DataPrefetcher(data_loader, opt)
    batch = prefetcher.next()
    iter_id = 0
    while batch is not None:
        iter_id += 1
        if iter_id >= num_iters:
            break
        run_step()
        batch = prefetcher.next()

    """

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            # return
            raise ("over")
        with torch.cuda.stream(self.stream):
            # for k in self.batch:
            #     if k != 'meta':
            #         self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)
            if isinstance(self.batch, (list, tuple)):
                for i in range(len(self.batch)):
                    self.batch[i] = self.batch[i].to(device=self.device, non_blocking=True)
            else:
                self.batch = self.batch.to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def __iter__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        while True:
            try:
                batch = self.batch
                yield batch
                self.preload()
            except:
                break


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        while True:
            try:
                batch = next(self.loader)
                with torch.cuda.stream(self.stream):
                    if isinstance(batch, (list, tuple)):
                        for i in range(len(batch)):
                            batch[i] = batch[i].to(device=self.device, non_blocking=True)
                    else:
                        batch = batch.to(device=self.device, non_blocking=True)
                yield batch
            except:
                break


def set_seed(seed, only_torch=True):
    # set the seed
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
    if not only_torch:
        random.seed(seed)
        np.random.seed(seed)


def get_device(device='cuda:0'):
    if torch.cuda.is_available():
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def create_model(pretrained=False, num_classes=1000, model_name="resnet18"):
    if model_name in ["resnet18", "resnet34", "resnet50", "dla34", "dla46_c", "dla60_res2next"]:
        from torchvision.models.resnet import resnet18, resnet34, resnet50
        from timm.models.dla import dla34, dla46_c, dla60_res2next
        if model_name == "resnet18":
            Model = resnet18
        elif model_name == "resnet34":
            Model = resnet34
        elif model_name == "resnet50":
            Model = resnet50
        elif model_name == "dla34":
            Model = dla34
        elif model_name == "dla46_c":
            Model = dla46_c
        elif model_name == "dla60_res2next":
            Model = dla60_res2next

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.fc = nn.Linear(model.inplanes, num_classes)
            _initParmas(model.fc.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["resnetrs50", "seresnet50", 'seresnext50_32x4d', "resnest50d", "res2next50",
                        "res2net50_14w_8s"]:
        from timm.models.resnet import resnetrs50, seresnet50, seresnext50_32x4d
        from timm.models.resnest import resnest50d
        from timm.models.res2net import res2next50, res2net50_14w_8s
        if model_name == "resnetrs50":
            Model = resnetrs50
        elif model_name == "seresnet50":
            Model = seresnet50
        elif model_name == "seresnext50_32x4d":
            Model = seresnext50_32x4d
        elif model_name == "resnest50d":
            Model = resnest50d
        elif model_name == "res2next50":
            Model = res2next50
        elif model_name == "res2net50_14w_8s":
            Model = res2net50_14w_8s

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.fc = nn.Linear(model.num_features, num_classes)
            _initParmas(model.fc.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["tresnet_m", "tresnet_l", "cspdarknet53", "cspresnext50", "cspresnet50", "repvgg_b0"]:
        from timm.models.tresnet import tresnet_m, tresnet_l
        from timm.models.cspnet import cspdarknet53, cspresnext50, cspresnet50
        from timm.models.byobnet import repvgg_b0
        if model_name == "tresnet_m":
            Model = tresnet_m
        elif model_name == "tresnet_l":
            Model = tresnet_l
        elif model_name == "cspdarknet53":
            Model = cspdarknet53
        elif model_name == "cspresnext50":
            Model = cspresnext50
        elif model_name == "cspresnet50":
            Model = cspresnet50
        elif model_name == "repvgg_b0":
            Model = repvgg_b0

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.head = nn.Linear(model.num_features, num_classes)
            _initParmas(model.head.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["inception_resnet_v2", "ens_adv_inception_resnet_v2"]:
        from timm.models.inception_resnet_v2 import inception_resnet_v2, ens_adv_inception_resnet_v2
        if model_name == "inception_resnet_v2":
            Model = inception_resnet_v2
        elif model_name == "ens_adv_inception_resnet_v2":
            Model = ens_adv_inception_resnet_v2

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.classif = nn.Linear(model.num_features, num_classes)
            _initParmas(model.classif.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["ghostnet_050"]:
        from timm.models.ghostnet import ghostnet_050
        if model_name == "ghostnet_050":
            Model = ghostnet_050

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.classifier = nn.Linear(model.num_features, num_classes)
            _initParmas(model.classifier.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["mnasnet_100", "semnasnet_100", "mobilenetv2_100", "efficientnet_b0", "efficientnet_es",
                        "efficientnet_lite0", "tf_efficientnet_b0", "tf_efficientnet_b0_ap", "tf_efficientnet_b0_ns",
                        "mixnet_s", "mixnet_m"]:

        from timm.models.efficientnet import (mnasnet_100, semnasnet_100, mobilenetv2_100,
                                              efficientnet_b0, efficientnet_es, efficientnet_lite0, tf_efficientnet_b0,
                                              tf_efficientnet_b0_ap, tf_efficientnet_b0_ns, mixnet_s, mixnet_m)
        if model_name == "mnasnet_100":
            Model = mnasnet_100
        if model_name == "semnasnet_100":
            Model = semnasnet_100
        if model_name == "mobilenetv2_100":
            Model = mobilenetv2_100
        if model_name == "efficientnet_b0":
            Model = efficientnet_b0
        if model_name == "efficientnet_es":
            Model = efficientnet_es
        if model_name == "efficientnet_lite0":
            Model = efficientnet_lite0
        if model_name == "tf_efficientnet_b0":
            Model = tf_efficientnet_b0
        if model_name == "tf_efficientnet_b0_ap":
            Model = tf_efficientnet_b0_ap
        if model_name == "tf_efficientnet_b0_ns":
            Model = tf_efficientnet_b0_ns
        if model_name == "mixnet_s":
            Model = mixnet_s
        if model_name == "mixnet_m":
            Model = mixnet_m

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.classifier = nn.Linear(model.num_features, num_classes)
            _initParmas(model.classifier.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    elif model_name in ["swin_base_patch4_window7_224", "vit_base_patch16_224", "visformer_small",
                        "resmlp_12_224", "gmixer_12_224", "mixer_b16_224"]:
        from timm.models.swin_transformer import swin_base_patch4_window7_224
        from timm.models.vision_transformer import vit_base_patch16_224
        from timm.models.visformer import visformer_small
        from timm.models.mlp_mixer import resmlp_12_224, gmixer_12_224, mixer_b16_224
        if model_name == "swin_base_patch4_window7_224":
            Model = swin_base_patch4_window7_224
        if model_name == "vit_base_patch16_224":
            Model = vit_base_patch16_224
        if model_name == "visformer_small":
            Model = visformer_small
        if model_name == "resmlp_12_224":
            Model = resmlp_12_224
        if model_name == "gmixer_12_224":
            Model = gmixer_12_224
        if model_name == "mixer_b16_224":
            Model = mixer_b16_224

        if pretrained:
            model = Model(pretrained)
            for parma in model.parameters():
                parma.requires_grad_(False)
            model.head = nn.Linear(model.num_features, num_classes)
            _initParmas(model.head.modules())
        else:
            model = Model(pretrained, num_classes=num_classes)

    # from timm.models.regnet import regnetx_004,regnety_004
    # from timm.models.dpn import dpn68

    return model


def test_model(pretrained=False, num_classes=1000):
    # from timm.models.cspnet import cspdarknet53, cspresnext50
    # from timm.models.res2net import res2next50
    # from timm.models.resnet import resnetrs50, seresnext50_32x4d
    # from timm.models.tresnet import tresnet_l
    # from timm.models.resnest import resnest26d, resnest50d
    # from timm.models.rexnet import rexnet_100
    # from timm.models.senet import legacy_seresnet50
    # from timm.models.sknet import skresnext50_32x4d
    # from timm.models.inception_resnet_v2 import ens_adv_inception_resnet_v2
    # from timm.models.regnet import regnetx_004
    from timm.models.efficientnet import efficientnet_b0, eca_efficientnet_b0

    model = efficientnet_b0(pretrained)
    for parma in model.parameters():
        parma.requires_grad_(False)
    for parma in model.blocks[-1].parameters():
        parma.requires_grad_(True)
    for parma in model.conv_head.parameters():
        parma.requires_grad_(True)

    model.classifier = nn.Linear(model.num_features, num_classes)
    _initParmas(model.classifier.modules())

    return model


def get_train_val_dataset(train_dataset, val_dataset, train_ratio=0.8):
    """
    nums = len(train_dataset)
    nums_train = int(nums * train_ratio)
    nums_val = nums - nums_train
    train_dataset, val_dataset = random_split(train_dataset, [nums_train, nums_val])
    """

    # train_ratio = 1 - 0.1
    num_datas = len(train_dataset)
    num_train = int(train_ratio * num_datas)
    indices = torch.randperm(num_datas).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[num_train:])

    return train_dataset, val_dataset


def get_optim_scheduler(model, T_max, lr=2e-3, weight_decay=1e-4, mode="radam",
                        scheduler_mode="CosineAnnealingLR".lower(), lrf=0.01, gamma=0.6):
    if gamma <= 0:
        params = model.parameters()
        params = [param for param in params if param.requires_grad]
    else:
        params = get_params(model.modules(), lr, weight_decay, gamma)

    if mode == "radam":
        optim = RAdam(params, lr, weight_decay=weight_decay)
    elif mode == "adam":
        optim = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif mode == "adamw":
        optim = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    elif mode == "RMSprop".lower():
        optim = torch.optim.RMSprop(params, lr, weight_decay=weight_decay, momentum=0.9)
    elif mode == "RMSpropTF".lower():
        optim = RMSpropTF(params, lr, weight_decay=weight_decay, momentum=0.9)
    elif mode == "SGD".lower():
        optim = torch.optim.SGD(params, lr, weight_decay=weight_decay, momentum=0.9)

    if scheduler_mode == "CosineAnnealingLR".lower():
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max, lr * lrf)
    else:
        scheduler = SineAnnealingLROnecev2(optim, T_max, lrf)  # len(train_dataloader) * 4

    return optim, scheduler


def train_step(model, optimizer, scheduler, dataloader, criterion, device, epoch, log_interval=100):
    start = time.time()
    model.train()
    total_acc, total_count = 0, 0
    total_loss = 0
    # dpar = tqdm(enumerate(dataloader))
    for idx, (data, label) in enumerate(dataloader):
        start_time = time.time()
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predited_label = model(data)
        loss = criterion(predited_label, label)
        total_loss += loss.item()
        loss = loss / predited_label.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if label.ndim == 2: label = label.argmax(1)
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        scheduler.step()

        """
        elapsed = time.time() - start_time
        desc = '| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed {:.5f}'. \
            format(epoch, idx,len(dataloader),total_acc / total_count,elapsed)
        dpar.set_description(desc)
        """
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            desc = '| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed {:.5f}'. \
                format(epoch, idx, len(dataloader), total_acc / total_count, elapsed)
            print(desc)
            # dpar.set_description(desc)

        # """
    end = time.time()

    return total_acc / total_count, total_loss / total_count, end - start


def test_step(model, dataloader, criterion, device):
    start = time.time()
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0

    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            label = label.to(device)
            data = data.to(device)
            predited_label = model(data)
            loss = criterion(predited_label, label)
            total_loss += loss.item()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    end = time.time()

    return total_acc / total_count, total_loss / total_count, end - start


def fit(model, optim, scheduler, criterion, train_dataloader, val_dataloader, device, epochs, log_interval,
        weight_path="weight.pth", log_file="log.csv"):
    # file = open(log_file, 'a')
    csv_file = csv.writer(open(log_file, 'a'))
    csv_file.writerow(
        ["epoch", "train_time", "train_accuracy", "train_loss", "test_time", "test_accuracy", "test_loss"])
    _test_loss = 0
    test_accuracy_list = []
    cost_time_list = []
    for epoch in range(epochs):
        print(epoch, "开始！！")
        # 训练循环
        epoch_st = time.time()
        train_accuracy, train_loss, train_time = train_step(model, optim, scheduler, train_dataloader, criterion,
                                                            device, epoch, log_interval)

        test_accuracy, test_loss, test_time = test_step(model, val_dataloader, criterion, device)

        # scheduler.step()

        # if test_loss > _test_loss:
        #     scheduler.step()
        #     _test_loss = test_loss

        # if epoch % 50 == 0:
        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        line = template.format(epoch, train_loss,
                               train_accuracy * 100, test_loss,
                               test_accuracy * 100)

        cost_time = time.time() - epoch_st
        cost_time_list.append(cost_time)
        test_accuracy_list.append(test_accuracy)

        print("-" * 50)
        print(line)
        print("-" * 50)
        print("epoch", epoch, "::", cost_time)
        print("-" * 50)

        torch.save(model.state_dict(), weight_path)

        # file.write("epoch:%d train_time:%.5f train_acc:%.5f train_loss:%.5f "
        #            "val_time:%.5f val_acc:%.5f val_loss:%.5f"
        #            "\n" % (epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss))
        #
        # file.flush()
        csv_file.writerow([epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss])

    # file.close()

    print("mean time:%.3f" % np.mean(cost_time_list))
    print("max test acc:%.5f" % np.max(test_accuracy_list))


def get_transforms(input_size=(224, 224), mode="v1"):
    expand = (int(input_size[0] * 1.1), int(input_size[1] * 1.1))
    if mode == "v1":
        # bsts = [T.Resize((256, 256)), T.CenterCrop(input_size)]
        bsts = [T.Resize(expand), T.CenterCrop(input_size)]
    elif mode == "v2":
        bsts = [RandomResizedCropAndInterpolation(input_size)]
    elif mode == "v3":
        bsts = [RandomResizedCropAndInterpolation(input_size), RandAugment(rand_augment_ops())]
    elif mode == "v4":
        bsts = [RandomResizedCropAndInterpolation(input_size), AutoAugment(auto_augment_policy())]
    else:
        bsts = [T.Resize(input_size), T.CenterCrop(input_size)]

    train_transforms = T.Compose([
        *bsts,
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        # T.Resize((256, 256)),
        T.Resize(expand),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def get_transformsv2(input_size=(224, 224)):
    train_transforms = create_transform(input_size, is_training=True)
    val_transforms = create_transform(input_size, is_training=False)

    return train_transforms, val_transforms


def get_criterion(reduction="sum", mode="labelsmooth"):
    if mode == "labelsmooth":
        criterion = LabelSmooth(reduction=reduction)
    elif mode == "labelsmoothfocal":
        criterion = LabelSmoothFocal(reduction=reduction)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    return criterion


def getitem(self, index, num_classes=None, mode=None):
    """
    # mixup or mosica

    Args:
        index (int): Index

    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """

    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    if num_classes is None: return sample, target
    if mode is None:
        mode = random.choice(["mixup", "mosaictwo", "mosaicfour", "none"])
    # mixup or mosica
    if mode == "mixup":
        while True:
            idx = random.choice(range(self.__len__()))
            path2, target2 = self.samples[idx]
            if target2 != target:
                sample2 = self.loader(path2)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                break

        sample, target = mixup_cls([sample, sample2], [torch.tensor(target), torch.tensor(target2)], num_classes)

    elif mode == "mosaictwo":
        while True:
            idx = random.choice(range(self.__len__()))
            path2, target2 = self.samples[idx]
            if target2 != target:
                sample2 = self.loader(path2)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                break

        mode = random.choice(["horizontal", "vertical"])
        use_random = random.choice([False, True])
        sample, target = mosaictwo_cls([sample, sample2], [torch.tensor(target), torch.tensor(target2)], use_random,
                                       mode, num_classes)

    elif mode == "mosaicfour":
        imgs = [sample]
        targets = [torch.tensor(target)]
        while True:
            idx = random.choice(range(self.__len__()))
            path2, target2 = self.samples[idx]
            if target2 not in targets:
                sample2 = self.loader(path2)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                imgs.append(sample2)
                targets.append(torch.tensor(target2))
            if len(targets) == 4: break
        use_random = random.choice([False, True])
        sample, target = mosaicfour_cls(imgs, targets, use_random, num_classes=num_classes)
    else:
        target = F.one_hot(torch.tensor(target), num_classes)

    return sample, target


def getitem_err(self, index, num_classes):
    """90%选择正确的标签，10%选择错误的标签"""
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    target2 = target
    if random.random() > 0.9:
        while True:
            target2 = random.choice(range(num_classes))
            if target2 != target:
                break

    return sample, target, target2


def oneshot_getitem(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    if random.choice([0, 1]):
        # 选择相同的
        while True:
            idx = random.choice(range(self.__len__()))
            path2, target2 = self.samples[idx]
            if target2 == target and idx != index:
                sample2 = self.loader(path2)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                break
    else:
        # 选择不同的
        while True:
            idx = random.choice(range(self.__len__()))
            path2, target2 = self.samples[idx]
            if target2 != target:
                sample2 = self.loader(path2)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                break

    return sample, sample2, target, target2


def triplet_getitem(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    # 一张相同的一张不同的
    # 选择相同的
    while True:
        idx = random.choice(range(self.__len__()))
        path2, target2 = self.samples[idx]
        if target2 == target and idx != index:
            sample2 = self.loader(path2)
            if self.transform is not None:
                sample2 = self.transform(sample2)
            break

    # 选择不同的
    while True:
        idx = random.choice(range(self.__len__()))
        path3, target3 = self.samples[idx]
        if target3 != target:
            sample3 = self.loader(path3)
            if self.transform is not None:
                sample3 = self.transform(sample3)
            break

    return sample, sample2, sample3, target, target2, target3


class ComDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None,
                 num_classes=None, mode="none"
                 ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.num_classes = num_classes
        self.mode = mode

    def __getitem__(self, index):
        if self.mode in ["mixup", "mosaictwo", "mosaicfour", "none"]:
            return getitem(self, index, self.num_classes, self.mode)
        elif self.mode == "selecterr":
            return getitem_err(self, index, self.num_classes)
        elif self.mode == "oneshot":
            return oneshot_getitem(self, index)
        elif self.mode == "triplet":
            return triplet_getitem(self, index)
        else:
            # return getitem(self, index)
            return super(ComDataset, self).__getitem__(index)


def create_dataset(root, transforms, num_classes=None, mode="none"):
    # m = ImageFolder(root, transforms)
    # if num_classes is not None:
    #     m.__getitem__ = lambda m, index: getitem(m, index, num_classes)
    #     return m
    # if mode == "oneshot":
    #     m.__getitem__ = lambda m, index: oneshot_getitem(m, index)
    #     return m
    # if mode == "triplet":
    #     m.__getitem__ = lambda m, index: triplet_getitem(m, index)
    #     return m

    m = ComDataset(root, transforms, num_classes=num_classes, mode=mode)
    return m


def load_model_weight(model, device="cpu", weight_path='weight.pth'):
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, device)
            tmp_state_dict = {}
            for k, v in model.state_dict().items():
                if k in state_dict and state_dict[k].numel() == v.numel():
                    tmp_state_dict.update({k: state_dict[k]})
                else:
                    tmp_state_dict.update({k: v})
            # model.load_state_dict(state_dict)
            model.load_state_dict(tmp_state_dict)
            print("---------load weight successful-----------")
        else:
            print("---------%s not exists-----------" % weight_path)
    except:
        print("---------load weight fail-----------")


def load_weight(model, weight_path='weight.pth', url="", device="cpu"):
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, device)
        else:
            state_dict = load_state_dict_from_url(url, map_location=device)
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].numel() == v.numel():
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = v
                print("%s not load weight" % k)
        model.load_state_dict(new_state_dict)
        print("---------load weight successful-----------")
        del state_dict
        del new_state_dict
    except:
        print("---------load weight fail-----------")


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, criterion,
                 train_dataloader, test_dataloader, device, log_interval, amp=False, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = device
        self.log_interval = log_interval

        self.amp = amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

        self.__dict__.update(kwargs)

    def configure_optimizers(self):
        lr = 1e-3
        T_max = 20
        if self.optimizer is None:
            params = self.model.parameters()
            params = [param for param in params if param.requires_grad]
            self.optimizer = RAdam(params, weight_decay=5e-5)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, lr * 0.01)

    def train_step(self, batch, step):
        data, label = batch
        data = data.to(self.device)
        label = label.to(self.device)
        self.optimizer.zero_grad()
        predited_label = self.model(data)
        loss = self.criterion(predited_label, label)
        correct_number = (predited_label.argmax(1) == label).sum().item()

        return loss, correct_number

    def train_stepv2(self, batch, step):
        data, label = batch
        data = data.to(self.device)
        label = label.to(self.device)
        # self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.amp):
            predited_label = self.model(data)
            loss = self.criterion(predited_label, label)
        correct_number = (predited_label.argmax(1) == label).sum().item()

        return loss, correct_number

    def train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        total_acc, total_count = 0, 0
        total_loss = 0
        # dpar = tqdm(enumerate(self.train_dataloader))
        for idx, (data, label) in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, correct_number = self.train_step((data, label), idx)
            total_loss += loss.item()
            loss = loss / data.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if label.ndim == 2: label = label.argmax(1)
            total_acc += correct_number
            total_count += label.size(0)

            self.scheduler.step()

            if idx % self.log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                desc = '| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed {:.5f}'. \
                    format(epoch, idx, len(self.train_dataloader), total_acc / total_count, elapsed)
                print(desc)

                # dpar.set_description(desc)

        end = time.time()

        return total_acc / total_count, total_loss / total_count, end - start

    def train_one_epochv2(self, epoch):
        self.model.train()
        start = time.time()
        total_acc, total_count = 0, 0
        total_loss = 0
        # dpar = tqdm(enumerate(self.train_dataloader))
        for idx, (data, label) in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, correct_number = self.train_stepv2((data, label), idx)
            total_loss += loss.item()
            loss = loss / data.size(0)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            if label.ndim == 2: label = label.argmax(1)
            total_acc += correct_number
            total_count += label.size(0)

            self.scheduler.step()

            if idx % self.log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                desc = '| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed {:.5f}'. \
                    format(epoch, idx, len(self.train_dataloader), total_acc / total_count, elapsed)
                print(desc)

                # dpar.set_description(desc)

        end = time.time()

        return total_acc / total_count, total_loss / total_count, end - start

    @torch.no_grad()
    def test_step(self, batch, step):
        data, label = batch
        label = label.to(self.device)
        data = data.to(self.device)
        predited_label = self.model(data)
        loss = self.criterion(predited_label, label)
        correct_number = (predited_label.argmax(1) == label).sum().item()

        return loss, correct_number

    @torch.no_grad()
    def test_one_epoch(self, epoch):
        self.model.eval()
        start = time.time()
        total_acc, total_count = 0, 0
        total_loss = 0
        for idx, (data, label) in enumerate(self.test_dataloader):
            loss, correct_number = self.test_step((data, label), idx)
            total_loss += loss.item()
            total_acc += correct_number
            total_count += label.size(0)
        end = time.time()

        return total_acc / total_count, total_loss / total_count, end - start

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        pass

    @torch.no_grad()
    def evalute(self, **kwargs):
        pass

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, **kwargs):
        pass

    def fit(self, epochs=50, weight_path='weight.pth', log_file="log.csv", **kwargs):

        fp = open(log_file, 'a')
        csv_file = csv.writer(fp)
        csv_file.writerow(
            ["epoch", "train_time", "train_accuracy", "train_loss", "test_time", "test_accuracy", "test_loss"])
        _test_loss = 0
        test_accuracy_list = []
        cost_time_list = []
        for epoch in range(epochs):
            print(epoch, "开始！！")
            epoch_st = time.time()
            if self.amp:
                train_accuracy, train_loss, train_time = self.train_one_epochv2(epoch)
            else:
                train_accuracy, train_loss, train_time = self.train_one_epoch(epoch)

            test_accuracy, test_loss, test_time = self.test_one_epoch(epoch)

            # self.scheduler.step()
            # if test_loss > _test_loss:
            #     self.scheduler.step()
            #     _test_loss = test_loss

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}")
            line = template.format(epoch, train_loss,
                                   train_accuracy * 100, test_loss,
                                   test_accuracy * 100)

            cost_time = time.time() - epoch_st
            cost_time_list.append(cost_time)
            test_accuracy_list.append(test_accuracy)

            print("-" * 50)
            print(line)
            print("-" * 50)
            print("epoch", epoch, "::", cost_time)
            print("-" * 50)

            torch.save(self.model.state_dict(), weight_path)

            # fp.write("epoch:%d train_time:%.5f train_acc:%.5f train_loss:%.5f "
            #            "val_time:%.5f val_acc:%.5f val_loss:%.5f"
            #            "\n" % (epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss))
            #
            fp.flush()
            csv_file.writerow([epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss])

        fp.close()
        print("mean time:%.3f" % np.mean(cost_time_list))
        print("max test acc:%.5f" % np.max(test_accuracy_list))

    def get_current_lr(self):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        return learning_rate

    def load_weight(self, weight_path='weight.pth'):
        load_model_weight(self.model, self.device, weight_path)

    def save_weight(self, epoch, lr, weight_path='weight.pth'):
        torch.save({"weight": self.model.state_dict(), "epoch": epoch, "lr": lr}, weight_path)


class TrainerV2:
    def __init__(self, model: nn.Module, optimizer, scheduler, criterion,
                 train_dataloader, test_dataloader, device, log_interval, amp=False, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = device
        self.log_interval = log_interval

        self.amp = amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

        self.__dict__.update(kwargs)

    def configure_optimizers(self):
        lr = 1e-3
        T_max = 20
        if self.optimizer is None:
            params = self.model.parameters()
            params = [param for param in params if param.requires_grad]
            self.optimizer = RAdam(params, weight_decay=5e-5)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, lr * 0.01)

    def train_step(self, batch, step):
        data, label = batch
        data = data.to(self.device)
        label = label.to(self.device)
        # self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.amp):
            predited_label = self.model(data)
            loss = self.criterion(predited_label, label)
        correct_number = (predited_label.argmax(1) == label).sum().item()

        return loss, correct_number

    def train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        total_acc, total_count = 0, 0
        total_loss = 0
        # dpar = tqdm(enumerate(self.train_dataloader))
        for idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, correct_number = self.train_stepv2(batch, idx)
            total_loss += loss.item()
            loss = loss / data.size(0)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            total_acc += correct_number
            total_count += batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)

            self.scheduler.step()

            if idx % self.log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                desc = '| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} | elapsed {:.5f}'. \
                    format(epoch, idx, len(self.train_dataloader), total_acc / total_count, elapsed)
                print(desc)

                # dpar.set_description(desc)

        end = time.time()

        return total_acc / total_count, total_loss / total_count, end - start

    @torch.no_grad()
    def test_step(self, batch, step):
        data, label = batch
        label = label.to(self.device)
        data = data.to(self.device)
        predited_label = self.model(data)
        loss = self.criterion(predited_label, label)
        correct_number = (predited_label.argmax(1) == label).sum().item()

        return loss, correct_number

    @torch.no_grad()
    def test_one_epoch(self, epoch):
        self.model.eval()
        start = time.time()
        total_acc, total_count = 0, 0
        total_loss = 0
        for idx, batch in enumerate(self.test_dataloader):
            loss, correct_number = self.test_step(batch, idx)
            total_loss += loss.item()
            total_acc += correct_number
            total_count += batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
        end = time.time()

        return total_acc / total_count, total_loss / total_count, end - start

    @torch.no_grad()
    def evalute_step(self, batch, step, **kwargs):
        pass

    @torch.no_grad()
    def evalute(self, **kwargs):
        pass

    @torch.no_grad()
    def pred_step(self, imgs, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, **kwargs):
        pass

    def fit(self, epochs=50, weight_path='weight.pth', log_file="log.csv", **kwargs):

        fp = open(log_file, 'a')
        csv_file = csv.writer(fp)
        csv_file.writerow(
            ["epoch", "train_time", "train_accuracy", "train_loss", "test_time", "test_accuracy", "test_loss"])
        _test_loss = 0
        test_accuracy_list = []
        cost_time_list = []
        for epoch in range(epochs):
            print(epoch, "开始！！")
            epoch_st = time.time()
            train_accuracy, train_loss, train_time = self.train_one_epoch(epoch)
            test_accuracy, test_loss, test_time = self.test_one_epoch(epoch)

            # self.scheduler.step()
            # if test_loss > _test_loss:
            #     self.scheduler.step()
            #     _test_loss = test_loss

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}")
            line = template.format(epoch, train_loss,
                                   train_accuracy * 100, test_loss,
                                   test_accuracy * 100)

            cost_time = time.time() - epoch_st
            cost_time_list.append(cost_time)
            test_accuracy_list.append(test_accuracy)

            print("-" * 50)
            print(line)
            print("-" * 50)
            print("epoch", epoch, "::", cost_time)
            print("-" * 50)

            torch.save(self.model.state_dict(), weight_path)

            # fp.write("epoch:%d train_time:%.5f train_acc:%.5f train_loss:%.5f "
            #            "val_time:%.5f val_acc:%.5f val_loss:%.5f"
            #            "\n" % (epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss))
            #
            fp.flush()
            csv_file.writerow([epoch, train_time, train_accuracy, train_loss, test_time, test_accuracy, test_loss])

        fp.close()
        print("mean time:%.3f" % np.mean(cost_time_list))
        print("max test acc:%.5f" % np.max(test_accuracy_list))

    def get_current_lr(self):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        return learning_rate

    def load_weight(self, weight_path='weight.pth'):
        load_model_weight(self.model, self.device, weight_path)

    def save_weight(self, epoch, lr, weight_path='weight.pth'):
        torch.save({"weight": self.model.state_dict(), "epoch": epoch, "lr": lr}, weight_path)


if __name__ == "__main__":
    import time

    dataset = TensorDataset(torch.randn([1000, 3, 224, 224]), torch.from_numpy(np.random.choice(5, [1000])))
    dataloader = DataLoaderX(dataset, 20, False, num_workers=1, pin_memory=True)
    dataloader2 = DataLoader(dataset, 20, False, num_workers=1, pin_memory=True)
    device = "cuda:0"

    start = time.time()
    for batch in dataloader2:
        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        print(batch[1])
    print(time.time() - start)  # 6.623945713043213

    start = time.time()
    for batch in dataloader:
        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        print(batch[1])
    print(time.time() - start)  # 4.242402791976929
    exit(0)
    start = time.time()
    prefetcher = DataPrefetcherv2(dataloader, device)
    for batch in prefetcher:
        print(batch[1])
    print(time.time() - start)  # 4.220939874649048

    start = time.time()
    prefetcher = DataPrefetcher(dataloader, device)
    for batch in prefetcher:
        print(batch[1])
    print(time.time() - start)  # 4.245075941085815
