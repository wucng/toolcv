# https://www.kaggle.com/fengzhongyouxia/flower-photos-clsv2
# https://www.kaggle.com/fengzhongyouxia/cv-test-tricks-cls

"""
推荐方式：
1、模型迁移
    - model:efficientnet_b0
    - epochs:15
    - batch_size:64
    - weight_decay:5e-5
    - lr:5e-4
    - 80%训练 20%测试
    - size:(224,224)
    - transforms: mode = "v1"
    - loss: CrossEntropyLoss
    - optim:RAdam
    - lr_scheduler:CosineAnnealingLR

mean time:23.893
max test acc:0.93052

2、模型微调
    - weight_decay:1e-4  # 正则变大
    - lr:5e-5 # 学习率降低
    - transforms: mode = "v2" # 使用数据增强
    - loss: labelsmooth
    - create_dataset:mode="oneshot"

    通过 #1 的训练发现：
        * 1、训练acc 高 而 测试的acc 低 出现过拟合 - 增加 weight_decay （减少有效参数）
        * 2、出现过拟合 增大 transforms
        * 3、使用模型微调  需增加 weight_decay 并 学习率降低

mean time:52.906
max test acc:0.99455


2.1、模型微调
    - weight_decay:5e-5
    - lr:5e-4
    - transforms: mode = "v2" # 使用数据增强
    - loss: labelsmooth
    - create_dataset:mode="oneshot"

    通过 #2 的训练发现：
    *  1、训练acc 低 出现欠拟合 - 降低 weight_decay （增加有效参数）
    *  2、学习能力略有不足 - 适当增加 lr
    *  3、未出现过拟合 不修改 transforms
    *  4、如果 降低 weight_decay 还是出现欠拟合 需要换一个更大的网络

mean time:52.325
max test acc:1.00000


如果出现过拟合 修改
    1、增大 weight_decay
    2、使用 dropout、dropblock、droppath
    3、数据增强 get_transforms(mode) mode=v2,v3,v4, get_transformsv2() （推荐 v2）
    3.1、数据增强 create_dataset(mode) mode = mixup", "mosaictwo", "mosaicfour" （推荐 "mosaictwo"）

3、训练策略：
    数据选择：create_dataset(mode) mode = "selecterr","oneshot","triplet" （推荐 "oneshot"）
    模型蒸馏 + 逐步放大 size + 正则
4、 修改 criterion、optimizer, lr_scheduler
"""

"""
推荐方式：
1、模型迁移
    - model:efficientnet_b0
    - epochs:15
    - batch_size:64
    - weight_decay:5e-5
    - lr:5e-4
    - 80%训练 20%测试
    - size:(224,224)
    - transforms: mode = "v1"
    - loss: CrossEntropyLoss
    - optim:RAdam
    - lr_scheduler:CosineAnnealingLR

mean time:23.893
max test acc:0.93052

2、模型微调
    - epochs:20
    - weight_decay:5e-5
    - lr:5e-4
    - transforms: mode = "v2" # 使用数据增强
    - loss: labelsmooth
    - create_dataset:mode="oneshot"

mean time:53.204
max test acc:1.00000
"""

"""
# 说明
1、
- model:efficientnet_b0 (模型迁移)
- epochs:15
- batch_size:64
- weight_decay:5e-5
- lr:5e-4
- 80%训练 20%测试
- size:(224,224)
- transforms: mode = "v1"
- loss: CrossEntropyLoss
- optim:RAdam
- lr_scheduler:CosineAnnealingLR

mean time:26.676
max test acc:0.932
注意：#1 训练 出现过拟合

2、
# 2 是在 #1的训练基础上训练 并且使用 #1 训练的权重
2.1、
- transforms: mode = "v2" # RandomResizedCropAndInterpolation
mean time:25.330
max test acc:0.989

2.2、
- transforms: mode = "v2"
- loss: labelsmoothfocal
mean time:25.406
max test acc:0.981

2.3、
- transforms: mode = "v2"
- loss: labelsmooth
mean time:24.776
max test acc:0.990

2.4、
- transforms: mode = "v2"
- loss: labelsmooth
- lr_scheduler:SineAnnealingLROnecev2
mean time:25.044
max test acc:0.985

2.5、
- transforms: mode = "v3" # 随机数据增强
- loss: labelsmooth
mean time:29.614
max test acc:0.988

2.6、
- transforms: mode = "v4" # 自动数据增强
- loss: labelsmooth
mean time:29.420
max test acc:0.986

2.7、
- transforms: get_transformsv2
- loss: labelsmooth
mean time:32.792
max test acc:0.980

2.8、
- transforms: mode = "v1"
- loss: labelsmooth
- create_dataset:mode="mixup"
mean time:43.269
max test acc:0.973

2.9、
- transforms: mode = "v1"
- loss: labelsmooth
- create_dataset:mode="mosaictwo"
mean time:37.763
max test acc:0.988

2.10、
- transforms: mode = "v1"
- loss: labelsmooth
- create_dataset:mode="mosaicfour"

mean time:68.898
max test acc:0.982

2.11、
- transforms: mode = "v1"
- loss: labelsmooth
- create_dataset:mode=random.choice(["mixup", "mosaictwo", "mosaicfour", "none"])
mean time:66.789
max test acc:0.981

2.12、
- transforms: mode = "v2"
- loss: labelsmooth
- create_dataset:mode="selecterr"
mean time:21.681
max test acc:0.988

2.13、
- transforms: mode = "v2"
- loss: labelsmooth
- create_dataset:mode="oneshot"
mean time:38.844
max test acc:0.992

2.14、
- transforms: mode = "v2"
- loss: labelsmooth
- create_dataset:mode="triplet"
mean time:57.885
max test acc:0.990

2.15、
- transforms: mode = "v2"
- loss: labelsmooth
- create_dataset:mode="oneshot"
- 训练方式 蒸馏 + 逐步放大 size + 正则

2.16
- 模型微调
    - weight_decay:1e-4
    - lr:5e-5
- transforms: mode = "v2"
- loss: labelsmooth
- create_dataset:mode="oneshot"

mean time:52.906
max test acc:0.99455

"""
import torch
import os
import random
from torch.utils.data import DataLoader  # ,TensorDataset, Dataset,random_split
import time
from torch.nn import functional as F

from toolcv.tools.cls.utils.tools import get_device, get_transforms, get_transformsv2, create_dataset, \
    get_train_val_dataset, test_model, create_model, load_model_weight, model_profile, get_optim_scheduler, \
    get_criterion, fit, Trainer
from toolcv.tools.cls.loss.lossv2 import labelsmooth, oneshotLoss, tripletLoss, selecterrLoss


def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data, target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return torch.stack(data_list, 0), torch.stack(target_list, 0)


"""
# - create_dataset:mode="selecterr"
def train_step(self, batch, step):
    data, label2, label = batch
    data = data.to(self.device)
    label = label.to(self.device)
    label2 = label2.to(self.device)
    self.optimizer.zero_grad()
    predited_label = self.model(data)
    # loss = self.criterion(predited_label, label)

    loss = selecterrLoss(predited_label,label2,label)

    if label.ndim == 2: label = label.argmax(1)
    correct_number = (predited_label.argmax(1) == label).sum().item()

    return loss, correct_number


def train_one_epoch(self, epoch):
    self.model.train()
    start = time.time()
    total_acc, total_count = 0, 0
    total_loss = 0
    # dpar = tqdm(enumerate(self.train_dataloader))
    for idx, (data, label2, label) in enumerate(self.train_dataloader):
        start_time = time.time()
        loss, correct_number = self.train_step((data, label2, label), idx)
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

# """


# """
# - create_dataset:mode="oneshot"
def train_step(self, batch, step):
    data, data2, label, label2 = batch
    data = data.to(self.device)
    label = label.to(self.device)
    data2 = data2.to(self.device)
    label2 = label2.to(self.device)

    self.optimizer.zero_grad()
    predited_label = self.model(data)
    predited_label2 = self.model(data2)

    # loss = self.criterion(predited_label, label)
    loss = oneshotLoss(predited_label, predited_label2, label, label2, 0.3)

    if label.ndim == 2: label = label.argmax(1)
    correct_number = (predited_label.argmax(1) == label).sum().item()

    return loss, correct_number


def train_one_epoch(self, epoch):
    self.model.train()
    start = time.time()
    total_acc, total_count = 0, 0
    total_loss = 0
    # dpar = tqdm(enumerate(self.train_dataloader))
    for idx, (data, data2, label, label2) in enumerate(self.train_dataloader):
        start_time = time.time()
        loss, correct_number = self.train_step((data, data2, label, label2), idx)
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


# """

"""
# - create_dataset:mode="triplet"
def train_step(self, batch, step):
    data, data2,data3, label, label2,label3 = batch
    data = data.to(self.device)
    label = label.to(self.device)
    data2 = data2.to(self.device)
    label2 = label2.to(self.device)
    data3 = data3.to(self.device)
    label3 = label3.to(self.device)

    self.optimizer.zero_grad()
    predited_label = self.model(data)
    predited_label2 = self.model(data2)
    predited_label3 = self.model(data3)

    # loss = self.criterion(predited_label, label)
    loss = tripletLoss(predited_label,predited_label2,predited_label3,label,label2,label3,0.3)

    if label.ndim == 2: label = label.argmax(1)
    correct_number = (predited_label.argmax(1) == label).sum().item()

    return loss, correct_number

def train_one_epoch(self, epoch):
    self.model.train()
    start = time.time()
    total_acc, total_count = 0, 0
    total_loss = 0
    # dpar = tqdm(enumerate(self.train_dataloader))
    for idx, (data, data2,data3, label, label2,label3) in enumerate(self.train_dataloader):
        start_time = time.time()
        loss, correct_number = self.train_step((data, data2,data3, label, label2,label3), idx)
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
# """


def main00(root, num_classes):
    # --------------params---------------------------
    model_name = "resnet18"
    pretrained = True
    weight_path = model_name + ".pth"  # 'weight.pth'
    log_file = (model_name + "_T" if pretrained else model_name) + ".csv"  # "log.csv"
    in_c = 3
    # num_classes = 5
    dropout = 0.0
    lr = 5e-4 if pretrained else 1e-3
    weight_decay = 5e-5 if pretrained else 1e-4
    gamma = 0.9
    epochs = 30 if pretrained else 50
    batch_size = 64 if pretrained else 32
    log_interval = 100
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = get_device()

    # --------------dataset---------------------------
    train_transforms, val_transforms = get_transforms()
    train_dataset = create_dataset(root, train_transforms, num_classes, None)
    val_dataset = create_dataset(root, val_transforms)
    train_dataset, val_dataset = get_train_val_dataset(train_dataset, val_dataset, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size, True)  # ,collate_fn=collate_fn
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    # --------------model---------------------------
    # model = create_model(pretrained, num_classes, model_name)
    model = test_model(pretrained, num_classes)
    model.to(device)
    load_model_weight(model, device, weight_path)

    flops, params = model_profile(model, torch.randn([1, 3, 224, 224]).to(device))
    with open(log_file, 'w') as fp:
        fp.write("flops=%s,params=%s\n" % (flops, params))

    optim, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr, weight_decay)
    criterion = get_criterion(reduction="sum", mode="CrossEntropyLoss")
    # criterion = get_criterion(reduction="sum", mode="labelsmooth")

    # --------------train---------------------------
    # fit(model, optim, scheduler, criterion, train_dataloader, val_dataloader,
    #     device, epochs, log_interval, weight_path, log_file)

    trainer = Trainer(model, optim, scheduler, criterion, train_dataloader, val_dataloader, device, log_interval)
    # trainer.train_step = lambda batch, step: train_step(trainer, batch, step)
    # trainer.train_one_epoch = lambda epoch: train_one_epoch(trainer, epoch)
    trainer.fit(epochs, weight_path, log_file)


def main(root, num_classes):
    """
    # 如果出现过拟合(修改以下设置 否则不做修改)
    1、增大 weight_decay
    2、使用 dropout、dropblock、droppath
    3、数据增强 get_transforms(mode) mode=v2,v3,v4, get_transformsv2()
    3.1、数据增强 create_dataset(mode) mode = mixup", "mosaictwo", "mosaicfour"
    4、训练策略：
        数据选择：create_dataset(mode) mode = "selecterr","oneshot","triplet"
        模型蒸馏 + 逐步放大 size + 正则
    5、 修改 criterion、optimizer, lr_scheduler
    """

    # --------------params---------------------------
    model_name = "resnet18"
    pretrained = True
    weight_path = model_name + ".pth"  # 'weight.pth'
    log_file = (model_name + "_T" if pretrained else model_name) + ".csv"  # "log.csv"
    in_c = 3
    # num_classes = 5
    dropout = 0.0
    lr = 5e-4 if pretrained else 1e-3
    weight_decay = 5e-5 if pretrained else 1e-4
    gamma = 0.9
    epochs = 30 if pretrained else 50
    batch_size = 64 if pretrained else 32
    log_interval = 100
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = get_device()

    # --------------dataset---------------------------
    train_transforms, val_transforms = get_transforms()
    train_dataset = create_dataset(root, train_transforms, num_classes, "oneshot")
    val_dataset = create_dataset(root, val_transforms)
    train_dataset, val_dataset = get_train_val_dataset(train_dataset, val_dataset, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size, True)  # ,collate_fn=collate_fn
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    # --------------model---------------------------
    # model = create_model(pretrained, num_classes, model_name)
    model = test_model(pretrained, num_classes)
    model.to(device)
    load_model_weight(model, device, weight_path)

    flops, params = model_profile(model, torch.randn([1, 3, 224, 224]).to(device))
    with open(log_file, 'w') as fp:
        fp.write("flops=%s,params=%s\n" % (flops, params))

    optim, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr, weight_decay)
    criterion = get_criterion(reduction="sum", mode="CrossEntropyLoss")
    # criterion = get_criterion(reduction="sum", mode="labelsmooth")

    # --------------train---------------------------
    # fit(model, optim, scheduler, criterion, train_dataloader, val_dataloader,
    #     device, epochs, log_interval, weight_path, log_file)

    trainer = Trainer(model, optim, scheduler, criterion, train_dataloader, val_dataloader, device, log_interval)
    trainer.train_step = lambda batch, step: train_step(trainer, batch, step)
    trainer.train_one_epoch = lambda epoch: train_one_epoch(trainer, epoch)

    trainer.fit(epochs, weight_path, log_file)


if __name__ == "__main__":
    """
    root = "../input/defect01/H0025_CTG_DEP_200914"
    num_classes = 9
    """
    root = r"D:\data\flower_photos"
    num_classes = 5
    # """
    main(root, num_classes)
