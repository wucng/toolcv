import time
import torch
import math
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torchvision.datasets import ImageFolder
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation   #导入负责绘制动画的接口

from toolcv.tools.utils import glob_format


class BaseDataset(Dataset):
    def __init__(self, root="", classes=[], transforms=None):
        self.root = root
        self.classes = classes
        self.transforms = transforms

    def __len__(self):
        raise ('error')

    def _load(self, idx):
        raise ('error')

    def __getitem__(self, idx):
        raise ('error')

    def get_height_and_width(self, idx):
        img = np.array(self._load(idx)[0])
        return img.shape[:2]

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels


class FlowerPhotosDataset(BaseDataset):
    """
    5类鲜花数据集
    https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

    cross_entropy(output,target)
    """

    def __init__(self, root="", classes=[], transforms=None):
        super().__init__(root, classes, transforms)

        self.paths = glob_format(root)

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = Image.open(path).convert("RGB")

        return image, label

    def __getitem__(self, idx):
        image, label = self._load(idx)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


class FlowerPhotosDatasetV2(FlowerPhotosDataset):
    """
    %90 选择正确标签 10% 选择错误标签
    ture_class*cross_entropy(output,target)-(1-ture_class)*cross_entropy(output,target)
    """

    def __init__(self, root="", classes=[], transforms=None, p=0.9):
        super().__init__(root, classes, transforms)
        self.p = p

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = Image.open(path).convert("RGB")

        true_class = 1
        if np.random.random() > self.p:
            # 选择错误的label
            true_class = 0
            while True:
                _label = np.random.choice(len(self.classes))
                if _label != label:
                    break
            label = _label

        return image, label, true_class

    def __getitem__(self, idx):
        image, label, true_class = self._load(idx)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, true_class

    @staticmethod
    def collate_fn(batch):
        images, labels, true_class = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        true_class = torch.as_tensor(true_class)

        return images, labels, true_class


class FlowerPhotosDatasetV3(FlowerPhotosDataset):
    """
    加入测试样本 logit  loss = 0.1*cross_entropy(logit,logit.detach().argmax(-1))

    cross_entropy(output,target)+0.1*cross_entropy(logit,logit.detach().argmax(-1))
    """

    def __init__(self, root="", test_root="", classes=[], transforms=None):
        super().__init__(root, classes, transforms)
        self.test_paths = glob_format(test_root)

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = Image.open(path).convert("RGB")

        test_image = Image.open(self.test_paths[idx % len(self.test_paths)]).convert("RGB")

        return image, label, test_image

    def __getitem__(self, idx):
        image, label, test_image = self._load(idx)
        if self.transforms is not None:
            image = self.transforms(image)
            test_image = self.transforms(test_image)

        return image, label, test_image

    @staticmethod
    def collate_fn(batch):
        images, labels, test_images = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        test_images = torch.stack(test_images, dim=0)

        return images, labels, test_images


class FlowerPhotosDatasetV4(FlowerPhotosDataset):
    """
    1、%90 选择正确标签 10% 选择错误标签
    2、加入测试样本 logit  loss = 0.1*cross_entropy(logit,logit.detach().argmax(-1))

    true_class*cross_entropy(output,target)-(1-true_class)*cross_entropy(output,target)+
    0.1*cross_entropy(logit,logit.detach().argmax(-1))
    """

    def __init__(self, root="", test_root="", classes=[], transforms=None, p=0.9):
        super().__init__(root, classes, transforms)
        self.test_paths = glob_format(test_root)
        self.p = p

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = Image.open(path).convert("RGB")

        true_class = 1
        if np.random.random() > self.p:
            # 选择错误的label
            true_class = 0
            while True:
                _label = np.random.choice(len(self.classes))
                if _label != label:
                    break
            label = _label

        test_image = Image.open(self.test_paths[idx % len(self.test_paths)]).convert("RGB")

        return image, label, true_class, test_image

    def __getitem__(self, idx):
        image, label, true_class, test_image = self._load(idx)
        if self.transforms is not None:
            image = self.transforms(image)
            test_image = self.transforms(test_image)

        return image, label, true_class, test_image

    @staticmethod
    def collate_fn(batch):
        images, labels, true_class, test_images = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        true_class = torch.as_tensor(true_class)
        test_images = torch.stack(test_images, dim=0)

        return images, labels, true_class, test_images


def oneshot_loss(outputs1, outputs2, same_classes, margin=2, reduction='sum'):
    # 结合one-shot-learning loss : 类内小 内间大
    # # same_classes 不同为0 相同为1
    euclidean_distance = torch.nn.functional.pairwise_distance(outputs1, outputs2)
    loss_contrastive = (same_classes) * torch.pow(euclidean_distance, 2) + \
                       (1 - same_classes) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    if reduction == 'sum':
        loss_contrastive = loss_contrastive.sum()
    elif reduction == 'mean':
        loss_contrastive = loss_contrastive.mean()

    return loss_contrastive


# one-shot-learning
class FlowerPhotosDatasetSiamese(FlowerPhotosDataset):
    """
    def oneshot_loss(outputs1,outputs2,label, margin=2):
        # 结合one-shot-learning loss : 类内小 内间大
        # # label 不同为0 相同为1
        euclidean_distance = torch.nn.functional.pairwise_distance(outputs1,outputs2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive * loss_contrastive.detach()
    """

    def __init__(self, root="", classes=[], transforms=None):
        super().__init__(root, classes, transforms)

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = Image.open(path).convert("RGB")
        # 使得50%的训练数据为一对图像属于同一类别
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # 循环直到一对图像属于同一类别
                idx = np.random.choice(self.__len__())
                _path = self.paths[idx]
                _label = self.classes.index(os.path.basename(os.path.dirname(_path)))
                if _label == label:
                    break
        else:
            while True:
                # 循环直到一对图像属于不同的类别
                idx = np.random.choice(self.__len__())
                _path = self.paths[idx]
                _label = self.classes.index(os.path.basename(os.path.dirname(_path)))
                if _label != label:
                    break
        _image = Image.open(_path).convert("RGB")

        return image, label, _image, _label, should_get_same_class  # 1 相同 ；0 不同

    def __getitem__(self, idx):
        image, label, _image, _label, should_get_same_class = self._load(idx)
        if self.transforms is not None:
            image = self.transforms(image)
            _image = self.transforms(_image)

        return image, label, _image, _label, should_get_same_class

    @staticmethod
    def collate_fn(batch):
        images, labels, _images, _labels, same_classes = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        _images = torch.stack(_images, dim=0)
        _labels = torch.as_tensor(_labels)
        same_classes = torch.as_tensor(same_classes)

        return images, labels, _images, _labels, same_classes


# -----------------------------------------------------------------------------

def dtrain(model, optimizer, dataloader, criterion, device, epoch, log_interval=500):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    total_loss = 0

    for idx, (data, label) in enumerate(dataloader):
        label = label.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        # model.zero_grad()
        predited_label = model(data)
        loss = criterion(predited_label, label)
        total_loss += loss.item()
        loss = loss / len(predited_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        # total_acc += (predited_label.round() == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

    return total_acc / total_count, total_loss / total_count


def devaluate(model, dataloader, criterion, device):
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
            # total_acc += (predited_label.round() == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, total_loss / total_count


class History():
    def __init__(self):
        self.epoch = []
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        self.fig = fig
        self.ax = ax

    # 打印训练结果信息
    # @staticmethod
    def show_final_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(self.epoch, self.history["loss"], label="Train loss")
        ax[0].plot(self.epoch, self.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(self.epoch, self.history["acc"], label="Train acc")
        ax[1].plot(self.epoch, self.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        ax[0].grid()
        ax[1].grid()
        plt.show()

    def show_dynamic_history(self):
        self.ax[0].cla()  # 清除键
        self.ax[1].cla()  # 清除键

        self.ax[0].set_title('loss')
        self.ax[0].plot(self.epoch, self.history["loss"], label="Train loss")
        self.ax[0].plot(self.epoch, self.history["val_loss"], label="Validation loss")
        self.ax[1].set_title('acc')
        self.ax[1].plot(self.epoch, self.history["acc"], label="Train acc")
        self.ax[1].plot(self.epoch, self.history["val_acc"], label="Validation acc")
        self.ax[0].legend()
        self.ax[1].legend()

        self.ax[0].grid()
        self.ax[1].grid()

        plt.pause(0.1)


def fit(train_data, test_data, model, batch_size=64, epochs=10,
        lr=5e-4, device="cuda:0", lrf=0.1, log_interval=500, save_path="./output",
        train=None, evaluate=None, draw=True):
    if train is None:
        train = dtrain
    if evaluate is None:
        evaluate = devaluate

    if draw:
        history = History()

    if not os.path.exists(save_path): os.makedirs(save_path)
    # train_data = TensorDataset(torch.tensor(train_images).float(),torch.tensor(train_labels).long())
    # test_data = TensorDataset(torch.tensor(test_images).float(),torch.tensor(test_labels).long())
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    # ,drop_last=True 不足一个batch 丢弃

    model = model.to(device)
    if os.path.exists(os.path.join(save_path, 'weight.pth')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'weight.pth')))
        print("--------load weight successful!!---------------")

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.AdamW([parm for parm in model.parameters() if parm.requires_grad], lr=lr, weight_decay=5e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    total_acc = None
    for epoch in range(epochs):
        epoch_start_time = time.time()
        acc_train, loss_train = train(model, optimizer, train_dataloader, criterion, device, epoch, log_interval)
        acc_val, loss_val = evaluate(model, valid_dataloader, criterion, device)
        if total_acc is not None and total_acc > acc_val:
            scheduler.step()
        else:
            total_acc = acc_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               acc_val))

        print('-' * 59)

        print("acc_train:%.3f loss_train:%.3f acc_val:%.3f loss_val:%.3f" % (
            acc_train, loss_train, acc_val, loss_val))

        print('-' * 59)

        torch.save(model.state_dict(), os.path.join(save_path, "weight.pth"))

        if draw:
            history.epoch.append(epoch)
            history.history['loss'].append(loss_train)
            history.history['val_loss'].append(loss_val)
            history.history['acc'].append(acc_train)
            history.history['val_acc'].append(acc_val)

            history.show_dynamic_history()

    if draw:
        history.show_final_history()
