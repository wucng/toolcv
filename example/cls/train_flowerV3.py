"""
5类鲜花数据集
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
# from torchvision.datasets import ImageFolder
import time

from toolcv.cls.net import Simple  # ,WNet,MultiStageUnionNet
# from toolcv.cls.data import load_mnist
from toolcv.cls.tools import fit, devaluate  # , dtrain, devaluate
from toolcv.cls.tools import FlowerPhotosDataset, FlowerPhotosDatasetV3
from toolcv.cls.dataAugment import get_transforms


def train(model, optimizer, dataloader, criterion, device, epoch, log_interval=500):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    total_loss = 0

    for idx, (data, label, test_image) in enumerate(dataloader):
        label = label.to(device)
        data = data.to(device)
        test_image = test_image.to(device)
        optimizer.zero_grad()
        # model.zero_grad()
        predited_label = model(data)
        test_label = model(test_image)
        # loss = criterion(predited_label, label)
        loss = criterion(predited_label, label) + 0.1 * criterion(test_label, test_label.detach().argmax(1))

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


model = Simple('resnet18', True, 5, 0.3, 5)

data_path = r"D:\data\flower_photos"
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
train_ratio = 0.8
train_transforms, test_transforms = get_transforms()

train_dataset = FlowerPhotosDatasetV3(data_path, data_path, classes, train_transforms)
test_dataset = FlowerPhotosDataset(data_path, classes, test_transforms)

num_datas = len(train_dataset)
num_train = int(train_ratio * num_datas)
indices = torch.randperm(num_datas).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
test_dataset = torch.utils.data.Subset(test_dataset, indices[num_train:])

fit(train_dataset, test_dataset, model, epochs=3, log_interval=20, train=train, evaluate=devaluate)
