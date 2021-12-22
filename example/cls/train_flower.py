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
from toolcv.cls.tools import fit, dtrain, devaluate
from toolcv.cls.tools import FlowerPhotosDataset
from toolcv.cls.dataAugment import get_transforms

model = Simple('resnet18', True, 5, 0.3, 5)

data_path = r"D:\data\flower_photos"
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
train_ratio = 0.8
train_transforms,test_transforms = get_transforms()

train_dataset = FlowerPhotosDataset(data_path, classes, train_transforms)
test_dataset = FlowerPhotosDataset(data_path, classes, test_transforms)

num_datas = len(train_dataset)
num_train = int(train_ratio * num_datas)
indices = torch.randperm(num_datas).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
test_dataset = torch.utils.data.Subset(test_dataset, indices[num_train:])


fit(train_dataset,test_dataset,model,epochs=3,log_interval=50)