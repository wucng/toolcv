import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torchvision.datasets import ImageFolder
import time

from toolcv.cls.net import Simple,WNet,MultiStageUnionNet
from toolcv.cls.data import load_mnist
from toolcv.cls.tools import fit#,train,evaluate

"""
model = nn.Sequential(
    nn.Conv2d(1, 64, 3, 2, 1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),

    nn.Conv2d(64, 128, 3, 2, 1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),

    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(128, 10)
)

train_images, train_labels, test_images, test_labels = load_mnist(mode='fashion-mnist')
train_images = torch.tensor(train_images / 255.).float().unsqueeze(1)
test_images = torch.tensor(test_images / 255.).float().unsqueeze(1)
train_data = TensorDataset(train_images, torch.tensor(train_labels).long())
test_data = TensorDataset(test_images, torch.tensor(test_labels).long())
fit(train_data, test_data, model, epochs=5, log_interval=50)
# """
"""
model = Simple("resnet18",True,10,0.5,5)

train_images, train_labels, test_images, test_labels = load_mnist(mode='fashion-mnist')

train_images = F.interpolate(torch.tensor(train_images / 255.).float().unsqueeze(1),
                             size=(64,64),mode="bilinear",align_corners=True)
test_images = F.interpolate(torch.tensor(test_images / 255.).float().unsqueeze(1),
                             size=(64,64),mode="bilinear",align_corners=True)
train_data = TensorDataset(train_images, torch.tensor(train_labels).long())
test_data = TensorDataset(test_images, torch.tensor(test_labels).long())

fit(train_data, test_data, model, epochs=5, log_interval=50)
# """

def train(model, optimizer, dataloader, criterion, device, epoch, log_interval=500):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    total_loss = 0

    for idx, (data, label) in enumerate(dataloader):
        data = F.interpolate(data,size=(224,224),mode="bilinear",align_corners=True)
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            data = F.interpolate(data, size=(224, 224), mode="bilinear", align_corners=True)
            label = label.to(device)
            data = data.to(device)
            predited_label = model(data)
            loss = criterion(predited_label, label)
            total_loss += loss.item()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            # total_acc += (predited_label.round() == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, total_loss / total_count


# model = WNet(1,1,10,0.5)

model = MultiStageUnionNet(1,10)

train_images, train_labels, test_images, test_labels = load_mnist(mode='fashion-mnist')

train_images = torch.tensor(train_images / 255.).float().unsqueeze(1)
test_images = torch.tensor(test_images / 255.).float().unsqueeze(1)
train_data = TensorDataset(train_images, torch.tensor(train_labels).long())
test_data = TensorDataset(test_images, torch.tensor(test_labels).long())

fit(train_data, test_data, model,4, epochs=5, log_interval=50,train=train,evaluate=evaluate)