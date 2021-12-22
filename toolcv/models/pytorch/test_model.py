import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms as T
import time
import os

from torchvision.models import resnet18
from cnn_tools.SE_Inception_resnet_v2 import SE_Inception_resnet_v2

in_c = 3
num_classes = 9
dropout = 0.0
lr = 2e-3
weight_decay = 5e-5
gamma = 0.9
epochs = 20
batch_size = 32
log_interval = 100

root = "/opt/ml/input/data/images_data/H0025_CTG_DEP_200914"
transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])
dataset = ImageFolder(root, transforms)
nums = len(dataset)
nums_train = int(nums * 0.8)
nums_val = nums - nums_train
train_data, val_data = random_split(dataset, [nums_train, nums_val])
train_dataloader = DataLoader(train_data, batch_size, True)
val_dataloader = DataLoader(val_data, batch_size, False)

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = SE_Inception_resnet_v2().build_model(in_c, dropout, num_classes).to(device)
"""
model = resnet18(True)
model.fc = nn.Linear(model.inplanes, num_classes)
model.to(device)
for param in model.parameters():
    param.requires_grad_(False)

for param in model.fc.parameters():
    param.requires_grad_(True)
"""

params = model.parameters()
params = [param for param in params if param.requires_grad]
optim = torch.optim.Adam(params, lr, weight_decay=weight_decay)
# optim = torch.optim.SGD(model.parameters(),lr,weight_decay=weight_decay,momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

try:
    model.load_state_dict(torch.load('weight.pth', device))
    print("---------load weight successful-----------")
except:
    pass


def train_step(model, optimizer, dataloader, criterion, device, log_interval=100):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    total_loss = 0

    for idx, (data, label) in enumerate(dataloader):
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
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            # total_acc, total_count = 0, 0
            start_time = time.time()

    return total_acc / total_count, total_loss / total_count


def test_step(model, dataloader, criterion, device):
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
    return total_acc / total_count, total_loss / total_count


_test_loss = 0
for epoch in range(epochs):
    print(epoch + 1, "开始！！")
    epoch_st = time.time()
    # 训练循环
    train_accuracy, train_loss = train_step(model, optim, train_dataloader, criterion, device, log_interval)
    test_accuracy, test_loss = test_step(model, val_dataloader, criterion, device)

    if test_loss > _test_loss:
        scheduler.step()
        _test_loss = test_loss

    # if epoch % 50 == 0:
    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                "Test Accuracy: {}")
    line = template.format(epoch + 1, train_loss,
                           train_accuracy * 100, test_loss,
                           test_accuracy * 100)
    print("-" * 50)
    print(line)
    print("-" * 50)
    print("epoch", epoch + 1, "::", time.time() - epoch_st)
    print("-" * 50)

    torch.save(model.state_dict(), 'weight.pth')
