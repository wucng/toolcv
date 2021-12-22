import torch
from torch import nn
from torch.utils.data import random_split
import torchvision.transforms as T
from torchvision.models.resnet import resnet18
import os

from toolcv.api.pytorch_lightning.data import FlowerPhotosDataset
from toolcv.api.pytorch_lightning.net import BaseModelV2
from toolcv.api.pytorch_lightning.data import BaseDataModule
from toolcv.api.pytorch_lightning.utils import load_callbacks,load_logger,set_seed
from toolcv.api.pytorch_lightning.train import fit,test

training = False
set_seed(100)
dir_data = r"D:\data\flower_photos"
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
num_classes = len(classes)
batch_size = 32
epochs = 5
device = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint_path = r"logs\default\version_1\checkpoints\last.ckpt"

transforms = T.Compose([T.Resize((224, 224)), T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = FlowerPhotosDataset(dir_data,classes,transforms,mode='none', label_smooth=False)
# dataset.show(20, 'plt')

nums = len(dataset)
nums_train = int(nums * 0.9)
train_dataset, val_dataset = random_split(dataset, [nums_train, nums - nums_train])

dm = BaseDataModule(train_dataset,val_dataset,val_dataset,batch_size,use_cuda=False)

_model = resnet18(True)
_model.fc = nn.Linear(_model.inplanes, num_classes)

model = BaseModelV2(_model,epochs=epochs,warpstep=1000,lr=5e-4)
if training:
    if os.path.exists(checkpoint_path):
        # model.load_from_checkpoint(checkpoint_path)
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['state_dict'])
        print("------load weight successful!!---------")
    config = {"callbacks":load_callbacks(),"logger":load_logger()}
    fit(model,epochs,config,datamodule=dm)
else:
    test(model,checkpoint_path,None,dm,device,1)
