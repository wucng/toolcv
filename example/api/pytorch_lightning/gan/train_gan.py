import torch
import os

from toolcv.api.pytorch_lightning.net import GANModel
from toolcv.api.pytorch_lightning.data import LitDataModule,load_dataloader
from toolcv.api.pytorch_lightning.utils import load_callbacks,load_logger
from toolcv.api.pytorch_lightning.train import fit,test

training = False
batch_size = 32
epochs = 5
lr = 1e-3
lrf = 0.1
warpstep = 1000
checkpoint_path = r"logs\default\version_4\checkpoints\best-epoch=05-val_acc=0.984.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dm = LitDataModule(r'D:\data\mnist',batch_size=batch_size)
model = GANModel(None,None,epochs,warpstep,lr,lrf)

config={"stochastic_weight_avg":False,"gradient_clip_val":0.0}
fit(model,epochs,config,datamodule=dm)