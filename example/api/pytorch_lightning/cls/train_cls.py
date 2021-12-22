import torch
import os

from toolcv.api.pytorch_lightning.net import BaseModel
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
model = BaseModel(None,epochs,warpstep,lr,lrf)

# model.save(mode=2,shape=[batch_size,28*28])
# model.save_onnx(shape=[batch_size,28*28])
# model.to_torchscript("model.jit")
# model.to_onnx("model.onnx",torch.rand([batch_size,28*28]),verbose=True)

if training:
    if os.path.exists(checkpoint_path):
        model.load_from_checkpoint(checkpoint_path)
        print("------load weight successful!!---------")
    config = {"callbacks":load_callbacks(),"logger":load_logger(),
              # "auto_lr_find":True,
              # "auto_scale_batch_size":True
              }
    fit(model,epochs,config,datamodule=dm)
    # model.save(0) # 'weight.pth'
else:
    test(model,checkpoint_path,None,dm,device,1)

    # state_dict = torch.load('weight.pth')
    # model.load_state_dict({"model."+k:v for k,v in state_dict.items()})
    # test(model,None,None,dm,device,1)
