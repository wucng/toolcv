import torch
import os

from toolcv.api.pytorch_lightning.net import BaseModel
from toolcv.api.pytorch_lightning.data import LitDataModule, load_dataloader
from toolcv.api.pytorch_lightning.utils import fit

batch_size = 32
epochs = 10
lr = 1e-3
lrf = 0.1
warpstep = 1000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print_step = 100
use_amp = True
accumulate = 4  # max(round(64 / batch_size), 1)
checkpoint_path = 'best-epoch-4-acc-0.983-loss-0.062.pth'

dm = LitDataModule(r'D:\data\mnist', batch_size=batch_size)
dm.setup('fit')
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()

model = BaseModel(None, epochs, warpstep, lr, lrf)
_model = model.model.to(device)
[optimizer], [scheduler] = model.configure_optimizers()

criterion = None
fit(_model, optimizer, None, None, criterion, checkpoint_path, scheduler, train_dataloader, val_dataloader, epochs,
    print_step, batch_size, device, use_amp, accumulate, 0, 0.1, 'none', True, 1000, 'v2', True)
