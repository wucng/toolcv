from toolcv.api.mobileNeXt.codebase.scheduler.plateau_lr import PlateauLRScheduler
from toolcv.api.mobileNeXt.codebase.scheduler.step_lr import StepLRScheduler
from toolcv.api.mobileNeXt.codebase.optim.radam import RAdam, PlainRAdam

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import torch
from torch import nn
import numpy as np

model = nn.Sequential(nn.Conv2d(3,10,3))
# optimizer = RAdam(model.parameters(), 0.1, weight_decay=5e-6)
optimizer = AdamW(model.parameters(), 0.1, weight_decay=5e-6)
scheduler = PlateauLRScheduler(optimizer,warmup_updates=10,warmup_lr_init=0.1)
# scheduler = StepLRScheduler(optimizer,5,0.5)
# scheduler = StepLR(optimizer,5,0.5)

for i in range(200):
   scheduler.step(i,np.random.random())
   learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
   print(learning_rate)