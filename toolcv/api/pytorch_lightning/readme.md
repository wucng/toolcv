- https://pytorch-lightning.readthedocs.io/en/latest/
- https://github.com/PyTorchLightning/pytorch-lightning
- https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
- https://zhuanlan.zhihu.com/p/353985363
- https://zhuanlan.zhihu.com/p/319810661
- https://github.com/miracleyoo/pytorch-lightning-template

# Install
```python
pip install pytorch-lightning -i https://pypi.doubanio.com/simple
```

# AutoEncoder
```python
# - https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

# Define a LightningModule (nn.Module subclass)
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Train
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

# classify-simple
```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

class LitClsModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(28 * 28, 256),nn.BatchNorm1d(256),nn.ReLU(),
                                   nn.Linear(256, 512),nn.BatchNorm1d(512),nn.ReLU(),
                                   nn.Linear(512,10))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train, val = random_split(dataset, [55000, 5000])
train = MNIST(os.getcwd(),train=True, download=True, transform=transforms.ToTensor())
val = MNIST(os.getcwd(),train=False, download=True, transform=transforms.ToTensor())


model = LitClsModel()
trainer = pl.Trainer(max_epochs=5,gpus=[0],log_every_n_steps=50)
trainer.fit(model, DataLoader(train,32,True), DataLoader(val,32,False))
```

# classify
```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import time
import math
import logging
logging.basicConfig(level=logging.INFO)  # ??????????????????

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:min(1.0,x/warmup_iters))
    # """

class LitClsModel(pl.LightningModule):
    def __init__(self,epochs,warpstep):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(28 * 28, 256),nn.BatchNorm1d(256),nn.ReLU(),
                                   nn.Linear(256, 512),nn.BatchNorm1d(512),nn.ReLU(),
                                   nn.Linear(512,10))
        self.epochs = epochs
        self.warpstep = warpstep
        # self.batch_size = 1
        # self.lr = 0
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        """??????step?????????"""
        # ??????????????? model.train()
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss)
        
        # warpup 
        if self.current_epoch == 0:
            self.warmup_lr_scheduler.step()

        return loss
    
    # def training_step_end(self, *args, **kwargs):
    #     # warpup 
        # if self.current_epoch == 0:
        #     self.warmup_lr_scheduler.step()
    
   # def on_train_epoch_end(self, unused=None) -> None:
   #     self.lr_schedulers().step() # ???????????????

    def on_validation_epoch_start(self) -> None:
        self.start = time.time()
    
    def on_validation_epoch_end(self) -> None:
        self.end = time.time()
        cost_time = self.end - self.start
        self.log('cost_time', cost_time)
        print('epoch:%d cost time:%.5f'%(self.current_epoch,cost_time))
        
    def validation_step(self, batch, batch_idx):
        """??????step?????????"""
        # ??????????????? model.eval()
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss)
        
        acc = (out.argmax(1)==y).sum()/out.size(0)
        self.log('val_acc', acc)
        return {'loss': loss, 'acc': acc}
    
    def validation_step_end(self, batch_parts):
        """validation_step ???????????? ???????????????"""
        loss = batch_parts['loss'].item()
        acc = batch_parts['acc'].item()
        return {'loss': loss, 'acc': acc}
    
    def validation_epoch_end(self, validation_step_outputs):
        """??????epoch?????????"""
        loss_list = []
        acc_list = []
        for out in validation_step_outputs: 
            loss_list.append(out['loss'])
            acc_list.append(out['acc'])
        
        mean_loss = np.mean(loss_list)
        mean_acc = np.mean(acc_list)
        self.log('val_acc_epoch', mean_acc)
        self.log('val_loss_epoch', mean_loss)
        learning_rate = self.optimizers().state_dict()['param_groups'][0]['lr']
        self.log('learning_rate', learning_rate)
        print("epoch:%d acc:%.3f loss:%.3f lr:%.5f"%(self.current_epoch,mean_acc,mean_loss,learning_rate))        

    def configure_optimizers(self):
        # self.parameters()
        optimizer = torch.optim.Adam([param for param in self.parameters() if param.requires_grad], 
                                        lr=1e-3,weight_decay=5e-5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2000)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)
        
        lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        self.warmup_lr_scheduler = warmup_lr_scheduler(optimizer, self.warpstep, 1/self.warpstep)
        
        return [optimizer],[scheduler]

# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train, val = random_split(dataset, [55000, 5000])
train = MNIST(os.getcwd(),train=True, download=True, transform=transforms.ToTensor())
val = MNIST(os.getcwd(),train=False, download=True, transform=transforms.ToTensor())

# Or use the same format as others
tb_logger = pl_loggers.TensorBoardLogger('logs/')
# One Logger
# comet_logger = pl_loggers.CometLogger(save_dir='logs/')
epochs=5
batch_size = 32
warpstep = len(train)//batch_size//2
model = LitClsModel(epochs,warpstep)
#trainer = pl.Trainer(logger=tb_logger,callbacks=[ModelCheckpoint(dirpath="./output",monitor='val_loss'),
#                                EarlyStopping('val_loss'),LearningRateMonitor(logging_interval='step')],
#                        max_epochs=5,gpus=[0],log_every_n_steps=50)

# accumulate every 4 batches (effective batch size is batch*4)
# trainer = Trainer(accumulate_grad_batches=4)
# no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
# trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

# ???????????? tensorboard 
trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor='val_acc'),EarlyStopping(monitor='val_acc')],
                     max_epochs=epochs,gpus=[0],
                     log_every_n_steps=50,
                     gradient_clip_val=0.1, # ????????????
                     precision=16, # ????????? 16,32,64 ??????????????????
                     # amp_backend='apex', # using NVIDIA Apex ?????????????????? ???????????????https://github.com/NVIDIA/apex#linux
                     accumulate_grad_batches=4, # ???4???batch?????????????????????????????????????????????????????????batch??????????????????
                     stochastic_weight_avg=True,
                     # auto_scale_batch_size='binsearch', # ???????????????????????????batch_size (# run batch size scaling, result overrides hparams.batch_size)
                     # auto_lr_find=True # ???????????????????????????????????????
)

# call tune to find the batch size
# trainer.tune(model) # ????????? auto_scale_batch_size???auto_lr_find ??????????????????????????? model????????? dataloader???????????????????????????????????? 

trainer.fit(model, DataLoader(train,batch_size,True), DataLoader(val,batch_size,False))

# torch.save(model.state_dict(),"weight.pth")
# torch.save(model,"model.pth")

# ????????????
# load the ckpt
# model = LitClsModel.load_from_checkpoint(path)
# or
# load the ckpt
# ckpt = torch.load('path/to/checkpoint.ckpt')
# equivalent to the above
# model = LitClsModel()
# model.load_state_dict(ckpt['state_dict'])
```

# ?????????DataModule
- https://zhuanlan.zhihu.com/p/319810661

```python
class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        ...blablabla...
    def setup(self, stage):
        # ?????????????????????????????????GPU?????????????????????, stage ?????????????????????????????????
        if stage == 'fit' or stage is None:
            self.train_dataset = DCKDataset(self.train_file_path, self.train_file_num, transform=None)
            self.val_dataset = DCKDataset(self.val_file_path, self.val_file_num, transform=None)
        if stage == 'test' or stage is None:
            self.test_dataset = DCKDataset(self.test_file_path, self.test_file_num, transform=None)
    def prepare_data(self):
        # ?????????????????????????????????????????????????????????cuda:0 ??????????????????
        pass
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True)
```

```python
dm = MyDataModule(args)
if not is_predict:# ??????
    # ?????????????????????callback?????????????????????
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    # ????????????
    model = MyModel()
    # ??????logger
    logger = TensorBoardLogger('log_dir', name='test_PL')
    # ????????????????????????????????????
    dm.setup('fit')
    # ??????trainer
    trainer = pl.Trainer(gpus=gpu, logger=logger, callbacks=[checkpoint_callback]);
    # ????????????
    trainer.fit(dck, datamodule=dm)
else:
    # ????????????
    dm.setup('test')
    # ????????????
    model = MyModel.load_from_checkpoint(checkpoint_path='trained_model.ckpt')
    # ??????trainer?????????
    trainer = pl.Trainer(gpus=1, precision=16, limit_test_batches=0.05)
    trainer.test(model=model, datamodule=dm)
```

# ?????????????????????
```python
# ----------------------------------
# torchscript
# ----------------------------------
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
os.path.isfile("model.pt")

# ----------------------------------
# onnx
# ----------------------------------
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
     autoencoder = LitAutoEncoder()
     input_sample = torch.randn((1, 28 * 28))
     autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
     os.path.isfile(tmpfile.name)
```

```python
# torch.save(model.state_dict(),"weight.pth")
# torch.save(model,"model.pth")

# ????????????
# load the ckpt
# model = LitClsModel.load_from_checkpoint(path)
# or
# load the ckpt
# ckpt = torch.load('path/to/checkpoint.ckpt')
# equivalent to the above
# model = LitClsModel()
# model.load_state_dict(ckpt['state_dict'])
```

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# saves checkpoints to 'my/path/' at every epoch
checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
trainer = Trainer(callbacks=[checkpoint_callback])

# save epoch and val_loss in name
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='my/path/', filename='sample-mnist-{epoch:02d}-{val_loss:.2f}')
```

## ?????????????????????
```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
trainer = Trainer(callbacks=[checkpoint_callback])
model = ...
trainer.fit(model)
# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????apply?????????????????????
checkpoint_callback.best_model_path
```

## ??????????????????
```python
from collections import deque
import os
# ??????????????????
self.save_models = deque(maxlen=3)
# ?????????self ?????????????????????????????????pl.LightningModule???????????????training_step()????????????
def manual_save_model(self):
    model_path = 'your_model_save_path_%s' % (your_loss)
    if len(self.save_models) >= 3:
        # ??????????????????????????????????????????????????????????????????
        old_model = self.save_models.popleft()
        if os.path.exists(old_model):
            os.remove(old_model)
    # ????????????
    self.trainer.save_checkpoint(model_path)
    # ??????????????????????????????????????????
    self.save_models.append(model_path)
```
````python
model = MyLightningModule(hparams)
trainer.fit(model)
trainer.save_checkpoint("example.ckpt")
new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")
````

## ??????Checkpoint
```python
model = MyLightingModule.load_from_checkpoint(PATH)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model.eval()
y_hat = model(x)
```

## ???????????????Trainer
```python
model = LitModel()
trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
# ?????????????????????epoch???step???????????????????????????LR schedulers???????????????
# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model)
```

# training_step
```python
def __init__(self):
    self.automatic_optimization = False

def training_step(self, batch, batch_idx):
    # access your optimizers with use_pl_optimizer=False. Default is True
    opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

    loss_a = self.generator(batch)
    opt_a.zero_grad()
    # use `manual_backward()` instead of `loss.backward` to automate half precision, etc...
    self.manual_backward(loss_a)
    opt_a.step()

    loss_b = self.discriminator(batch)
    opt_b.zero_grad()
    self.manual_backward(loss_b)
    opt_b.step()

def training_step(self, batch, batch_idx):
    x, y, z = batch
    out = self.encoder(x)
    loss = self.loss(out, x)
    return loss

# Multiple optimizers (e.g.: GANs)
def training_step(self, batch, batch_idx, optimizer_idx):
    if optimizer_idx == 0:
        # do training_step with encoder
    if optimizer_idx == 1:
        # do training_step with decoder
        
# Truncated back-propagation through time
def training_step(self, batch, batch_idx, hiddens):
    # hiddens are the hidden states from the previous truncated backprop step
    ...
    out, hiddens = self.lstm(data, hiddens)
    ...
    return {'loss': loss, 'hiddens': hiddens}
```

# configure_optimizers
```python
# most cases
def configure_optimizers(self):
    opt = Adam(self.parameters(), lr=1e-3)
    return opt

# multiple optimizer case (e.g.: GAN)
def configure_optimizers(self):
    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
    return generator_opt, disriminator_opt

# example with learning rate schedulers
def configure_optimizers(self):
    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
    return [generator_opt, disriminator_opt], [discriminator_sched]

# example with step-based learning rate schedulers
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
    gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
                 'interval': 'step'}  # called after each training step
    dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
    return [gen_opt, dis_opt], [gen_sched, dis_sched]

# example with optimizer frequencies
# see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
# https://arxiv.org/abs/1704.00028
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
    n_critic = 5
    return (
        {'optimizer': dis_opt, 'frequency': n_critic},
        {'optimizer': gen_opt, 'frequency': 1}
    )
```

# callbacks
- https://github.com/miracleyoo/pytorch-lightning-template/blob/master/classification/main.py
```python
import pytorch_lightning.callbacks as plc

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks
```

# trainer
- https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html