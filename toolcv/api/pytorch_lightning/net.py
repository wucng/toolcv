import torch
from torch import nn
import torch.nn.functional as F
import time
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import pytorch_lightning as pl
# from pytorch_lightning.metrics import functional as FM
from torchmetrics.functional.classification.accuracy import accuracy

from toolcv.api.pytorch_lightning.utils import warmup_lr_scheduler
from toolcv.api.pytorch_lightning.utils import training_step_yolov1, training_step_yolov2, training_step_yolov3, \
    training_step_ssd,training_step_ssdMS,training_step_fcos
from toolcv.network.net import _initParmas


class BaseModel(pl.LightningModule):
    def __init__(self, model: nn.Module = None, criterion=None, epochs: int = 10, warpstep: int = 0, lr=1e-3,
                 lrf: float = 0.1):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = nn.Sequential(nn.Linear(28 * 28, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                       nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                       nn.Linear(512, 10))

            _initParmas(self.model.modules())

        self.criterion = criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        self.epochs = epochs
        self.warpstep = warpstep
        self.lr = lr
        self.lrf = lrf

        self.batch_size = 1

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        """每个step后执行"""
        # 会自动调用 model.train()
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        out = self(x)
        loss = self.criterion(out, y)
        acc = accuracy(out, y)
        # self.log('train_loss', loss)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics)

        # if (batch_idx + 1) % 50 == 0:
        #     print("epoch:%d step:%d train_acc:%.3f train_loss:%.3f" % (
        #         self.current_epoch, batch_idx, acc.item(), loss.item()))

        # warpup
        if self.warpstep > 0:
            if self.current_epoch == 0:
                self.warmup_lr_scheduler.step()

        # return loss
        return {'loss': loss, 'acc': acc}

    # def training_step_end(self, *args, **kwargs):
    #     # warpup
    # if self.current_epoch == 0:
    #     self.warmup_lr_scheduler.step()

    # def on_train_epoch_end(self, unused=None) -> None:
    #     self.lr_schedulers().step() # 会自动调用

    def on_validation_epoch_start(self) -> None:
        self.start = time.time()

    def on_validation_epoch_end(self) -> None:
        self.end = time.time()
        cost_time = self.end - self.start
        self.log('cost_time', cost_time)
        # print('epoch:%d cost time:%.5f' % (self.current_epoch, cost_time))

    def validation_step(self, batch, batch_idx):
        """每个step后执行"""
        # 会自动调用 model.eval()
        x, y = batch
        # x = x.view(x.size(0), -1)
        out = self(x)
        loss = self.criterion(out, y)
        # self.log('val_loss', loss)

        # acc = (out.argmax(1) == y).sum() / out.size(0)
        acc = accuracy(out, y)
        # self.log('val_acc', acc)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

        return metrics

    def validation_step_end(self, batch_parts):
        """validation_step 执行完成 执行该函数"""
        loss = batch_parts['val_loss'].item()
        acc = batch_parts['val_acc'].item()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, validation_step_outputs):
        """每个epoch后执行"""
        loss_list = []
        acc_list = []
        for out in validation_step_outputs:
            loss_list.append(out['val_loss'])
            acc_list.append(out['val_acc'])

        mean_loss = np.mean(loss_list)
        mean_acc = np.mean(acc_list)
        self.log('val_acc_epoch', mean_acc)
        self.log('val_loss_epoch', mean_loss)
        learning_rate = self.optimizers().state_dict()['param_groups'][0]['lr']
        self.log('learning_rate', learning_rate)
        print("epoch:%d acc:%.3f loss:%.3f lr:%.5f" % (self.current_epoch, mean_acc, mean_loss, learning_rate))

    # def test_step(self, batch, batch_idx):
    #     metrics = self.validation_step(batch, batch_idx)
    #     metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
    #     self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        # 会自动调用 model.eval()
        x, y = batch
        # x = x.view(x.size(0), -1)
        out = self(x)

        return out.argmax(1).cpu().numpy(), y.cpu().numpy()

    def test_step_end(self, batch_parts):
        return batch_parts

    def test_epoch_end(self, outputs):
        preds = []
        trues = []
        for out in outputs:
            preds.extend(out[0])
            trues.extend(out[1])

        print("test acc:%.3f" % (sum(np.array(preds) == np.array(trues)) / len(preds)))

    def configure_optimizers(self):
        # self.parameters()
        optimizer = torch.optim.Adam([param for param in self.parameters() if param.requires_grad],
                                     lr=self.lr, weight_decay=5e-5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2000)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)

        # lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (
                1 - self.lrf) + self.lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        if self.warpstep > 0:
            self.warmup_lr_scheduler = warmup_lr_scheduler(optimizer, self.warpstep, 1 / self.warpstep)

        return [optimizer], [scheduler]

    def save(self, mode=0, shape=[1, 28 * 28]):
        if mode == 0:
            torch.save(self.model.state_dict(), "weight.pth")
        elif mode == 1:
            torch.save(self.model, "model.pth")
        elif mode == 2:
            self.model.to(self.device)
            x = torch.randn(shape).to(self.device)
            traced_script_module = torch.jit.trace(self.model, x)
            # 保存模型
            traced_script_module.save("model_jit.pth")

        print("------save model successful!!---------")

    def save_onnx(self, shape=[1, 28 * 28]):
        self.model.to(self.device)
        x = torch.randn(shape).to(self.device)
        torch.onnx.export(self.model, x, 'model.onnx', verbose=True)

        print("------save model successful!!---------")


norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_params(modules, lr=5e-4, weight_decay=5e-5, gamma=0.8):
    # params = [param for param in self.parameters() if param.requires_grad]
    params = []
    memo = set()
    for module in modules:  # self.modules()
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            _lr = lr
            _weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                _weight_decay = 0.0
            elif "bias" in key:
                _lr = lr
                _weight_decay = weight_decay

            elif 'backbone' in key:
                _lr = lr * gamma
                _weight_decay = weight_decay * gamma

            params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]

    return params


class CLSModel(BaseModel):
    def __init__(self, model: nn.Module = None, criterion=None, optimizer=None,
                 scheduler=None, epochs: int = 10, warpstep: int = 0,
                 lr=1e-3, lrf: float = 0.1, weight_decay=5e-5, gamma=0.8):
        super().__init__(model, criterion, epochs, warpstep, lr, lrf)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.gamma = gamma

    def configure_optimizers(self):
        params = get_params(self.modules(), self.lr, self.weight_decay, self.gamma)
        if self.optimizer is not None:
            optimizer = self.optimizer(params)
        else:
            optimizer = torch.optim.AdamW(params)
            # optimizer = torch.optim.AdamW(params,lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
        else:
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2000)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)

            # lrf = 0.1
            lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (
                    1 - self.lrf) + self.lrf  # cosine  last lr=lr*lrf
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        if self.warpstep > 0:
            self.warmup_lr_scheduler = warmup_lr_scheduler(optimizer, self.warpstep, 1 / self.warpstep)

        return [optimizer], [scheduler]


class GANModel(BaseModel):
    def __init__(self, model_disc, model_gen, epochs: int = 10, warpstep: int = 0, lr=1e-3, lrf: float = 0.1,
                 nz: int = 100):
        super().__init__(nn.Sequential(), epochs, warpstep, lr, lrf)
        self.model = None
        nc = 1
        dim = 64
        self.nz = nz
        BN = nn.BatchNorm2d
        if model_disc is None:
            self.model_disc = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Sequential(nn.Conv2d(nc, dim, 4, 2, 1, bias=False),
                              nn.LeakyReLU(0.2, inplace=True)),

                nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
                BN(dim * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False),
                BN(dim * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(dim * 4, dim * 8, 4, 2, 1, bias=False),
                BN(dim * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(dim * 8, 1, 4, 1, 0, bias=False),  # [4,4]->[1,1]
                nn.Flatten()
            )
            _initParmas(self.model_disc.modules())
        else:
            self.model_disc = model_disc

        if model_gen is None:
            self.model_gen = nn.Sequential(
                # input is Z, going into a convolution
                nn.Sequential(nn.ConvTranspose2d(nz, dim * 8, 4, 1, 0, bias=False),  # [1,1]->[4,4]
                              BN(dim * 8), nn.ReLU(True)),

                # state size. (dim*8) x 4 x 4
                nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1, bias=False),
                BN(dim * 4),
                nn.ReLU(True),

                # state size. (dim*4) x 8 x 8
                nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1, bias=False),
                BN(dim * 2),
                nn.ReLU(True),

                # state size. (dim*2) x 16 x 16
                nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1, bias=False),
                BN(dim),
                nn.ReLU(True),

                # state size. (dim) x 32 x 32
                nn.ConvTranspose2d(dim, nc, 4, 2, 1, bias=False),
                nn.Sigmoid(),  # 对应图像 norm (0.,1.)
                # nn.Hardtanh(0, 1, True), # [0.,1.]
                # nn.Tanh() # 对应图像norm -1.0~1.0

            )
            _initParmas(self.model_gen.modules())
        else:
            self.model_gen = model_gen

    def forward(self, x):
        out = self.model_gen(x)
        return out

    # example with learning rate schedulers
    def configure_optimizers(self):
        generator_opt = torch.optim.Adam([param for param in self.model_gen.parameters() if param.requires_grad],
                                         lr=self.lr)
        disriminator_opt = torch.optim.Adam([param for param in self.model_disc.parameters() if param.requires_grad],
                                            lr=self.lr)  # self.lr * 2
        discriminator_sched = torch.optim.lr_scheduler.CosineAnnealingLR(disriminator_opt, T_max=10)

        return [disriminator_opt, generator_opt]  # , [discriminator_sched]

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        # .unsqueeze(1)
        # x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=True)
        x = F.interpolate(x, (64, 64), mode='nearest')
        z = torch.randn([x.size(0), self.nz, 1, 1]).to(x.device)

        if optimizer_idx == 0:
            # do training_step with encoder
            out_real = self.model_disc(x)
            fake_img = self.model_gen(z).detach()
            out_fake = self.model_disc(fake_img)

            # loss_d = F.binary_cross_entropy(out_real.sigmoid(), torch.ones_like(out_real).detach()) + \
            #          F.binary_cross_entropy(out_fake.sigmoid(), torch.zeros_like(out_fake).detach())
            loss = (1 - out_real).clamp(min=0).mean() + (1 + out_fake).clamp(min=0).mean()
            # loss_d = (1 - out_real.sigmoid()).clamp(min=0).mean() + out_fake.sigmoid().clamp(min=0).mean()

        if optimizer_idx == 1:
            # do training_step with decoder
            fake_img = self.model_gen(z)
            out_fake = self.model_disc(fake_img)
            # loss_g = F.binary_cross_entropy(out_fake.sigmoid(), torch.ones_like(out_fake).detach()) + \
            #          0.2 * F.mse_loss(fake_img, x)
            loss = (1 - out_fake).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)
            # loss_g = (1 - out_fake.sigmoid()).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)

        return loss

    def training_step_bk(self, batch, batch_idx, optimizer_idx):
        """效果差？？？"""
        # 把自动优化关掉
        if self.automatic_optimization:
            self.automatic_optimization = False
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt_d, opt_g = self.optimizers()

        x, _ = batch
        # .unsqueeze(1)
        # x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=True)
        x = F.interpolate(x, (64, 64), mode='nearest')
        z = torch.randn([x.size(0), self.nz, 1, 1]).to(x.device)

        out_real = self.model_disc(x)
        fake_img = self.model_gen(z).detach()
        out_fake = self.model_disc(fake_img)

        # loss_d = F.binary_cross_entropy(out_real.sigmoid(), torch.ones_like(out_real).detach()) + \
        #          F.binary_cross_entropy(out_fake.sigmoid(), torch.zeros_like(out_fake).detach())
        loss_d = (1 - out_real).clamp(min=0).mean() + (1 + out_fake).clamp(min=0).mean()
        # loss_d = (1 - out_real.sigmoid()).clamp(min=0).mean() + out_fake.sigmoid().clamp(min=0).mean()

        opt_d.zero_grad()
        # use `manual_backward()` instead of `loss.backward` to automate half precision, etc...
        self.manual_backward(loss_d)
        opt_d.step()

        if batch_idx % 5 == 0:
            # z = torch.randn([x.size(0), self.nz, 1, 1]).to(x.device)
            fake_img = self.model_gen(z)
            out_fake = self.model_disc(fake_img)
            # loss_g = F.binary_cross_entropy(out_fake.sigmoid(), torch.ones_like(out_fake).detach()) + \
            #          0.2 * F.mse_loss(fake_img, x)
            loss_g = (1 - out_fake).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)
            # loss_g = (1 - out_fake.sigmoid()).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            x, _ = batch
            z = torch.randn([64, self.nz, 1, 1]).to(x.device)
            out = (self(z).squeeze(1) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)

            h = 8 * 64
            w = 8 * 64
            tmp = np.zeros([h, w], np.uint8)
            for i in range(0, 8):
                for j in range(0, 8):
                    tmp[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = out[j + i * 8]

            Image.fromarray(tmp).save('test.jpg')
            print("-----------epoch:{} save image successful--------".format(self.current_epoch))
            plt.imshow(tmp, cmap='gray')
            plt.show()

    def validation_step_end(self, batch_parts):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass


class DetecteModel(CLSModel):
    def __init__(self, model: nn.Module = None, criterion=None, optimizer=None,
                 scheduler=None, epochs: int = 10, warpstep: int = 0,
                 lr=1e-3, lrf: float = 0.1, weight_decay=5e-5, gamma=0.8, mode='yolov1', method='yolov1',
                 boxes_weight=5.0, thres=0.3, box_norm="log", multiscale=False, reduction="mean",focal_loss=True):
        super().__init__(model, criterion, optimizer, scheduler, epochs, warpstep, lr, lrf, weight_decay, gamma)
        self.mode = mode
        self.method = method
        self.boxes_weight = boxes_weight
        self.thres = thres
        self.box_norm = box_norm
        self.multiscale = multiscale
        self.reduction = reduction
        self.focal_loss = focal_loss

    def training_step(self, batch, batch_idx):
        if self.method == "yolov1":
            return training_step_yolov1(self, batch, batch_idx, 'train_loss')
        elif self.method == "yolov2":
            return training_step_yolov2(self, batch, batch_idx, 'train_loss')
        elif self.method == "yolov3":
            return training_step_yolov3(self, batch, batch_idx, 'train_loss')
        elif self.method == "ssd":
            return training_step_ssd(self, batch, batch_idx, 'train_loss')
        elif self.method == "ssdms":
            return training_step_ssdMS(self, batch, batch_idx, 'train_loss')
        elif self.method == "fcos":
            return training_step_fcos(self, batch, batch_idx, 'train_loss')


    def validation_step(self, batch, batch_idx):
        if self.method == "yolov1":
            return training_step_yolov1(self, batch, batch_idx, 'val_loss')
        elif self.method == "yolov2":
            return training_step_yolov2(self, batch, batch_idx, 'val_loss')
        elif self.method == "yolov3":
            return training_step_yolov3(self, batch, batch_idx, 'val_loss')
        elif self.method == "ssd":
            return training_step_ssd(self, batch, batch_idx, 'val_loss')
        elif self.method == "ssdms":
            return training_step_ssdMS(self, batch, batch_idx, 'val_loss')
        elif self.method == "fcos":
            return training_step_fcos(self, batch, batch_idx, 'val_loss')

    def validation_step_end(self, batch_parts):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass


if __name__ == "__main__":
    batch_size = 32
    model = BaseModel()

    model.save(mode=2, shape=[batch_size, 28 * 28])
    model.save_onnx(shape=[batch_size, 28 * 28])
    model.to_torchscript("model.jit")
    model.to_onnx("model.onnx", torch.rand([batch_size, 28 * 28]), verbose=True)
