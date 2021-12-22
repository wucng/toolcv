import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_,clip_grad_value_
from torch.autograd import Variable
import numpy as np
from PIL import Image
import os


"""传统的GAN loss更新"""
def _train(self,epoch,lossFunc="bce",condition=False,noise="uniform"):
    assert lossFunc in ["bce","mse_loss","smooth_l1_loss","wganloss","wganloss_gp","other"]
    assert noise in ["uniform","randn","rand"]

    self.G.train()
    self.D.train()
    # d_total_loss = 0
    # g_total_loss = 0
    num_trains = len(self.train_loader.dataset)
    for batch, (data,y) in enumerate(self.train_loader):
        data = data.to(self.device)
        y = y.to(self.device)

        if noise=="rand":
            noiseData = torch.rand(data.size(0), self.nz, 1, 1, device=self.device)  # 随机噪声数据,[0,1)均匀分布
        elif noise=="randn":
            noiseData = torch.randn(data.size(0), self.nz, 1, 1, device=self.device)  # 随机噪声数据,标准正态分布 N～(0,1)
        elif noise == "uniform":
            # (-1,1)均匀分布
            noiseData = torch.from_numpy(np.random.uniform(-1,1,[data.size(0),self.nz, 1, 1]).astype(np.float32)).to(self.device)
        else:
            pass

        fake = self.G(noiseData,y if condition else None)
        d_real = self.D(data,y if condition else None)
        d_fake = self.D(fake.detach(),y if condition else None)


        if lossFunc=="bce":
            d_loss = F.binary_cross_entropy(d_real.sigmoid(),torch.ones_like(d_real),reduction=self.reduction)+\
                        F.binary_cross_entropy(d_fake.sigmoid(),torch.zeros_like(d_real),reduction=self.reduction)
        elif lossFunc=="wganloss":
            d_loss = d_fake.mean() - d_real.mean()
        elif lossFunc=="wganloss_gp":
            d_loss = d_fake.mean() - d_real.mean()

            alpha = torch.rand(data.shape).to(self.device)
            differences = fake - data  # This is different from MAGAN
            interpolates = Variable(data + alpha*differences,True)
            d_inter = self.D(interpolates,y if condition else None).mean()
            d_inter.backward()
            gradients = interpolates.grad.data
            slopes = gradients.square().sum(1).sqrt()
            gradient_penalty = ((slopes-1.0)**2).mean()
            d_loss = d_loss+self.lambd * gradient_penalty

        elif lossFunc == "smooth_l1_loss":
            d_loss = F.smooth_l1_loss(d_real, torch.ones_like(d_real), reduction=self.reduction) + \
                     F.smooth_l1_loss(d_fake, torch.zeros_like(d_real), reduction=self.reduction)

        elif lossFunc == "mse_loss":
            d_loss = F.mse_loss(d_real, torch.ones_like(d_real), reduction=self.reduction) + \
                     F.mse_loss(d_fake, torch.zeros_like(d_real), reduction=self.reduction)

        elif lossFunc == "other":
            # d_loss = d_fake.exp().mean() +1.0/d_real.exp().mean()
            d_loss = -(1-d_fake.sigmoid()).log().mean()-d_real.sigmoid().log().mean()


        self.D_opt.zero_grad()
        d_loss.backward()
        clip_grad_value_(self.D.parameters(), 1e-2)
        self.D_opt.step()

        g_real = self.D(fake,y if condition else None)

        if lossFunc=="bce":
            g_loss = F.binary_cross_entropy(g_real.sigmoid(),torch.ones_like(g_real),reduction=self.reduction)
        elif lossFunc == "wganloss":
            g_loss = -g_real.mean()
        elif lossFunc == "smooth_l1_loss":
            g_loss = F.smooth_l1_loss(g_real,torch.ones_like(g_real),reduction=self.reduction)
        elif lossFunc == "mse_loss":
            g_loss = F.mse_loss(g_real, torch.ones_like(g_real), reduction=self.reduction)
        elif lossFunc == "other":
            # g_loss = 1.0/g_real.exp().mean()
            g_loss = -g_real.sigmoid().log().mean()

        g_loss = g_loss+0.5*F.mse_loss(fake,data,reduction=self.reduction)

        self.G_opt.zero_grad()
        g_loss.backward()
        self.G_opt.step()

        if batch % self.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDLoss: {:.6f} GLoss: {:.6f}'.format(epoch, batch * len(data),
                                                                                          num_trains,
                                                                                          100. * batch * len(
                                                                                              data) / num_trains,
                                                                                          d_loss.data.item(),
                                                                                          g_loss.data.item()))

def _predict(self,condition=False,noise="uniform"):
    assert noise in ["uniform", "rand", "randn"]
    self.G.eval()
    self.D.eval()
    with torch.no_grad():
        labels = torch.randint(0, self.num_classes, [64], dtype=torch.long, device=self.device)

        if noise=="rand":
            fixed_noise = torch.rand(64, self.nz, 1, 1, device=self.device)  # 随机噪声数据,[0,1)均匀分布
        elif noise=="randn":
            fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)  # 随机噪声数据,标准正态分布 N～(0,1)
        elif noise=="uniform":
            # (-1,1)均匀分布
            fixed_noise = torch.from_numpy(np.random.uniform(-1,1,[64,self.nz, 1, 1]).astype(np.float32)).to(self.device)

        decode = self.G(fixed_noise,labels if condition else None)

        if not condition:
            pred = self.D(decode,labels if condition else None)
            pred = torch.sigmoid(pred).squeeze().round()
            acc = (pred == torch.ones_like(pred)).sum().float()/pred.size(0)
            print("acc:",acc)

        # decode得到的图片
        decode = torch.clamp(decode * 255, 0, 255)  # 对应0~1
        # decode = torch.clamp((decode * 0.5 + 0.5) * 255, 0, 255) # -1.0~1.0
        # decode = torch.clamp((decode + 0.5) * 255, 0, 255) # -0.5~0.5
        decode = decode.detach().cpu().numpy().astype(np.uint8)
        decode = np.transpose(decode, [0, 2, 3, 1])

        imgs = np.zeros([8 * self.image_size, 8 * self.image_size, self.nc], np.uint8)
        for i in range(8):
            for j in range(8):
                imgs[i * self.image_size:(i + 1) * self.image_size, j * self.image_size:(j + 1) * self.image_size] = \
                decode[i * 8 + j]

        # 保存
        Image.fromarray(imgs.squeeze()).save("test.jpg")
