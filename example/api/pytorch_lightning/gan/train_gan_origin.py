import torch
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from toolcv.api.pytorch_lightning.net import GANModel
from toolcv.api.pytorch_lightning.data import LitDataModule

training = False
batch_size = 64
epochs = 10
lr = 1e-3
lrf = 0.1
warpstep = 1000
nz = 100
print_step = 100

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dm = LitDataModule(r'D:\data\mnist', batch_size=batch_size)
dm.setup('fit')
train_dataloader = dm.train_dataloader()

model = GANModel(None, None, epochs, warpstep, lr, lrf)
model_gen = model.model_gen.to(device)
model_disc = model.model_disc.to(device)

opt_g = torch.optim.Adam([param for param in model_gen.parameters() if param.requires_grad],
                         lr=lr)
opt_d = torch.optim.Adam([param for param in model_disc.parameters() if param.requires_grad],
                         lr=lr)
sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)

for epoch in range(epochs):
    # trainer
    for step, (x, _) in enumerate(train_dataloader):
        x = x.to(device)
        # x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=True)
        x = F.interpolate(x, (64, 64), mode='nearest')
        z = torch.randn([x.size(0), nz, 1, 1]).to(device)

        out_real = model_disc(x)
        fake_img = model_gen(z)
        out_fake = model_disc(fake_img)

        # loss_d = F.binary_cross_entropy(out_real.sigmoid(), torch.ones_like(out_real).detach()) + \
        #          F.binary_cross_entropy(out_fake.sigmoid(), torch.zeros_like(out_fake).detach())
        # loss_d = (1 - out_real).clamp(min=0).mean() + (1 + out_fake).clamp(min=0).mean()
        loss_d = (1 - out_real.sigmoid()).clamp(min=0).mean() + out_fake.sigmoid().clamp(min=0).mean()

        opt_d.zero_grad()
        loss_d.backward()
        # torch.nn.utils.clip_grad_value_(model_disc.parameters(), 0.01)
        # torch.nn.utils.clip_grad_norm_(model_disc.parameters(), 0.1)
        opt_d.step()

        if step % 5 == 0:
            # z = torch.randn([x.size(0), nz, 1, 1]).to(device)
            fake_img = model_gen(z)
            out_fake = model_disc(fake_img)
            # loss_g = F.binary_cross_entropy(out_fake.sigmoid(), torch.ones_like(out_fake).detach()) + \
            #          0.2 * F.mse_loss(fake_img, x)
            # loss_g = (1 - out_fake).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)
            loss_g = (1 - out_fake.sigmoid()).clamp(min=0).mean() + 0.2 * F.mse_loss(fake_img, x)

            opt_g.zero_grad()
            loss_g.backward()
            # torch.nn.utils.clip_grad_norm_(model_gen.parameters(), 0.1)
            opt_g.step()

        if step % print_step == 0:
            print("epoch:%d step:%d loss_d:%.3f loss_g:%.3f" % (epoch, step, loss_d.item(), loss_g.item()))

    # evaluter
    z = torch.randn([64, nz, 1, 1]).to(device)
    out = (model_gen(z).squeeze(1) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)

    h = 8 * 64
    w = 8 * 64
    tmp = np.zeros([h, w], np.uint8)
    for i in range(0, 8):
        for j in range(0, 8):
            tmp[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = out[j + i * 8]

    Image.fromarray(tmp).save('test.jpg')
    print("-----------epoch:{} save image successful--------".format(epoch))
    plt.imshow(tmp,cmap='gray')
    plt.show()

    # sched_d.step()
