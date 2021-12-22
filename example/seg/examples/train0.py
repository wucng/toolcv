import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnext50_32x4d,resnet34
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

classes = ["__bg__","scratches"]
out_channles = 256
num_classes = len(classes)

def _initParmas(modules, std=0.01):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # model = resnext50_32x4d(False)
        model = resnet34(True)
        self.backbone = nn.Sequential(
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.out_channels = model.inplanes
        for parme in self.backbone.parameters():
            parme.requires_grad_(False)

    def forward(self, x):
        x4 = self.backbone[:2](x)  # c2
        x8 = self.backbone[2](x4)  # c3
        x16 = self.backbone[3](x8)  # c4
        x32 = self.backbone[4](x16)  # c5

        return x4, x8, x16, x32

class FPN(nn.Module):
    def __init__(self, in_c, out_c=256, Conv2d=nn.Conv2d):  # conv = nn.Conv2d
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.lateral_convs.extend(
            [Conv2d(in_c // 8, out_c, 1), Conv2d(in_c // 4, out_c, 1),
             Conv2d(in_c // 2, out_c, 1), Conv2d(in_c, out_c, 1)])

        self.fpn_convs= Conv2d(out_c, out_c, 3, 1, 1)

        self.upsample = nn.Upsample(None, 2, 'nearest')  # False

    def forward(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        m5 = self.lateral_convs[3](c5)  # C5
        m4 = self.upsample(m5) + self.lateral_convs[2](c4)  # C4
        m3 = self.upsample(m4) + self.lateral_convs[1](c3)  # C3
        m2 = self.upsample(m3) + self.lateral_convs[0](c2)  # C2

        p2 = self.fpn_convs(m2)

        return p2


head = nn.Sequential(
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(out_channles, out_channles, 3, 2, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, out_channles, 3, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(out_channles, out_channles, 3, 2, 1, 1),
    nn.BatchNorm2d(out_channles),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channles, num_classes, 3, 1, 1)
)

backbone = Backbone()
fpn = FPN(backbone.out_channels,out_channles)
_initParmas(fpn.modules())
_initParmas(head.modules())

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model = nn.Sequential(backbone, fpn, head).to(device)

# x = torch.randn([1,3,352,640]).to(device)
# pred = model(x)
# print(pred.shape)

class WaferDataset(Dataset):
    def __init__(self,root):
        self.image_paths = glob(os.path.join(root,"image","*.jpg"))
        self.mask_paths = glob(os.path.join(root,"mask","*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert('RGB')
        mask = Image.open(self.mask_paths[item])#.convert('GRAY')

        # resize
        img = img.resize((640,352))
        mask = mask.resize((640,352),0)
        img = np.array(img,np.float32)
        mask = np.array(mask,np.float32)
        # 镜像
        if random.random() < 0.5:
            img = np.flip(img,1)
            mask = np.flip(mask,1)
        # to tensor
        img = torch.tensor(img.copy()/255.).permute(2,0,1)
        mask = torch.tensor(mask.copy()).long()
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img-torch.tensor(mean)[:,None,None])/torch.tensor(std)[:,None,None]

        return img,mask

if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth',device))
    print("----------load weight successful------------")

def train():
    optim = AdamW([parma for parma in model.parameters() if parma.requires_grad],lr=5e-4)
    scheduler = StepLR(optim,20,0.8)
    train_dataset = WaferDataset("./datas")
    train_dataLoader = DataLoader(train_dataset,4,True)

    for epoch in range(250):
        for step,(img,mask) in enumerate(train_dataLoader):
            model.train()
            img = img.to(device)
            mask = mask.to(device)
            pred = model(img)
            loss = F.cross_entropy(pred,mask)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if step%10==0:
                print("epoch:%d step:%d loss:%.5f"%(epoch,step,loss.item()))

        scheduler.step()
        torch.save(model.state_dict(),'model.pth')

@torch.no_grad()
def test():
    model.eval()
    img_paths = glob(os.path.join("./datas","image","*"))
    for img_path in img_paths:
        mask_path = img_path.replace("image","mask")
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)  # .convert('GRAY')
        shape = mask.size
        # resize
        img = img.resize((640, 352))
        mask = mask.resize((640, 352), 0)
        img = np.array(img, np.float32)
        mask = np.array(mask, np.float32)
        # to tensor
        img = torch.tensor(img.copy() / 255.).permute(2, 0, 1)
        mask = torch.tensor(mask.copy()).long()
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]

        pred = model(img[None].to(device))[0]
        pred = torch.argmax(pred,0).cpu().numpy().astype(np.uint8)

        # acc
        acc = ((mask-pred)==0).sum()/mask.numel()
        print(acc)

        # resize
        save_path = mask_path.replace("mask","pred")
        if not os.path.exists(os.path.dirname(save_path)):os.makedirs(os.path.dirname(save_path))
        Image.fromarray(pred).resize(shape,0).save(save_path)

if __name__ == "__main__":
    # train()
    test()