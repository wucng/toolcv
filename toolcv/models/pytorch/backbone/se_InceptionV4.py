"""
深度学习卷积神经网络——经典网络GoogLeNet(Inception V3)网络的搭建与实现:
https://blog.csdn.net/loveliuzz/article/details/79135583

Network             k l m n
Inception-v4        192 224 256 384
Inception-ResNet-v1 192 192 256 384
Inception-ResNet-v2 256 256 384 384
"""

from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBA(nn.Module):
    """
        CBA: Convolution + Batchnormal + Activate
    """
    def __init__(self,in_c,out_c,ksize=3,stride=1,padding="same",dilation=1,groups=1,bias=False,
                 use_bn=True,activate=True):
        super().__init__()
        if padding=="same":
            if isinstance(ksize,int):
                padding = (ksize+2*(dilation-1))//2
            else:
                padding = ((ksize[0] + 2 * (dilation - 1)) // 2,(ksize[1]+2*(dilation-1))//2)
        else:
            padding = 0
        bias = not use_bn
        self.conv = nn.Conv2d(in_c,out_c,ksize,stride,padding,dilation,groups,bias)
        self.bn = nn.BatchNorm2d(out_c) if use_bn else nn.Sequential()
        if isinstance(activate,bool) and activate:
            self.act = nn.ReLU(inplace=True)
        elif isinstance(activate,nn.Module):
            self.act = activate
        else:
            self.act = nn.Sequential()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Stem(nn.Module):
    def __init__(self,in_c):
        super().__init__()
        self.m = nn.Sequential(
            CBA(in_c, 32, 3, 2, 'valid'),
            CBA(32, 32, 3, 1, 'valid'),
            CBA(32, 64, 3, 1, 'same'))
        self.maxpool = nn.MaxPool2d(3,2,0)
        self.conv3x3_v2 = CBA(64,96,3,2,'valid')

        self.m1 = nn.Sequential(
            CBA(160,64,1),
            CBA(64,96,3,1,'valid')
        )
        self.m2 = nn.Sequential(
            CBA(160, 64, 1),
            CBA(64, 64, (7,1)),
            CBA(64, 64, (1,7)),
            CBA(64,96,3,1,'valid')
        )

        self.conv3x3_v2_2 = CBA(192, 192, 3, 2, 'valid')
        self.maxpool_2 = nn.MaxPool2d(3, 2, 0)

    def forward(self,x):
        x = self.m(x)
        x = torch.cat((self.maxpool(x),self.conv3x3_v2(x)),1)
        x = torch.cat((self.m1(x), self.m2(x)), 1)
        x = torch.cat((self.conv3x3_v2_2(x),self.maxpool_2(x)), 1)
        return x

class InceptionA(nn.Module):
    def __init__(self,in_c=384,reduction=16):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.AvgPool2d(3,1,1),
            CBA(in_c,96,1)
        )
        self.m2 = CBA(in_c,96,1)
        self.m3 = nn.Sequential(
            CBA(in_c, 64, 1),
            CBA(64, 96, 3)
        )
        self.m4 = nn.Sequential(
            CBA(in_c, 64, 1),
            CBA(64, 96, 3),
            CBA(96, 96, 3)
        )

        self.se = SELayer(384,reduction)

    def forward(self,x):
        return self.se(torch.cat((self.m1(x),self.m2(x),self.m3(x),self.m4(x)),1))

class ReductionA(nn.Module):
    def __init__(self,in_c=384):
        super().__init__()
        self.m1 = nn.MaxPool2d(3,2,0)
        self.m2 = CBA(in_c,384,3,2,'valid')
        self.m3 = nn.Sequential(
            CBA(in_c,192,1,1,'same'),
            CBA(192,224,3,1,'same'),
            CBA(224,256,3,2,'valid')
        )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # return self.relu(torch.cat((self.m1(x),self.m2(x),self.m3(x)),1))
        return torch.cat((self.m1(x),self.m2(x),self.m3(x)),1)

class InceptionB(nn.Module):
    def __init__(self,in_c=1024,reduction=16):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.AvgPool2d(3,1,1),
            CBA(in_c,128,1)
        )
        self.m2 = CBA(in_c,384,1)
        self.m3 = nn.Sequential(
            CBA(in_c, 192, 1),
            CBA(192, 224, (1,7)),
            CBA(224, 256, (1,7))
        )
        self.m4 = nn.Sequential(
            CBA(in_c, 192, 1),
            CBA(192, 192, (1, 7)),
            CBA(192, 224, (7, 1)),
            CBA(224, 224, (1, 7)),
            CBA(224, 256, (7, 1))
        )

        self.se = SELayer(1024, reduction)

    def forward(self,x):
        return self.se(torch.cat((self.m1(x),self.m2(x),self.m3(x),self.m4(x)),1))

class ReductionB(nn.Module):
    def __init__(self,in_c=1024):
        super().__init__()
        self.m1 = nn.MaxPool2d(3,2,0)
        self.m2 = nn.Sequential(
            CBA(in_c,192,1),
            CBA(192,192,3,2,'valid')
        )
        self.m3 = nn.Sequential(
            CBA(in_c,256,1),
            CBA(256,256,(1,7)),
            CBA(256,320,(7,1)),
            CBA(320,320,3,2,'valid')
        )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # return self.relu(torch.cat((self.m1(x),self.m2(x),self.m3(x)),1))
        return torch.cat((self.m1(x),self.m2(x),self.m3(x)),1)


class InceptionC(nn.Module):
    def __init__(self,in_c=1536,reduction=16):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.AvgPool2d(3,1,1),
            CBA(in_c,256,1)
        )
        self.m2 = CBA(in_c,256,1)
        self.m3 = nn.Sequential(
            CBA(in_c, 384, 1),
            CBA(384, 256, (1,3)),
            CBA(384, 256, (3,1))
        )
        self.m4 = nn.Sequential(
            CBA(in_c, 384, 1),
            CBA(384, 448, (1, 3)),
            CBA(448, 512, (3, 1)),
            CBA(512, 256, (3, 1)),
            CBA(512, 256, (1, 3))
        )

        self.se = SELayer(1536, reduction)

    def forward(self,x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        x3 = self.m3[0](x)
        x31 = self.m3[1](x3)
        x32 = self.m3[2](x3)
        x4 = self.m4[:3](x)
        x41 = self.m4[3](x4)
        x42 = self.m4[4](x4)

        return self.se(torch.cat((x1,x2,x31,x32,x41,x42),1))

class SEInceptionV4(nn.Module):
    def __init__(self,in_c=3,dropout=0.2,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c)
        self.inceptionA = nn.Sequential(*[InceptionA() for _ in range(4)])
        self.reductionA = ReductionA()
        self.inceptionB = nn.Sequential(*[InceptionB() for _ in range(7)])
        self.reductionB = ReductionB()
        self.inceptionC = nn.Sequential(*[InceptionC() for _ in range(3)])

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(1536,num_classes)

        _init_weights(self)


    def forward(self,x):
        x= self.stem(x)
        x= self.inceptionA(x)
        x= self.reductionA(x)
        x= self.inceptionB(x)
        x= self.reductionB(x)
        x= self.inceptionC(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = SEInceptionV4(3)
    print(model)
    x = torch.randn([1,3,299,299])
    pred = model(x)
    print(pred.shape)

    # torch.save(model.state_dict(),'SEInceptionV4.pth')