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

class InceptionResnetA(nn.Module):
    def __init__(self,in_c=384,reduction=16):
        super().__init__()
        self.m1 = CBA(in_c,32,1)
        self.m2 = nn.Sequential(
            CBA(in_c,32,1),
            CBA(32,32,3)
        )
        self.m3 = nn.Sequential(
            CBA(in_c, 32, 1),
            CBA(32, 48, 3),
            CBA(48, 64, 3)
        )
        self.l1 = CBA(128,384,1,activate=False)
        self.se = SELayer(384, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = torch.cat((self.m1(x),self.m2(x),self.m3(x)),1)
        x1 = self.l1(x1)
        x1 = self.se(x1)

        return self.relu(x1+x)

class ReductionA(nn.Module):
    def __init__(self,in_c=384):
        super().__init__()
        self.m1 = nn.MaxPool2d(3,2,0)
        self.m2 = CBA(in_c,384,3,2,'valid')
        self.m3 = nn.Sequential(
            CBA(in_c,256,1,1,'same'),
            CBA(256,256,3,1,'same'),
            CBA(256,384,3,2,'valid')
        )


    def forward(self,x):
        return torch.cat((self.m1(x),self.m2(x),self.m3(x)),1)

class InceptionResnetB(nn.Module):
    def __init__(self,in_c=1152,reduction=16):
        super().__init__()
        self.m1 = CBA(in_c,192,1)
        self.m2 = nn.Sequential(
            CBA(in_c,128,1),
            CBA(128,160,(1,7)),
            CBA(160,192,(7,1))
        )

        self.l1 = CBA(384,in_c,1,activate=False)
        self.se = SELayer(in_c, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = torch.cat((self.m1(x),self.m2(x)),1)
        x1 = self.l1(x1)
        x1 = self.se(x1)
        return self.relu(x1+x)

class ReductionB(nn.Module):
    def __init__(self,in_c=1152):
        super().__init__()
        self.m1 = nn.MaxPool2d(3,2,0)
        self.m2 = nn.Sequential(
            CBA(in_c,256,1),
            CBA(256,384,3,2,'valid')
        )
        self.m3 = nn.Sequential(
            CBA(in_c, 256, 1),
            CBA(256, 288, 3, 2, 'valid')
        )

        self.m4 = nn.Sequential(
            CBA(in_c,256,1,1,'same'),
            CBA(256,288,3,1,'same'),
            CBA(288,320,3,2,'valid')
        )


    def forward(self,x):
        return torch.cat((self.m1(x),self.m2(x),self.m3(x),self.m4(x)),1)

class InceptionResnetC(nn.Module):
    def __init__(self,in_c=2144,reduction=16):
        super().__init__()
        self.m1 = CBA(in_c,192,1)
        self.m2 = nn.Sequential(
            CBA(in_c,192,1),
            CBA(192,224,(1,3)),
            CBA(224,256,(3,1))
        )

        self.l1 = CBA(448,in_c,1,activate=False)
        self.se = SELayer(in_c, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = torch.cat((self.m1(x),self.m2(x)),1)
        x1 = self.l1(x1)
        x1 = self.se(x1)

        return self.relu(x1+x)

class SEInceptionResnetV2(nn.Module):
    def __init__(self,in_c=3,dropout=0.2,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c)
        self.inceptionResnetA = nn.Sequential(*[InceptionResnetA() for _ in range(5)])
        self.reductionA = ReductionA()
        self.inceptionResnetB = nn.Sequential(*[InceptionResnetB() for _ in range(10)])
        self.reductionB = ReductionB()
        self.inceptionResnetC = nn.Sequential(*[InceptionResnetC() for _ in range(5)])

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(2144,num_classes)

        _init_weights(self)

    def forward(self,x):
        x = self.stem(x)
        x = self.inceptionResnetA(x)
        x = self.reductionA(x)
        x = self.inceptionResnetB(x)
        x = self.reductionB(x)
        x = self.inceptionResnetC(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = SEInceptionResnetV2(3)
    print(model)
    x = torch.randn([1,3,299,299])
    pred = model(x)
    print(pred.shape)

    # torch.save(model.state_dict(),'InceptionV4.pth')