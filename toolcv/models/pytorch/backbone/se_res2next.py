"""
经典分类CNN模型系列其八：ResNeXt  https://www.jianshu.com/p/7478ce41e46b

Res2Net------论文理解  https://blog.csdn.net/ruoruojiaojiao/article/details/89074763
"""
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
import math


class CBA(nn.Module):
    """
        CBA: Convolution + Batchnormal + Activate
    """
    def __init__(self,in_c,out_c,ksize=3,stride=1,padding="same",dilation=1,groups=1,bias=False,
                 use_bn=True,activate=True):
        super().__init__()
        if padding=="same":
            padding = (ksize+2*(dilation-1))//2
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


class BottleBlock(nn.Module):
    expansion = 4
    C = 16 # 32
    n_split=4
    def __init__(self,in_c,out_c,stride=1,downsample=None,reduction=16):
        super().__init__()
        tmp_c = out_c//self.n_split
        self.conv1x1_1 = CBA(in_c,out_c,1)
        self.conv3x3 = nn.Sequential(
            CBA(tmp_c, tmp_c, 3, 1, groups=self.C),
            CBA(tmp_c, tmp_c, 3, 1, groups=self.C),
            CBA(tmp_c, tmp_c, 3, 1, groups=self.C)
        )
        self.maxpool = nn.MaxPool2d(3,2,1) if stride>1 else nn.Sequential()

        self.conv1x1_2 = CBA(out_c,out_c*self.expansion, 1,activate=False)
        self.downsample = downsample

        self.se = SELayer(out_c*self.expansion,reduction)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = x
        out = self.conv1x1_1(x)
        x1,x2,x3,x4 = torch.chunk(out,self.n_split,1)
        # x2 = self.conv3x3[0](x2)
        x2 = self.conv3x3[0](x1+x2)
        x3 = self.conv3x3[1](x2+x3)
        x4 = self.conv3x3[2](x3+x4)
        out = torch.cat((x1,x2,x3,x4),1)
        out = self.maxpool(out)
        out = self.conv1x1_2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,in_c,block,layers=[],num_classes=1000,dropout=0.5):
        super().__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            CBA(in_c,self.inplanes,7,2),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = _make_layer(self,block, 64, layers[0])
        self.layer2 = _make_layer(self,block, 128, layers[1], 2)
        self.layer3 = _make_layer(self,block, 256, layers[2], 2)
        self.layer4 = _make_layer(self,block, 512, layers[3], 2)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(self.inplanes,num_classes)

        _init_weights(self)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = CBA(self.inplanes, planes * block.expansion, 1, stride, activate=False)

    layers = []
    layers.append(block(self.inplanes,planes,stride,downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes,planes,1,None))

    return nn.Sequential(*layers)

def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def se_res2next50(in_c=3,num_classes=1000,dropout=0.0):
    """2+3*(3+4+6+3)=50"""
    layers = [3, 4, 6, 3]
    model = ResNet(in_c, BottleBlock, layers=layers, num_classes=num_classes, dropout=dropout)
    return model

def se_res2next101(in_c=3,num_classes=1000,dropout=0.0):
    layers = [3,4,23,3]
    model = ResNet(in_c, BottleBlock, layers=layers, num_classes=num_classes, dropout=dropout)
    return model

def se_res2next152(in_c=3,num_classes=1000,dropout=0.0):
    layers = [3,8,36,3]
    model = ResNet(in_c, BottleBlock, layers=layers, num_classes=num_classes, dropout=dropout)
    return model

if __name__ == "__main__":
    model = se_res2next50()
    x = torch.randn([1,3,224,224])
    pred = model(x)
    print(pred.shape)
    print(model)

    torch.save(model.state_dict(),'se_res2next50.pth')