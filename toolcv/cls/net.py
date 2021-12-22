import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from collections import OrderedDict

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

from toolcv.models.pytorch import dla
from toolcv.network.net import _initParmas, _make_detnet_layer
from toolcv.models.pytorch.common import CBA, BottleBlockX


class Simple(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False,
                 num_classes=10, dropout=0.5, freeze_at=5):
        super().__init__()

        if 'resnet' in model_name or 'resnext' in model_name:
            model = resnet.__dict__[model_name](pretrained)

            self.backbone = nn.Sequential(
                nn.Sequential(model.conv1,
                              model.bn1,
                              model.relu,
                              model.maxpool),

                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )

            out_channels = model.inplanes

        if 'dla' in model_name:
            _model = dla.__dict__[model_name](pretrained)
            self.backbone = nn.Sequential(
                nn.Sequential(_model.base_layer, _model.level0, _model.level1),
                _model.level2,
                _model.level3,
                _model.level4,
                _model.level5
            )

            out_channels = _model.channels[-1]

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)

        # 默认冻结 BN中的参数 不更新
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in m.parameters():
                    parameter.requires_grad_(False)

        self.logit = nn.Sequential(
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        )

        _initParmas(self.logit.modules())

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        x = self.backbone(x)
        x = self.logit(x)

        return x


class WConv(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()

        self.layer1 = CBA(in_c, out_c, 3, stride)
        self.layer2 = CBA(in_c, out_c, 3, stride)
        self.layer3 = CBA(in_c, out_c, 3, stride)
        self.layer4 = CBA(in_c, out_c, 3, stride)

    def forward(self, x):
        bs, c, h, w = x.shape
        x1 = x[..., torch.arange(0, h, 2), :][..., torch.arange(0, w, 2)]
        x2 = torch.fliplr(x1)
        x3 = x[..., torch.arange(1, h, 2), :][..., torch.arange(1, w, 2)]
        x4 = torch.fliplr(x3)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)
        x4 = self.layer4(x4)

        return torch.cat((x1, x2, x3, x4), 1)


class WNet(nn.Module):
    """
    x: [1,3,448,448]
    x: [1,3,224,224]
    """
    def __init__(self, in_c=3, stride=2, num_classes=10, dropout=0.5):
        super().__init__()

        self.conv1 = WConv(in_c, 32, 2)
        self.bottle1 = _make_detnet_layer(128, 128)

        self.conv2 = WConv(128, 64, 2)
        self.bottle2 = _make_detnet_layer(256, 256)

        self.conv3 = WConv(256, 128, stride)
        self.bottle3 = _make_detnet_layer(512, 512)

        self.logit = nn.Sequential(
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        _initParmas(self.modules())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottle1(x)

        x = self.conv2(x)
        x = self.bottle2(x)

        x = self.conv3(x)
        x = self.bottle3(x)

        x = self.logit(x)

        return x


class MultiStageUnionNet(nn.Module):
    """
    x: [1,3,224,224]
    """
    def __init__(self, in_c=3, num_classes=1000):
        super().__init__()

        self.layer1 = nn.Sequential(
            BottleBlockX(in_c * 4, 64, 64, 3, 2, True, 32),
            BottleBlockX(64, 128, 64, 3, 1, True, 32),
            BottleBlockX(64, 128, 64, 3, 1, True, 32))

        self.layer2 = nn.Sequential(
            BottleBlockX(64, 64, 128, 3, 2, True, 32),
            BottleBlockX(128, 256, 128, 3, 1, True, 32),
            BottleBlockX(128, 256, 128, 3, 1, True, 32))

        self.layer3 = nn.Sequential(
            BottleBlockX(128, 128, 256, 3, 2, True, 32),
            BottleBlockX(256, 512, 256, 3, 1, True, 32),
            BottleBlockX(256, 512, 256, 3, 1, True, 32))

        self.layer4 = nn.Sequential(
            BottleBlockX(256, 256, 512, 3, 2, True, 32),
            BottleBlockX(512, 1024, 512, 3, 1, True, 32),
            BottleBlockX(512, 1024, 512, 3, 1, True, 32))

        self.conv2 = nn.Conv2d(in_c * 16, 64, 1)
        self.conv3 = nn.Conv2d(in_c * 64, 128, 1)
        self.conv4 = nn.Conv2d(in_c * 256, 256, 1)
        self.conv5 = nn.Conv2d(in_c * 1024, 512, 1)

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)

        )

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = rearrange(x, 'b c (h1 s1) (w1 s2) -> b (c s1 s2) h1 w1', s1=2, s2=2)
        x2 = rearrange(x, 'b c (h1 s1) (w1 s2) -> b (c s1 s2) h1 w1', s1=4, s2=4)
        x3 = rearrange(x, 'b c (h1 s1) (w1 s2) -> b (c s1 s2) h1 w1', s1=8, s2=8)
        x4 = rearrange(x, 'b c (h1 s1) (w1 s2) -> b (c s1 s2) h1 w1', s1=16, s2=16)
        x5 = rearrange(x, 'b c (h1 s1) (w1 s2) -> b (c s1 s2) h1 w1', s1=32, s2=32)

        # 应用注意力机制
        x1 = self.layer1(x1) * torch.sigmoid(self.conv2(x2))
        x1 = self.layer2(x1) * torch.sigmoid(self.conv3(x3))
        x1 = self.layer3(x1) * torch.sigmoid(self.conv4(x4))
        x1 = self.layer4(x1) * torch.sigmoid(self.conv5(x5))

        # x1 = self.layer1(x1) + self.conv2(x2)
        # x1 = self.layer2(x1) + self.conv3(x3)
        # x1 = self.layer3(x1) + self.conv4(x4)
        # x1 = self.layer4(x1) + self.conv5(x5)

        x1 = self.cls(x1)

        return x1


if __name__ == "__main__":
    # m = WNet()
    m = MultiStageUnionNet(3)
    x = torch.rand([1, 3, 224, 224])
    pred = m(x)
    print(pred.shape)
