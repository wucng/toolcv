"""
backbone + SPP + FPN + head
SPP:多尺度融合 （PSPModule，ASPP,DilatedParllelResidualBlockB）
FPN:多级语义融合 （PAN）
"""
import torch
from torch import nn
from torch.nn import functional as F
# from torchvision.models.resnet import resnext50_32x4d,resnet34
from torchvision.models import resnet
# from timm.models import resnet,resnest,res2net,tresnet

from toolcv.tools.segment.net.espnet import DilatedParllelResidualBlockB as DRBlock
from toolcv.tools.segment.net.pspnet import PSPModule
from toolcv.tools.segment.net.YOLOP.common import SPP, BottleneckCSP, Conv
from toolcv.tools.segment.net.deeplabv3 import ASPP

from toolcv.tools.tools_summary import model_profile


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
    def __init__(self, model_name="resnet34"):
        super().__init__()
        # model = resnext50_32x4d(False)
        # model = resnet.__dict__[model_name](True)
        model = getattr(resnet, model_name)(True)
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

        # spp or aspp 多尺度融合
        # self.spp = SPP(self.out_channels, self.out_channels) # 18.982G 24.367M
        # self.spp = ASPP(self.out_channels,[6, 12, 18])
        # self.spp = PSPModule(self.out_channels, self.out_channels) # 19.092G 26.070M
        self.spp = DRBlock(self.out_channels, self.out_channels)  # 18.965G 24.233M
        self.fusion = BottleneckCSP(self.out_channels, self.out_channels, 1, False)

    def forward(self, x):
        x4 = self.backbone[:2](x)  # c2
        x8 = self.backbone[2](x4)  # c3
        x16 = self.backbone[3](x8)  # c4
        x32 = self.backbone[4](x16)  # c5

        x32 = self.spp(x32)
        x32 = self.fusion(x32)

        return x4, x8, x16, x32


class FPN(nn.Module):
    """多级语义融合"""

    def __init__(self, in_c, out_c=256, Conv2d=nn.Conv2d):  # conv = nn.Conv2d
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.lateral_convs.extend(
            [Conv2d(in_c // 8, out_c, 1), Conv2d(in_c // 4, out_c, 1),
             Conv2d(in_c // 2, out_c, 1), Conv2d(in_c, out_c, 1)])

        self.fpn_convs = Conv2d(out_c, out_c, 3, 1, 1)
        # self.fpn_convs = nn.ModuleList([Conv2d(out_c, out_c, 3, 1, 1) for _ in range(4)])

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


def get_head(out_channles=256, num_classes=2):
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
    return head


def get_headv2(out_channles=256, num_classes=2):
    aspp_dilate = [12, 24, 36]
    head = nn.Sequential(
        ASPP(out_channles, aspp_dilate),
        Conv(out_channles, 128, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(128, 64, 1, False),
        Conv(64, 32, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(32, 16, 1, False),
        Conv(16, num_classes, 3, 1, 1)
    )
    return head


def unet(model_name="resnet34", out_channles=256, num_classes=2):
    backbone = Backbone(model_name)
    fpn = FPN(backbone.out_channels, out_channles)
    head = get_headv2(out_channles, num_classes)
    _initParmas(fpn.modules())
    _initParmas(head.modules())

    model = nn.Sequential(backbone, fpn, head)
    return model


if __name__ == "__main__":
    m = unet()
    x = torch.randn([2, 3, 256, 256])
    # print(m)
    # print(m(x).shape)

    model_profile(m, x)
