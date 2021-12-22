import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.hub import load_state_dict_from_url

from toolcv.tools.segment.net.unet import FPN, _initParmas, DRBlock, BottleneckCSP, SPP, get_head, Conv,ASPP  # get_headv2


def get_headv2_0(out_channles=256, num_classes=2):
    # max mIOU:0.48631 mean_time:0.800
    # aspp_dilate = [12, 24, 36]
    head = nn.Sequential(
        # ASPP(out_channles, aspp_dilate),
        Conv(out_channles, 128, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(128, 64, 1, False),
        Conv(64, 32, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(32, 16, 1, False),
        Conv(16, num_classes, 3, 1, 1)
        # nn.Conv2d(16, num_classes, 3, 1, 1)
    )
    return head

def get_headv2(out_channles=256, num_classes=2):
    aspp_dilate = [12, 24, 36]
    head = nn.Sequential(
        # 不加ASPP max mIOU:0.52988 mean_time:1.171
        # 加ASPP max mIOU:0.50449 mean_time:1.403
        ASPP(out_channles, aspp_dilate),
        Conv(out_channles, out_channles, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(out_channles, out_channles, 1, False),
        Conv(out_channles, out_channles, 3, 1, 1),
        nn.Upsample(scale_factor=2),
        BottleneckCSP(out_channles, out_channles, 1, False),
        Conv(out_channles, num_classes, 3, 1, 1)
        # nn.Conv2d(16, num_classes, 3, 1, 1)
    )
    return head

class Backbone(nn.Module):
    def __init__(self, model_name="resnet34"):
        super().__init__()
        model = getattr(resnet, model_name)(True)
        # model = getattr(resnet, model_name)(False)
        # state_dict = load_state_dict_from_url(resnet.model_urls[model_name], progress=True)
        # # state_dict = torch.load("./resnet34-333f7ec4.pth")
        # model.load_state_dict(state_dict)

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
        self.spp = SPP(self.out_channels, self.out_channels)  # 18.982G 24.367M
        # self.spp = ASPP(self.out_channels,[6, 12, 18])
        # self.spp = PSPModule(self.out_channels, self.out_channels) # 19.092G 26.070M
        # self.spp = DRBlock(self.out_channels, self.out_channels)  # 18.965G 24.233M
        self.fusion = BottleneckCSP(self.out_channels, self.out_channels, 1, False)

    def forward(self, x):
        x4 = self.backbone[:2](x)  # c2
        x8 = self.backbone[2](x4)  # c3
        x16 = self.backbone[3](x8)  # c4
        x32 = self.backbone[4](x16)  # c5

        x32 = self.spp(x32)
        x32 = self.fusion(x32)

        return x4, x8, x16, x32


def unet(model_name="resnet34", out_channles=256, num_classes=2):
    backbone = Backbone(model_name)
    fpn = FPN(backbone.out_channels, out_channles)
    head = get_headv2(out_channles, num_classes) # 不加ASPP max mIOU:0.52988 mean_time:1.171
                                                 # 加ASPP max mIOU:0.50449 mean_time:1.403
    # head = get_head(out_channles, num_classes) # max mIOU:0.65293 mean_time:0.917
    _initParmas(fpn.modules())
    _initParmas(head.modules())

    model = nn.Sequential(backbone, fpn, head)
    return model
