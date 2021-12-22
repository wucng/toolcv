import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet, vgg
from collections import OrderedDict
from torchvision.ops import RoIAlign
import math

from toolcv.utils.tools.tools import _initParmasV2
from toolcv.utils.net.backbone import BaseNet


# from toolcv.utils.net.net import BackboneDilate, CBA, Backbone

class CBA(nn.Module):
    """
        CBA: Convolution + Batchnormal + Activate

        空洞卷积的等效卷积核大小尺寸计算公式如下：

        K=k+(k−1)∗(r−1)

        其中，K 代表等效卷积核尺寸，k 代表实际卷积核尺寸，而r 代表dilation，空洞卷积的参数。

        如： k = 3 ;dilation=2 最后卷积核大小为 3+(3-1)(2-1) = 5
    """

    def __init__(self, in_c, out_c, ksize=3, stride=1, padding="same", dilation=1, groups=1, bias=False,
                 use_bn=True, activate=True, conv2d=None):
        super().__init__()
        if padding == "same":
            if isinstance(ksize, int):
                padding = (ksize + (ksize - 1) * (dilation - 1)) // 2
            else:
                padding = (ksize[0] + (ksize[0] - 1) * (dilation - 1)) // 2, \
                          (ksize[1] + (ksize[1] - 1) * (dilation - 1)) // 2
        elif isinstance(padding, int):
            padding = padding
        else:
            padding = 0
        bias = not use_bn

        if conv2d is None: conv2d = nn.Conv2d
        self.conv = conv2d(in_c, out_c, ksize, stride, padding, dilation, groups, bias)
        # self.bn = nn.BatchNorm2d(out_c, momentum=0.03, eps=1E-4) if use_bn else nn.Sequential()

        if isinstance(use_bn, bool) and use_bn:
            self.bn = nn.BatchNorm2d(out_c, momentum=0.03, eps=1E-4)
        elif isinstance(use_bn, nn.Module):
            self.bn = use_bn
        else:
            self.bn = nn.Sequential()

        if isinstance(activate, bool) and activate:
            self.act = nn.ReLU(inplace=True)
        elif isinstance(activate, nn.Module):
            self.act = activate
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Backbone(BaseNet):
    """使用膨胀卷积 不做下采样 且保持相同的 感受野"""

    def __init__(self, model_name='resnet18', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 num_classes=1000, do_cls=False,norm_eval=True):
        super().__init__(freeze_at,norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        self.do_cls = do_cls

        # model = resnet.__dict__[model_name](pretrained)
        model = getattr(resnet, model_name)(pretrained)
        if stride == 4:
            model.layer2[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer2[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer2[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            model.layer3[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer3[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer3[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
            if dilation:
                model.layer2[0].conv2.dilation = (2, 2)
                model.layer2[0].conv2.padding = (2, 2)
                model.layer3[0].conv2.dilation = (4, 4)
                model.layer3[0].conv2.padding = (4, 4)
                model.layer4[0].conv2.dilation = (8, 8)
                model.layer4[0].conv2.padding = (8, 8)

        if stride == 8:
            model.layer3[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer3[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer3[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
            if dilation:
                model.layer3[0].conv2.dilation = (2, 2)
                model.layer3[0].conv2.padding = (2, 2)
                model.layer4[0].conv2.dilation = (4, 4)
                model.layer4[0].conv2.padding = (4, 4)

        if stride == 16:
            model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
            model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
            if dilation:
                model.layer4[0].conv2.dilation = (2, 2)
                model.layer4[0].conv2.padding = (2, 2)

        self.backbone = nn.Sequential(
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.out_channels = model.inplanes

        # for parme in self.backbone[:freeze_at].parameters():
        #     # for parme in self.backbone.parameters():
        #     parme.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

        if self.do_cls:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                      nn.Linear(self.out_channels, num_classes))

            _initParmas(self.head.modules())

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)
        if self.do_cls: return self.head(out[-1])

        index = len(out) - self.num_out
        assert index >= 0 and index < len(out)

        out = out[index:]
        return out if len(out) > 1 else out[0]

    def forwardMS(self, x):
        # 多分支
        x4 = self.backbone[:2](x)  # c2
        x8 = self.backbone[2](x4)  # c3
        x16 = self.backbone[3](x8)  # c4
        x32 = self.backbone[4](x16)  # c5

        return x4, x8, x16, x32


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self, in_c,use_bn=True,activate=True):
        super(SPP, self).__init__()
        self.conv = CBA(in_c * 4, in_c, 3, 1,use_bn=use_bn,activate=activate)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.conv(x)

        if isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
            inputs[-1] = x
        else:
            inputs = x

        return inputs


class SPPv2(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self, in_c, out_c,use_bn=True,activate=True):
        super(SPPv2, self).__init__()
        self.conv1 = CBA(in_c, out_c, 1,use_bn=use_bn,activate=activate)
        self.conv = CBA(out_c * 4, out_c, 3, 1,use_bn=use_bn,activate=activate)
        self.conv2 = CBA(out_c, in_c, 1,use_bn=use_bn,activate=activate)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        x = self.conv1(x)
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.conv(x)
        x = self.conv2(x)

        if isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
            inputs[-1] = x
        else:
            inputs = x

        return inputs


class Yolov1(nn.Module):
    def __init__(self, backbone, spp, neck, head, focalloss=False):
        """
        num_classes: 不算背景
        """
        super().__init__()

        self.backbone = backbone
        self.spp = spp
        self.neck = neck
        self.head = head
        # 初始化参数
        _initParmasV2(self.spp.modules(), mode="kaiming_normal")
        _initParmasV2(self.neck.modules(), mode="kaiming_normal")
        _initParmasV2(self.head.modules(), mode="normal")

        if focalloss:
            # 如果使用focal loss
            prior_probability = 0.01
            torch.nn.init.constant_(self.head.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = self.neck(x)
        x = self.head(x)

        # x = x.permute(0, 2, 3, 1)

        return x


class Yolov4(nn.Module):
    def __init__(self, backbone, spp, pan, head):
        super().__init__()
        self.backbone = backbone
        self.spp = spp
        self.pan = pan
        self.head = head

        # 初始化参数
        _initParmasV2(self.spp.modules(), mode="kaiming_normal")
        _initParmasV2(self.pan.modules(), mode="kaiming_normal")
        _initParmasV2(self.head.modules(), mode="normal")

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = self.neck(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    x = torch.rand([2, 3, 416, 416])
    m = Backbone("resnet18", True, 5, 16)
    m.eval()
    print(m(x).shape)
