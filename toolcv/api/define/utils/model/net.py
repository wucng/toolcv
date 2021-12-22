"""
一文看尽物体检测中的各种FPN:
https://zhuanlan.zhihu.com/p/148738276
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet, vgg
from collections import OrderedDict
from torchvision.ops import RoIAlign

from toolcv.models.pytorch import dla
from toolcv.utils.net.backbone import BaseNet

try:
    from mmcv.ops import DeformConv2dPack
    # from mmdet.core.bbox.iou_calculators import iou2d_calculator
    # from fvcore.nn import giou_loss
    # from mmcv.ops import RoIAlign

    # nas_fpn
    # from mmdet.models.necks import nas_fpn,pafpn

    from mmcv.cnn import ConvModule
    from mmcv.ops.merge_cells import GlobalPoolingCell, SumCell
    from mmcv.runner import BaseModule, ModuleList

    # from mmcv.ops import Conv2d, ConvTranspose2d, DeformConv2d, MaskedConv2d, SAConv2d, \
    # ModulatedDeformConv2d, ModulatedDeformConv2dPack

    # from torchvision.ops import DeformConv2d
except:
    print("warning:pip install mmcv")

# spp or aspp 多尺度融合
from toolcv.tools.segment.net.espnet import DilatedParllelResidualBlockB as DRBlock
from toolcv.tools.segment.net.pspnet import PSPModule
from toolcv.tools.segment.net.YOLOP.common import SPP, BottleneckCSP, Conv
from toolcv.tools.segment.net.deeplabv3 import ASPP


class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0, reduction='none'):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction=reduction)
            # self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def labelSmoothing(labels, num_classes, esp=0.1):
    if labels.ndim == 1:
        labels = F.one_hot(labels, num_classes).float()
    return (1 - labels) * (esp / (num_classes - 1)) + labels * (1 - esp)


def _initParmas(modules, std=0.01, mode='normal'):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if mode == 'normal':
                nn.init.normal_(m.weight, std=std)
            elif mode == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, 0, std=std)
        #     if m.bias is not None:
        #         # nn.init.zeros_(m.bias)
        #         nn.init.constant_(m.bias, 0)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv3x3_1 = CBA(in_c, out_c, 3, stride)
        self.conv3x3_2 = CBA(out_c, out_c, 3, 1, activate=False)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv3x3_1(x)
        out = self.conv3x3_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1x1_1 = CBA(in_c, out_c, 1)
        self.conv3x3 = CBA(out_c, out_c, 3, stride)
        self.conv1x1_2 = CBA(out_c, out_c * self.expansion, 1, activate=False)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1x1_1(x)
        out = self.conv3x3(out)
        out = self.conv1x1_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = CBA(self.inplanes, planes * block.expansion, 1, stride, activate=False)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, 1, None))

    return nn.Sequential(*layers)


def CBL(in_channels=3, out_channels=32,
        kernel_size=3, stride=1, padding=1,
        groups=1, bias=False, negative_slope=0.1):
    return nn.Sequential(OrderedDict([
        ('Conv2d', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)),
        ('BatchNorm2d', nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)),
        ('activation', nn.LeakyReLU(negative_slope, inplace=True))
    ]))


class SPPNet(nn.Module):
    """论文的版本"""

    def __init__(self, in_channels):
        super().__init__()
        self.spp = nn.Sequential(
            nn.MaxPool2d(1, 1, 0),
            nn.MaxPool2d(5, 1, 2),
            nn.MaxPool2d(9, 1, 4),
            nn.MaxPool2d(13, 1, 6),
            CBA(in_channels * 4, in_channels, 1, 1, activate=nn.LeakyReLU(0.1, True))
        )

    def forward(self, x):
        x1 = self.spp[0](x)
        x2 = self.spp[1](x)
        x3 = self.spp[2](x)
        x4 = self.spp[3](x)

        return self.spp[4:](torch.cat((x1, x2, x3, x4), 1))


class SeBlock(nn.Module):
    """SENet模块
        通道 注意力机制
    """

    def __init__(self, inputs, reduces=16):
        super(SeBlock, self).__init__()
        self.fc1 = nn.Linear(inputs, inputs // reduces)
        self.fc2 = nn.Linear(inputs // reduces, inputs)

    def forward(self, x):
        x1 = torch.mean(x, [2, 3])
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        # x1 = F.sigmoid(x1)
        x1 = torch.sigmoid(x1)

        return x * x1[..., None, None]


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.seblock = SeBlock(self.expansion * planes)
        # self.seblock = CBAM(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.seblock(out)

        out += self.downsample(x)
        out = F.relu(out)
        return out


def _make_detnet_layer(in_channels, hide_size=256):
    layers = []
    layers.append(detnet_bottleneck(in_planes=in_channels, planes=hide_size, block_type='B'))
    layers.append(detnet_bottleneck(in_planes=hide_size, planes=hide_size, block_type='A'))
    layers.append(detnet_bottleneck(in_planes=hide_size, planes=hide_size, block_type='A'))
    return nn.Sequential(*layers)


# --------------------------------------------------------------
class YoloBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=False, activate=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.l1 = CBA(in_c, out_c, 1, use_bn=use_bn, activate=activate)
        self.l2 = CBA(out_c, out_c, 3, use_bn=use_bn, activate=activate)
        self.l3 = CBA(out_c, out_c * 2, 1, use_bn=use_bn, activate=activate)

        self.se_block = SeBlock(out_c * 2)
        self.downsample = CBA(in_c, out_c * 2, 1, use_bn=use_bn, activate=False)
        self.activate = activate

    def forward(self, x):
        x1 = self.l1(x)
        x1 = self.l2(x1)
        x1 = self.l3(x1)
        x1 = self.se_block(x1)

        return self.activate(x1 + self.downsample(x))


class Yolov3Neckv1(nn.Module):
    def __init__(self, in_c, out_c=256, use_bn=False, activate=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.spp = SPPNet(in_c)
        self.upsample = nn.Upsample(None, 2, 'nearest')  # align_corners=False

        self.x32 = nn.Sequential(YoloBlock(in_c, in_c // 2, use_bn, activate),
                                 YoloBlock(in_c, in_c // 2, use_bn, activate),
                                 CBA(in_c, out_c, 3, use_bn=use_bn, activate=activate))
        self.x16 = nn.Sequential(YoloBlock(in_c + in_c // 2, in_c // 4, use_bn, activate),
                                 YoloBlock(in_c // 2, in_c // 4, use_bn, activate),
                                 CBA(in_c // 2, out_c, 3, use_bn=use_bn, activate=activate))
        self.x8 = nn.Sequential(YoloBlock(in_c // 2 + in_c // 4, in_c // 8, use_bn, activate),
                                YoloBlock(in_c // 4, in_c // 8, use_bn, activate),
                                CBA(in_c // 4, out_c, 3, use_bn=use_bn, activate=activate))

    def forward(self, x8, x16, x32):
        x32 = self.spp(x32)
        _x32 = self.x32[:2](x32)
        x32 = self.x32[2](_x32)

        x16 = torch.cat((self.upsample(_x32), x16), 1)
        _x16 = self.x16[:2](x16)
        x16 = self.x16[2](_x16)

        x8 = torch.cat((self.upsample(_x16), x8), 1)
        x8 = self.x8(x8)

        return x8, x16, x32


class YoloHeadv1(nn.Module):
    """num_classes不包含背景"""

    def __init__(self, in_c, out_c=256, num_classes=20, num_anchor=1, use_bn=False,
                 activate=nn.LeakyReLU(0.2, inplace=True), shared=True):
        super().__init__()
        self.shared = shared
        self.m = nn.Sequential(
            CBA(in_c, out_c, 1, 1, use_bn=use_bn, activate=activate),
            CBA(out_c, out_c * 2, 3, 1, use_bn=use_bn, activate=activate),
            CBA(out_c * 2, out_c, 1, 1, use_bn=use_bn, activate=activate),
            CBA(out_c, out_c * 2, 3, 1, use_bn=use_bn, activate=activate),
        )
        self.filter = (5 + num_classes) * num_anchor

        if self.shared:
            self.m2 = nn.Conv2d(out_c * 2, self.filter, 1)
        else:
            self.m8 = nn.Conv2d(out_c * 2, self.filter, 1)
            self.m16 = nn.Conv2d(out_c * 2, self.filter, 1)
            self.m32 = nn.Conv2d(out_c * 2, self.filter, 1)

    def forward(self, x):  # 全部共享权重
        assert self.shared
        return self.m2(self.m(x))

    def forwardMS(self, x8, x16, x32):  # 部分共享权重
        assert not self.shared
        x8 = self.m(x8)
        x16 = self.m(x16)
        x32 = self.m(x32)

        return self.m8(x8), self.m16(x16), self.m32(x32)


class YoloV1(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.conv1 = CBA(backbone.out_channels, backbone.out_channels // 4, 1)
        self.spp = SPPNet(backbone.out_channels // 4)
        self.conv2 = CBA(backbone.out_channels // 4, backbone.out_channels, 1)

        _initParmas(self.head.modules())
        _initParmas(self.spp.modules())

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.spp(x)
        x = self.conv2(x)
        x = self.head(x)

        return x


# --------------以下是参考 mmdetection(推荐)----------------------------------

class Backbone(BaseNet):
    def __init__(self, model_name='resnet18', pretrained=False, freeze_at=5, stride=32, num_out=1,
                 num_classes=1000, do_cls=False, norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        self.num_out = num_out
        self.do_cls = do_cls
        if 'resnet' in model_name or 'resnext' in model_name:
            model = resnet.__dict__[model_name](pretrained)

            if stride == 4:
                model.layer2[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer2[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer2[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            if stride in [4, 8]:
                model.layer3[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            if stride in [4, 8, 16]:
                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

            self.backbone = nn.Sequential(
                nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),

                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )

            self.out_channels = model.inplanes

        if 'dla' in model_name:
            _model = dla.__dict__[model_name](pretrained)

            if stride in [4]:
                _model.level3.tree1.tree1.conv1.stride = (1, 1)
                _model.level3.tree1.tree1.conv2.stride = (1, 1)
                _model.level3.downsample = nn.Sequential()

            if stride in [4, 8]:
                _model.level4.tree1.tree1.conv1.stride = (1, 1)
                _model.level4.tree1.tree1.conv2.stride = (1, 1)
                _model.level4.downsample = nn.Sequential()

            if stride in [4, 8, 16]:
                _model.level5.tree1.conv1.stride = (1, 1)
                _model.level5.tree1.conv2.stride = (1, 1)
                # _model.level5.downsample.stride = 1
                _model.level5.downsample = nn.Sequential()

            self.backbone = nn.Sequential(
                nn.Sequential(_model.base_layer, _model.level0, _model.level1),
                _model.level2,
                _model.level3,
                _model.level4,
                _model.level5
            )

            self.out_channels = _model.channels[-1]

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)

        if freeze_at > 0:
            # 默认冻结 BN中的参数 不更新
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    for parameter in m.parameters():
                        parameter.requires_grad_(False)

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


class BackboneDilate(BaseNet):
    """使用膨胀卷积 不做下采样 且保持相同的 感受野"""

    def __init__(self, model_name='resnet18', pretrained=False, freeze_at=5, stride=32, num_out=1,
                 num_classes=1000, do_cls=False, norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        self.num_out = num_out
        self.do_cls = do_cls
        if 'resnet' in model_name or 'resnext' in model_name:
            model = resnet.__dict__[model_name](pretrained)

            if stride == 4:
                model.layer2[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer2[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer2[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
                model.layer2[0].conv2.dilation = (2, 2)
                model.layer2[0].conv2.padding = (2, 2)

                model.layer3[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].conv2.dilation = (4, 4)
                model.layer3[0].conv2.padding = (4, 4)

                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.dilation = (8, 8)
                model.layer4[0].conv2.padding = (8, 8)

            if stride == 8:
                model.layer3[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
                model.layer3[0].conv2.dilation = (2, 2)
                model.layer3[0].conv2.padding = (2, 2)

                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.dilation = (4, 4)
                model.layer4[0].conv2.padding = (4, 4)

            if stride == 16:
                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)
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

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)

        if freeze_at > 0:
            # 默认冻结 BN中的参数 不更新
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    for parameter in m.parameters():
                        parameter.requires_grad_(False)

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


class Vgg16Backbone(nn.Module):
    def __init__(self, model_name='vgg16', pretrained=False, freeze_at=23, num_out=2):
        """
                                         vgg16  vgg16_bn
            最后一个stage不冻结   freeze_at    23   33
            全部冻结            freeze_at    30   43
        """
        super().__init__()
        self.model_name = model_name
        self.num_out = num_out
        if model_name == 'vgg16':
            model = vgg.__dict__[model_name](pretrained)
            model.features[16].ceil_mode = True
            model.features[30].kernel_size = 3
            model.features[30].stride = 1
            model.features[30].padding = 1
            if self.num_out == 2: conv7 = nn.Sequential(CBA(512, 1024, 3, 1, dilation=6, use_bn=False),
                                                        CBA(1024, 1024, 1, use_bn=False))
            # features[:23]

        elif model_name == 'vgg16_bn':
            model = vgg.__dict__[model_name](pretrained)
            model.features[23].ceil_mode = True
            model.features[43].kernel_size = 3
            model.features[43].stride = 1
            model.features[43].padding = 1

            # print(model.features[:33](x).shape)
            if self.num_out == 2: conv7 = nn.Sequential(CBA(512, 1024, 3, 1, dilation=6, use_bn=True),
                                                        CBA(1024, 1024, 1, use_bn=True))
        else:
            raise ('error')

        self.backbone = model.features
        if self.num_out == 2: self.conv7 = conv7

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)
        # 默认冻结 BN中的参数 不更新
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in m.parameters():
                    parameter.requires_grad_(False)

    def forward(self, x):  # stride = 16
        if self.num_out == 2:
            return self.forwardMS(x)
        else:
            return self.backbone(x)

    def forwardMS(self, x):
        if self.model_name == 'vgg16':
            conv4_3 = self.backbone[:23](x)
            conv7 = self.conv7(self.backbone[23:](conv4_3))
        elif self.model_name == 'vgg16_bn':
            conv4_3 = self.backbone[:33](x)
            conv7 = self.conv7(self.backbone[33:](conv4_3))

        return conv4_3, conv7


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSDHead300(nn.Module):
    """num_classes包含背景"""

    def __init__(self, num_classes=21, num_anchors=[4, 6, 6, 6, 4, 4], dim_transform=True):
        super().__init__()
        self.num_classes = num_classes
        # self.num_anchors = num_anchors
        self.dim_transform = dim_transform

        self.l2 = L2Norm(512)
        self.conv8 = nn.Sequential(CBA(1024, 256, 1, use_bn=False), CBA(256, 512, 3, 2, use_bn=False))
        self.conv9 = nn.Sequential(CBA(512, 128, 1, use_bn=False), CBA(128, 256, 3, 2, use_bn=False))
        self.conv10 = nn.Sequential(CBA(256, 128, 1, use_bn=False), CBA(128, 256, 3, 1, 'valid', use_bn=False))
        self.conv11 = nn.Sequential(CBA(256, 128, 1, use_bn=False), CBA(128, 256, 3, 1, 'valid', use_bn=False))

        self.cls_convs = nn.ModuleList([nn.Conv2d(512, num_anchors[0] * num_classes, 3, 1, 1),
                                        nn.Conv2d(1024, num_anchors[1] * num_classes, 3, 1, 1),
                                        nn.Conv2d(512, num_anchors[2] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[3] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[4] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[5] * num_classes, 3, 1, 1)])

        self.reg_convs = nn.ModuleList([nn.Conv2d(512, num_anchors[0] * 4, 3, 1, 1),
                                        nn.Conv2d(1024, num_anchors[1] * 4, 3, 1, 1),
                                        nn.Conv2d(512, num_anchors[2] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[3] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[4] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[5] * 4, 3, 1, 1)])

    def forward(self, x):
        conv4_3, conv7 = x
        out = [self.l2(conv4_3), conv7]
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        out.extend([conv8, conv9, conv10, conv11])

        cls_list, reg_list = [], []
        for i, x in enumerate(out):
            cls, reg = self.forward_single(x, i)
            cls_list.append(cls)
            reg_list.append(reg)

        if self.dim_transform:
            cls_list = torch.cat(cls_list, 1)
            reg_list = torch.cat(reg_list, 1)

        return cls_list, reg_list

    def forward_single(self, x, i):
        cls = self.cls_convs[i](x)
        reg = self.reg_convs[i](x)

        if self.dim_transform:
            # bs, c, h, w = cls.shape
            bs = cls.size(0)
            # cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors[i], self.num_classes)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        return cls, reg


class SSDHead512(nn.Module):
    """num_classes包含背景"""

    def __init__(self, num_classes=21, num_anchors=[4, 6, 6, 6, 6, 4, 4], dim_transform=True):
        super().__init__()
        self.num_classes = num_classes
        self.dim_transform = dim_transform
        self.l2 = L2Norm(512)
        self.conv8 = nn.Sequential(CBA(1024, 256, 1, use_bn=False), CBA(256, 512, 3, 2, use_bn=False))
        self.conv9 = nn.Sequential(CBA(512, 128, 1, use_bn=False), CBA(128, 256, 3, 2, use_bn=False))
        self.conv10 = nn.Sequential(CBA(256, 128, 1, use_bn=False), CBA(128, 256, 3, 2, use_bn=False))
        self.conv11 = nn.Sequential(CBA(256, 128, 1, use_bn=False), CBA(128, 256, 3, 2, use_bn=False))
        self.conv12 = nn.Sequential(CBA(256, 128, 1, use_bn=False), CBA(128, 256, 2, 1, 'valid', use_bn=False))

        self.cls_convs = nn.ModuleList([nn.Conv2d(512, num_anchors[0] * num_classes, 3, 1, 1),
                                        nn.Conv2d(1024, num_anchors[1] * num_classes, 3, 1, 1),
                                        nn.Conv2d(512, num_anchors[2] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[3] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[4] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[5] * num_classes, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[6] * num_classes, 3, 1, 1)])

        self.reg_convs = nn.ModuleList([nn.Conv2d(512, num_anchors[0] * 4, 3, 1, 1),
                                        nn.Conv2d(1024, num_anchors[1] * 4, 3, 1, 1),
                                        nn.Conv2d(512, num_anchors[2] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[3] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[4] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[5] * 4, 3, 1, 1),
                                        nn.Conv2d(256, num_anchors[6] * 4, 3, 1, 1)
                                        ])

    def forward(self, x):
        conv4_3, conv7 = x
        out = [self.l2(conv4_3), conv7]
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)
        out.extend([conv8, conv9, conv10, conv11, conv12])

        cls_list, reg_list = [], []
        for i, x in enumerate(out):
            cls, reg = self.forward_single(x, i)
            cls_list.append(cls)
            reg_list.append(reg)

        if self.dim_transform:
            cls_list = torch.cat(cls_list, 1)
            reg_list = torch.cat(reg_list, 1)

        return cls_list, reg_list

    def forward_single(self, x, i):
        cls = self.cls_convs[i](x)
        reg = self.reg_convs[i](x)

        if self.dim_transform:
            # bs, c, h, w = cls.shape
            bs = cls.size(0)
            # cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors[i], self.num_classes)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        return cls, reg


class CenterNetNeck(nn.Module):
    def __init__(self, in_c, out_c=256, use_dcn=False):
        super().__init__()
        self.neck = nn.Sequential(
            DeformConv2dPack(in_c, out_c, 3, 1, 1) if use_dcn else nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            DeformConv2dPack(out_c, out_c // 2, 3, 1, 1) if use_dcn else nn.Conv2d(out_c, out_c // 2, 3, 1, 1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_c // 2, out_c // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True),

            DeformConv2dPack(out_c // 2, out_c // 4, 3, 1, 1) if use_dcn else nn.Conv2d(out_c // 2, out_c // 4, 3, 1,
                                                                                        1),
            nn.BatchNorm2d(out_c // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_c // 4, out_c // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c // 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.neck(x)


class CenterNetHead(nn.Module):
    """num_classes不包含背景"""

    def __init__(self, in_c, num_classes=20):
        super().__init__()

        self.headmap = nn.Sequential(
            CBA(in_c, in_c, 3, 1, use_bn=False),
            nn.Conv2d(in_c, num_classes, 1)
        )

        self.wh = nn.Sequential(
            CBA(in_c, in_c, 3, 1, use_bn=False),
            nn.Conv2d(in_c, 2, 1)
        )

        self.offset = nn.Sequential(
            CBA(in_c, in_c, 3, 1, use_bn=False),
            nn.Conv2d(in_c, 2, 1)
        )

    def forward(self, x):
        return self.headmap(x), self.wh(x), self.offset(x)


class FPN(nn.Module):
    def __init__(self, in_c, out_c=256, only_three=False, Conv2d=CBA):  # conv = nn.Conv2d
        super().__init__()
        self.only_three = only_three
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.lateral_convs.extend(
            [Conv2d(in_c // 4, out_c, 1), Conv2d(in_c // 2, out_c, 1), Conv2d(in_c, out_c, 1)])

        if self.only_three:
            self.fpn_convs.extend(
                [Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1)])
        else:
            self.fpn_convs.extend(
                [Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1),
                 Conv2d(in_c, out_c, 3, 2, 1), Conv2d(out_c, out_c, 3, 2, 1)])

        self.upsample = nn.Upsample(None, 2, 'nearest')  # False

    def forward(self, x):
        assert len(x) == 3
        c3, c4, c5 = x
        m5 = self.lateral_convs[2](c5)  # C5
        m4 = self.upsample(m5) + self.lateral_convs[1](c4)  # C4
        m3 = self.upsample(m4) + self.lateral_convs[0](c3)  # C3

        p3 = self.fpn_convs[0](m3)
        p4 = self.fpn_convs[1](m4)
        p5 = self.fpn_convs[2](m5)
        if self.only_three:
            return p3, p4, p5

        p6 = self.fpn_convs[3](x[2])
        p7 = self.fpn_convs[4](p6)

        return p3, p4, p5, p6, p7


class FPNv2(nn.Module):
    def __init__(self, in_c, out_c=256, only_four=False, Conv2d=CBA):  # conv = nn.Conv2d
        super().__init__()
        self.only_four = only_four
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.lateral_convs.extend(
            [Conv2d(in_c // 8, out_c, 1), Conv2d(in_c // 4, out_c, 1),
             Conv2d(in_c // 2, out_c, 1), Conv2d(in_c, out_c, 1)])

        if self.only_four:
            self.fpn_convs.extend(
                [Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1),
                 Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1)])
        else:
            self.fpn_convs.extend(
                [Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1), Conv2d(out_c, out_c, 3, 1, 1),
                 Conv2d(out_c, out_c, 3, 1, 1), Conv2d(in_c, out_c, 3, 2, 1), Conv2d(out_c, out_c, 3, 2, 1)])

        self.upsample = nn.Upsample(None, 2, 'nearest')  # False

    def forward(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        m5 = self.lateral_convs[3](c5)  # C5
        m4 = self.upsample(m5) + self.lateral_convs[2](c4)  # C4
        m3 = self.upsample(m4) + self.lateral_convs[1](c3)  # C3
        m2 = self.upsample(m3) + self.lateral_convs[0](c2)  # C2

        p2 = self.fpn_convs[0](m2)
        p3 = self.fpn_convs[1](m3)
        p4 = self.fpn_convs[2](m4)
        p5 = self.fpn_convs[3](m5)
        if self.only_four:
            return p2, p3, p4, p5

        p6 = self.fpn_convs[4](x[2])
        p7 = self.fpn_convs[5](p6)

        return p3, p4, p5, p6, p7


class FCOSHead(nn.Module):
    def __init__(self, in_c=256, num_classes=80, use_gn=False):
        super().__init__()
        _convs = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(32, in_c) if use_gn else nn.Sequential(),
            nn.ReLU(inplace=True),
        )
        self.cls_convs = nn.Sequential(*[_convs for _ in range(4)])
        self.reg_convs = nn.Sequential(*[_convs for _ in range(4)])

        self.conv_cls = nn.Conv2d(in_c, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_reg = nn.Conv2d(in_c, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_centerness = nn.Conv2d(in_c, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        cls_pred_list, reg_pred_list, centerness_pred_list = [], [], []
        for _x in x:
            cls_pred, reg_pred, centerness_pred = self.forward_single(_x)
            cls_pred_list.append(cls_pred)
            reg_pred_list.append(reg_pred)
            centerness_pred_list.append(centerness_pred)
        return cls_pred_list, reg_pred_list, centerness_pred_list

    def forward_single(self, x):
        cls = self.cls_convs(x)
        reg = self.reg_convs(x)

        cls_pred = self.conv_cls(cls)
        centerness_pred = self.conv_centerness(cls)
        reg_pred = self.conv_reg(reg)

        return cls_pred, reg_pred, centerness_pred


class YOLOV3Neck(nn.Module):
    def __init__(self, in_c, use_spp=True):
        super().__init__()
        self.upsample = nn.Upsample(None, 2, 'nearest')  # False

        activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.detect1 = nn.Sequential(CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 2, in_c, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate),
                                     SPPNet(in_c // 2) if use_spp else nn.Sequential(),
                                     CBA(in_c // 2, in_c, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate))

        self.conv1 = CBA(in_c // 2, in_c // 4, 1, use_bn=True, activate=activate)

        self.detect2 = nn.Sequential(CBA(in_c // 2 + in_c // 4, in_c // 4, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 4, in_c // 2, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 2, in_c // 4, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 4, in_c // 2, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 2, in_c // 4, 1, 1, use_bn=True, activate=activate))
        self.conv2 = CBA(in_c // 4, in_c // 8, 1, use_bn=True, activate=activate)

        self.detect3 = nn.Sequential(CBA(in_c // 4 + in_c // 8, in_c // 8, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 8, in_c // 4, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 4, in_c // 8, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 8, in_c // 4, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 4, in_c // 8, 1, 1, use_bn=True, activate=activate))

    def forward(self, x):
        x32 = self.detect1(x[2])
        x16 = torch.cat((self.upsample(self.conv1(x32)), x[1]), 1)
        x16 = self.detect2(x16)
        x8 = torch.cat((self.upsample(self.conv2(x16)), x[0]), 1)
        x8 = self.detect3(x8)

        return x8, x16, x32


class YOLOV3Head(nn.Module):
    def __init__(self, in_c, num_anchor=3, num_classes=80, use_bn=True, dim_transform=True):
        super().__init__()
        self.dim_transform = dim_transform
        filters = num_anchor * (num_classes + 5)
        activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.convs_bridge = nn.ModuleList([CBA(in_c, in_c * 2, 3, 1, use_bn=use_bn, activate=activate),
                                           CBA(in_c // 2, in_c, 3, 1, use_bn=use_bn, activate=activate),
                                           CBA(in_c // 4, in_c // 2, 3, 1, use_bn=use_bn, activate=activate)
                                           ])
        self.convs_pred = nn.ModuleList([nn.Conv2d(in_c * 2, filters, 1),
                                         nn.Conv2d(in_c, filters, 1),
                                         nn.Conv2d(in_c // 2, filters, 1)])

        self.num_anchor = num_anchor
        self.num_classes = num_classes

    def forward(self, x):
        x8, x16, x32 = x
        x32 = self.convs_pred[0](self.convs_bridge[0](x32))
        x16 = self.convs_pred[1](self.convs_bridge[1](x16))
        x8 = self.convs_pred[2](self.convs_bridge[2](x8))

        if self.dim_transform:
            bs, _, h, w = x8.shape
            x8 = x8.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchor, -1)

            bs, _, h, w = x16.shape
            x16 = x16.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchor, -1)

            bs, _, h, w = x32.shape
            x32 = x32.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchor, -1)

        return x8, x16, x32


class YOLOXHead(nn.Module):
    def __init__(self, in_c=256, num_anchor=3, num_classes=80, activate=None):
        super().__init__()
        if activate is None:
            activate = nn.SiLU()
        self.multi_level_cls_convs = nn.ModuleList(
            [nn.Sequential(*[CBA(in_c, in_c, 3, 1, activate=activate) for _ in range(2)]) for _ in range(3)])

        self.multi_level_reg_convs = nn.ModuleList(
            [nn.Sequential(*[CBA(in_c, in_c, 3, 1, activate=activate) for _ in range(2)]) for _ in range(3)])

        self.multi_level_conv_cls = nn.ModuleList([nn.Conv2d(in_c, num_classes, 1) for _ in range(3)])
        self.multi_level_conv_reg = nn.ModuleList([nn.Conv2d(in_c, 4, 1) for _ in range(3)])
        self.multi_level_conv_obj = nn.ModuleList([nn.Conv2d(in_c, 1, 1) for _ in range(3)])

    def forward(self, x):
        # x8, x16, x32 = x
        cls_list, reg_list, obj_list = [], [], []
        for i, _x in enumerate(x):
            cls, reg, obj = self.forward_single(_x, i)
            cls_list.append(cls)
            reg_list.append(reg)
            obj_list.append(obj)

        return cls_list, reg_list, obj_list

    def forward_single(self, x, i):
        _cls_convs = self.multi_level_cls_convs[i](x)
        cls = self.multi_level_conv_cls[i](_cls_convs)
        _reg_convs = self.multi_level_reg_convs[i](x)
        reg = self.multi_level_conv_reg[i](_reg_convs)
        obj = self.multi_level_conv_obj[i](_reg_convs)

        return cls, reg, obj


class YOLOV1Neck(nn.Module):
    def __init__(self, in_c, use_spp=True):
        super().__init__()
        activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.detect1 = nn.Sequential(CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate),
                                     CBA(in_c // 2, in_c, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate),
                                     SPPNet(in_c // 2) if use_spp else nn.Sequential(),
                                     CBA(in_c // 2, in_c, 3, 1, use_bn=True, activate=activate),
                                     CBA(in_c, in_c // 2, 1, 1, use_bn=True, activate=activate))

    def forward(self, x):
        x = self.detect1(x)

        return x


class YOLOV1Head(nn.Module):
    def __init__(self, in_c, num_anchor=2, num_classes=80, use_bn=True):
        super().__init__()
        # filters = num_anchor * (num_classes + 5)
        filters = 5 * num_anchor + num_classes
        activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.convs_bridge = CBA(in_c, in_c * 2, 3, 1, use_bn=use_bn, activate=activate)
        self.convs_pred = nn.Conv2d(in_c * 2, filters, 1)

    def forward(self, x):
        x = self.convs_pred(self.convs_bridge(x))
        return x


class BottleBlockF(nn.Module):
    def __init__(self, in_c, out_c, dilation=2):
        super().__init__()
        self.conv1x1_1 = CBA(in_c, out_c, 1)
        self.conv3x3 = CBA(out_c, out_c, 3, 1, dilation=dilation)
        self.conv1x1_2 = CBA(out_c, in_c, 1)

    def forward(self, x):
        residual = x
        out = self.conv1x1_1(x)
        out = self.conv3x3(out)
        out = self.conv1x1_2(out)

        out = out + residual

        return out


class YOLOFNeck(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.m = nn.Sequential(CBA(in_c, in_c // 4, 1, use_bn=True, activate=False),
                               CBA(in_c // 4, in_c // 4, 3, use_bn=True, activate=False))
        self.dilated_encoder_blocks = nn.Sequential(*[BottleBlockF(in_c // 4, in_c // 16, 2 * i) for i in range(1, 5)])

    def forward(self, x):
        return self.dilated_encoder_blocks(self.m(x))


class YOLOFHead(nn.Module):
    def __init__(self, in_c, num_anchor=5, num_classes=80, use_bn=True):
        super().__init__()

        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_pred = self.bbox_pred(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)

        return cls_score, bbox_pred, object_pred


class YOLOFHeadSelf(nn.Module):
    """动态选择 anchor"""

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80, use_bn=True):
        super().__init__()

        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        self.wh_cls = nn.Conv2d(in_c, num_anchor * 2 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_offset = self.bbox_offset(bbox_subnet)
        wh_cls = self.wh_cls(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)

        return object_pred, cls_score, bbox_offset, wh_cls


class YOLOFHeadSelfAndKeypoints(nn.Module):
    """动态选择 anchor
        同时做 keypoints
    """

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80, num_keypoints=17):
        super().__init__()
        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        self.wh_cls = nn.Conv2d(in_c, num_anchor * 2 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor
        self.keypoints_pred = nn.Conv2d(in_c, num_keypoints, 3, 1, 1)

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_offset = self.bbox_offset(bbox_subnet)
        wh_cls = self.wh_cls(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)
        keypoints_pred = self.keypoints_pred(bbox_subnet)

        return object_pred, cls_score, bbox_offset, wh_cls, keypoints_pred


class YOLOFHeadSelfAndKeypointsv2(nn.Module):
    """动态选择 anchor
        同时做 keypoints
    """

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80, num_keypoints=17):
        super().__init__()
        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        self.wh_cls = nn.Conv2d(in_c, num_anchor * 2 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor
        self.keypoint_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.keypoints_logit = nn.Conv2d(in_c, num_keypoints, 3, 1, 1)
        self.keypoints_offset = nn.Conv2d(in_c, 2 * num_keypoints, 3, 1, 1)

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_offset = self.bbox_offset(bbox_subnet)
        wh_cls = self.wh_cls(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)
        #         keypoints_pred = self.keypoints_pred(bbox_subnet)
        keypoint_subnet = self.keypoint_subnet(x)
        keypoints_logit = self.keypoints_logit(keypoint_subnet)
        keypoints_offset = self.keypoints_offset(keypoint_subnet)

        return object_pred, cls_score, bbox_offset, wh_cls, keypoints_logit, keypoints_offset


class YOLOFHeadSelfv2(nn.Module):
    """动态选择 anchor 和 中心点
        do_bbox_offset = True   (1、bbox粗略分类 2、对bbox进一步做回归)
        do_bbox_offset = False  (1、bbox精细分类 只使用分类方式代替回归)
    """

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80, do_bbox_offset=True):
        super().__init__()
        self.do_bbox_offset = do_bbox_offset
        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        if self.do_bbox_offset: self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        self.bbox_cls = nn.Conv2d(in_c, num_anchor * 4 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        if self.do_bbox_offset: bbox_offset = self.bbox_offset(bbox_subnet)
        bbox_cls = self.bbox_cls(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)

        if self.do_bbox_offset:
            return object_pred, cls_score, bbox_offset, bbox_cls
        else:
            bbox_offset = torch.zeros_like(object_pred)  # 仅仅是用于占位
            return object_pred, cls_score, bbox_offset, bbox_cls


class RetinaHead(nn.Module):
    """
    多级语义 共享 head 不使用 BN 可以使用GN
    单层语义 head可以使用BN
    """

    def __init__(self, in_c, num_anchor=9, num_classes=80, dim_transform=True):
        super().__init__()
        self.num_classes = num_classes
        self.dim_transform = dim_transform
        self.cls_convs = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=False) for _ in range(4)])
        self.reg_convs = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=False) for _ in range(4)])
        self.retina_cls = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.retina_reg = nn.Conv2d(in_c, 4 * num_anchor, 3, 1, 1)

    def forward(self, x):
        cls_list, reg_list = [], []
        for _x in x:
            cls, reg = self.forward_single(_x)
            cls_list.append(cls)
            reg_list.append(reg)

        if self.dim_transform:
            cls_list = torch.cat(cls_list, 1)
            reg_list = torch.cat(reg_list, 1)

        return cls_list, reg_list

    def forward_single(self, x):
        cls = self.retina_cls(self.cls_convs(x))
        reg = self.retina_reg(self.reg_convs(x))

        if self.dim_transform:
            # bs, c, h, w = cls.shape
            bs = cls.size(0)
            # cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors[i], self.num_classes)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        return cls, reg


class RetinaHeadV2(nn.Module):
    """
    多级语义 共享 head 不使用 BN 可以使用GN
    单层语义 head可以使用BN
    """

    def __init__(self, in_c, num_anchor=[9, 9, 9, 9, 9], num_classes=80, dim_transform=True):
        super().__init__()
        self.num_anchor = num_anchor
        self.num_classes = num_classes
        self.dim_transform = dim_transform
        self.cls_convs = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=False) for _ in range(4)])
        self.reg_convs = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=False) for _ in range(4)])
        self.retina_cls = nn.ModuleList(
            [nn.Conv2d(in_c, num_anchor[i] * num_classes, 3, 1, 1) for i in range(len(num_anchor))])
        self.retina_reg = nn.ModuleList([nn.Conv2d(in_c, 4 * num_anchor[i], 3, 1, 1) for i in range(len(num_anchor))])

    def forward(self, x):
        assert len(x) == len(self.num_anchor)
        cls_list, reg_list = [], []
        for i, _x in enumerate(x):
            cls, reg = self.forward_single(_x, i)
            cls_list.append(cls)
            reg_list.append(reg)

        if self.dim_transform:
            cls_list = torch.cat(cls_list, 1)
            reg_list = torch.cat(reg_list, 1)

        return cls_list, reg_list

    def forward_single(self, x, i):
        cls = self.retina_cls[i](self.cls_convs(x))
        reg = self.retina_reg[i](self.reg_convs(x))

        if self.dim_transform:
            # bs, c, h, w = cls.shape
            bs = cls.size(0)
            # cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors[i], self.num_classes)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        return cls, reg


class RpnHead(nn.Module):
    def __init__(self, in_c, num_anchor=15, dim_transform=True):
        super().__init__()
        self.dim_transform = dim_transform
        self.rpn_conv = nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rpn_cls = nn.Conv2d(in_c, num_anchor, kernel_size=(1, 1), stride=(1, 1))
        self.rpn_reg = nn.Conv2d(in_c, num_anchor * 4, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            cls_list, reg_list = [], []
            for _x in x:
                cls, reg = self.forward_single(_x)
                cls_list.append(cls)
                reg_list.append(reg)

            if self.dim_transform:
                cls_list = torch.cat(cls_list, 1)
                reg_list = torch.cat(reg_list, 1)

            return cls_list, reg_list
        else:
            return self.forward_single(x)

    def forward_single(self, x):
        x = F.relu(self.rpn_conv(x))
        cls = self.rpn_cls(x)
        reg = self.rpn_reg(x)

        if self.dim_transform:
            # bs, c, h, w = cls.shape
            bs = cls.size(0)
            # cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors[i], self.num_classes)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(bs, -1)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        return cls, reg


class RoiHeadC4(nn.Module):
    """r50_c4
    roi 对齐到 c4
    """

    def __init__(self, C5, stride=16):
        super().__init__()
        self.roi_layers = RoIAlign(output_size=(14, 14), spatial_scale=1 / stride, sampling_ratio=0,
                                   aligned=True)  # 对齐到 c4
        self.shared_head = C5  # r50_c5

    def forward(self, x, proposal):
        return self.shared_head(self.roi_layers(x, proposal))


class FasterRcnnBBoxHeadC4(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7), padding=0)
        self.fc_cls = nn.Linear(in_features=in_c, out_features=num_classes, bias=True)
        self.fc_reg = nn.Linear(in_features=in_c, out_features=4 * (num_classes - 1), bias=True)

    def forward(self, x):
        x = self.avg_pool(x).flatten(1)
        return self.fc_cls(x), self.fc_reg(x)


class FasterRcnnBBoxHeadC4V2(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7), padding=0)
        self.fc_cls = nn.Linear(in_features=in_c, out_features=num_classes, bias=True)
        # self.fc_reg = nn.Linear(in_features=in_c, out_features=4 * (num_classes - 1), bias=True)

    def forward(self, x):
        x = self.avg_pool(x).flatten(1)
        # return self.fc_cls(x), self.fc_reg(x)
        return self.fc_cls(x)


class RoiHeadC5(nn.Module):
    """
    将c5的stride 从2改成1 整体网络的stride=16
    roi 对齐到 c5
    """

    def __init__(self, stride=16):
        super().__init__()
        self.roi_layers = RoIAlign(output_size=(7, 7), spatial_scale=1 / stride, sampling_ratio=0,
                                   aligned=True)  # 对齐到 c5

    def forward(self, x, proposal):
        return self.roi_layers(x, proposal)


class FasterRcnnBBoxHeadC5(nn.Module):
    def __init__(self, roi_in_c, in_c=1024, num_classes=81):
        super().__init__()
        self.shared_fcs = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=roi_in_c, out_features=in_c, bias=True),
                                        nn.Linear(in_features=in_c, out_features=in_c, bias=True))
        self.fc_cls = nn.Linear(in_features=in_c, out_features=num_classes, bias=True)
        self.fc_reg = nn.Linear(in_features=in_c, out_features=4 * (num_classes - 1), bias=True)

    def forward_single(self, x):
        x = self.shared_fcs(x)
        return self.fc_cls(x), self.fc_reg(x)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            cls_list, reg_list = [], []
            for _x in x:
                _cls, _reg = self.forward_single(_x)
                cls_list.append(_cls)
                reg_list.append(_reg)
            return cls_list, reg_list
        else:
            return self.forward_single(x)


class FasterrcnnFPN(nn.Module):
    def __init__(self, in_c, out_c=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_c // 8, out_c, kernel_size=(1, 1), stride=(1, 1)),
                                            nn.Conv2d(in_c // 4, out_c, kernel_size=(1, 1), stride=(1, 1)),
                                            nn.Conv2d(in_c // 2, out_c, kernel_size=(1, 1), stride=(1, 1)),
                                            nn.Conv2d(in_c // 1, out_c, kernel_size=(1, 1), stride=(1, 1))
                                            ])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                        ])
        self.upsample = nn.Upsample(None, 2, 'nearest')

    def forward(self, x):
        m5 = self.lateral_convs[3](x[3])
        p5 = self.fpn_convs[3](m5)
        m4 = self.upsample(m5) + self.lateral_convs[2](x[2])
        p4 = self.fpn_convs[2](m4)
        m3 = self.upsample(m4) + self.lateral_convs[1](x[1])
        p3 = self.fpn_convs[1](m3)
        m2 = self.upsample(m3) + self.lateral_convs[0](x[0])
        p2 = self.fpn_convs[0](m2)

        p6 = F.max_pool2d(p5, 2, 2)

        return p2, p3, p4, p5, p6


class RoiHeadFPN(nn.Module):
    def __init__(self, strides=[4, 8, 16, 32]):
        super().__init__()
        self.roi_layers = nn.ModuleList([RoIAlign(output_size=(7, 7), spatial_scale=1 / stride, sampling_ratio=0,
                                                  aligned=True) for stride in strides])

    def forward(self, x, proposal):
        return [self.roi_layers[i](_x, proposal[i]) for i, _x in enumerate(x)]


class MaskRoiHeadFPN(nn.Module):
    def __init__(self, strides=[4, 8, 16, 32]):
        super().__init__()
        self.roi_layers = nn.ModuleList([RoIAlign(output_size=(14, 14), spatial_scale=1 / stride, sampling_ratio=0,
                                                  aligned=True) for stride in strides])

    def forward(self, x, proposal):
        return [self.roi_layers[i](_x, proposal[i]) for i, _x in enumerate(x)]


class MaskRoiHead(nn.Module):
    def __init__(self, stride=16):
        super().__init__()
        self.roi_layers = RoIAlign(output_size=(14, 14), spatial_scale=1 / stride, sampling_ratio=0, aligned=True)

    def forward(self, x, proposal):
        return self.roi_layers(x, proposal)


class MaskHead(nn.Module):
    def __init__(self, in_c=256, num_classes=80):  # 不含背景
        super().__init__()
        self.convs = nn.Sequential(*[CBA(in_c, in_c, 3, 1, use_bn=False) for _ in range(4)])
        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.conv_logits = nn.Conv2d(in_c, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward_single(self, x):
        x = self.convs(x)
        x = self.upsample(x)
        x = self.relu(x)
        x = self.conv_logits(x)
        return x

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            out = []
            for _x in x:
                out.append(self.forward_single(_x))
            return out
        else:
            return self.forward_single(x)


class KeypointHead(nn.Module):
    def __init__(self, in_c=256, num_classes=17):  # 17个关键点
        super().__init__()
        self.convs = nn.Sequential(*[CBA(in_c, in_c, 3, 1, use_bn=False) for _ in range(4)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = nn.Sequential(nn.Flatten(), nn.Linear(in_c, num_classes))

    def forward_single(self, x):
        x = self.convs(x)
        x = self.gap(x)
        x = self.logits(x)

        return x

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            out = []
            for _x in x:
                out.append(self.forward_single(_x))
            return out
        else:
            return self.forward_single(x)


class KeypointHeadHeatMap(nn.Module):
    def __init__(self, in_c=256, num_classes=17):  # 17个关键点
        super().__init__()
        self.convs = nn.Sequential(*[CBA(in_c, in_c, 3, 1, use_bn=False) for _ in range(4)])
        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.upsample2 = nn.ConvTranspose2d(in_c, in_c, kernel_size=(2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_logits = nn.Conv2d(in_c, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward_single(self, x):
        x = self.convs(x)
        x = self.upsample(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.relu2(x)
        x = self.conv_logits(x)
        return x

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            out = []
            for _x in x:
                out.append(self.forward_single(_x))
            return out
        else:
            return self.forward_single(x)


class UnetHead(nn.Module):
    def __init__(self, in_c=256, num_classes=81):  # 含背景
        super().__init__()
        # self.up_c5 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2), nn.ReLU(inplace=True))
        # self.up_c4 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2), nn.ReLU(inplace=True))
        # self.up_c3 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2), nn.ReLU(inplace=True))
        # self.up_c2 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2), nn.ReLU(inplace=True))
        # self.up_c1 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 4, 0, 2), nn.ReLU(inplace=True),
        #                            nn.Conv2d(in_c, num_classes, 3, 2, 2, 2))

        self.up_c5 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2, 1, 1, groups=8, dilation=2),
                                   nn.ReLU(inplace=True))
        self.up_c4 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2, 1, 1, groups=8, dilation=2),
                                   nn.ReLU(inplace=True))
        self.up_c3 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2, 1, 1, groups=8, dilation=2),
                                   nn.ReLU(inplace=True))
        self.up_c2 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2, 1, 1, groups=8, dilation=2),
                                   nn.ReLU(inplace=True))
        self.up_c1 = nn.Sequential(nn.ConvTranspose2d(in_c, in_c, 2, 2, 1, 1, groups=8, dilation=2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_c, num_classes, 3, 1, 2, 2))

    def forward(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        m5 = self.up_c5(c5)
        m4 = self.up_c4(c4 + m5)
        m3 = self.up_c3(c3 + m4)
        m2 = self.up_c2(c2 + m3)
        m1 = self.up_c1(m2)

        return m1


# @NECKS.register_module
class NASFPN(nn.Module):
    """NAS-FPN.

    NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object
    Detection. (https://arxiv.org/abs/1904.07392)
    """

    # retinanet_nasfpn
    # neck=dict(
    #     type='NASFPN',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5,
    #     stack_times=7,
    #     start_level=1,
    #     add_extra_convs=True,
    #     norm_cfg=norm_cfg)
    # norm_cfg = dict(type='BN', requires_grad=True)

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 out_channels=256,
                 num_outs=5,
                 stack_times=7,
                 start_level=0,
                 # end_level=-1,
                 # add_extra_convs=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_times = stack_times
        self.start_level = start_level
        self.backbone_end_level = len(in_channels)
        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                activation=None)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                norm_cfg=norm_cfg,
                activation=None)
            self.extra_downsamples.append(
                nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))

        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GPCell(out_channels, norm_cfg=norm_cfg)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GPCell(with_conv=False)
            stage['sum_55_5'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GPCell(with_conv=False)
            stage['sum_77_7'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GPCell(out_channels, norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)

    def forward(self, inputs):
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build P6-P7 on top of P5
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return p3, p4, p5, p6, p7


# PAN
class PAFPN(nn.Module):
    def __init__(self, in_c, out_c=256, mode='fpn'):
        super().__init__()
        self.mode = mode
        self.lateral_convs = nn.ModuleList()
        self.lateral_convs.append(nn.Conv2d(in_c // 8, out_c, kernel_size=(1, 1), stride=(1, 1)))
        self.lateral_convs.append(nn.Conv2d(in_c // 4, out_c, kernel_size=(1, 1), stride=(1, 1)))
        self.lateral_convs.append(nn.Conv2d(in_c // 2, out_c, kernel_size=(1, 1), stride=(1, 1)))
        self.lateral_convs.append(nn.Conv2d(in_c // 1, out_c, kernel_size=(1, 1), stride=(1, 1)))

        self.upsample = nn.Upsample(None, 2)

        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(4)])

        if self.mode != 'fpn':
            self.downsample_convs = nn.ModuleList(
                [nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) for _ in range(3)])

            self.pafpn_convs = nn.ModuleList(
                [nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(4)])

    def forward(self, x):
        # x4,x8,x16,x32 = x
        # top to down
        m5 = self.lateral_convs[3](x[3])
        m4 = self.upsample(m5) + self.lateral_convs[2](x[2])
        m3 = self.upsample(m4) + self.lateral_convs[1](x[1])
        m2 = self.upsample(m3) + self.lateral_convs[0](x[0])

        p5 = self.fpn_convs[3](m5)
        p4 = self.fpn_convs[2](m4)
        p3 = self.fpn_convs[1](m3)
        p2 = self.fpn_convs[0](m2)

        if self.mode != 'fpn':
            # down to top
            m3 = self.downsample_convs[0](p2) + p3
            m4 = self.downsample_convs[1](m3) + p4
            m5 = self.downsample_convs[2](m4) + p5

            p3 = self.pafpn_convs[0](m3)
            p4 = self.pafpn_convs[1](m4)
            p5 = self.pafpn_convs[2](m5)

        return p2, p3, p4, p5


# ---------------------------------test---------------------------
def test_ssd():
    x = torch.rand([1, 3, 300, 300])
    backbone = Vgg16Backbone(model_name='vgg16')
    x = backbone.forwardMS(x)
    head = SSDHead300(81)
    cls_list, reg_list = head.forward(x)
    print(cls_list[0].shape, reg_list[0].shape, cls_list[-1].shape, reg_list[-1].shape)


def test_yolox():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50')
    x = backbone.forwardMS(x)
    neck = FPN(backbone.out_channels, 256, True)
    x = neck(x)
    head = YOLOXHead(256)
    cls_list, reg_list, obj_list = head(x)
    print()


def test_yolof():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50')
    x = backbone.forward(x)
    neck = YOLOFNeck(backbone.out_channels)
    x = neck(x)
    head = YOLOFHead(backbone.out_channels // 4)
    cls_score, bbox_pred, object_pred = head(x)
    print(cls_score.shape, bbox_pred.shape, object_pred.shape)


def test_yolov1():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone()
    x = backbone.forward(x)
    neck = YOLOV1Neck(backbone.out_channels, True)
    x = neck(x)
    head = YOLOV1Head(backbone.out_channels // 2)
    x = head(x)
    print(x.shape)


def test_yolov3():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone()
    x = backbone.forwardMS(x)
    neck = YOLOV3Neck(backbone.out_channels, True)
    x = neck(x)
    head = YOLOV3Head(backbone.out_channels // 2)
    x8, x16, x32 = head(x)
    print(x8.shape, x16.shape, x32.shape)


def test_centernet():
    devcie = "cuda:0"
    x = torch.rand([1, 3, 320, 320]).to(devcie)
    backbone = Backbone().to(devcie)
    neck = CenterNetNeck(backbone.out_channels, backbone.out_channels // 2, True).to(devcie)
    head = CenterNetHead(backbone.out_channels // 8, 20).to(devcie)
    x = backbone(x)
    x = neck(x)
    headmap, wh, reg = head.forward(x)
    print(headmap.shape, wh.shape, reg.shape)


def test_fcos():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone()
    fpn = FPN(backbone.out_channels)
    head = FCOSHead()

    a = backbone.forwardMS(x)
    b = fpn(a)
    c = head(b)
    print()


def test_retinanet():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone()
    fpn = FPN(backbone.out_channels, 256)
    head = RetinaHead(256)

    a = backbone.forwardMS(x)
    b = fpn(a)
    c = head(b)
    print()


def test_faster_rcnn_c4():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50')
    _, _, c4, _ = backbone.forwardMS(x)
    rpn = RpnHead(backbone.out_channels // 2)
    rpn_out = rpn(c4)
    roihead = RoiHeadC4(backbone.backbone[-1])
    # from torchvision.ops import RoIAlign
    proposal = torch.tensor([[10, 20, 100, 100], [15, 40, 130, 170]]).float()
    roi_out = roihead(c4, [proposal])

    # from mmcv.ops import RoIAlign
    # proposal = torch.tensor([[0, 10, 20, 100, 100], [0, 15, 40, 130, 170]]).float()  # 第一列 对应 不同的特征map层，因为只有一个所以其索引为0
    # roi_out = roihead(c4, proposal)
    bboxhead = FasterRcnnBBoxHeadC4(backbone.out_channels, 81)

    rcnn_out = bboxhead(roi_out)


def test_faster_rcnn_c5():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50', stride=16)
    c5 = backbone.forward(x)
    rpn = RpnHead(backbone.out_channels)
    rpn_out = rpn(c5)
    roihead = RoiHeadC5(stride=16)
    # from torchvision.ops import RoIAlign
    proposal = torch.tensor([[10, 20, 100, 100], [15, 40, 130, 170]]).float()
    roi_out = roihead(c5, [proposal])

    # from mmcv.ops import RoIAlign
    # proposal = torch.tensor([[0, 10, 20, 100, 100], [0, 15, 40, 130, 170]]).float()  # 第一列 对应 不同的特征map层，因为只有一个所以其索引为0
    # roi_out = roihead(c5, proposal)

    bboxhead = FasterRcnnBBoxHeadC5(backbone.out_channels * 7 * 7, backbone.out_channels // 2, 81)

    rcnn_out = bboxhead(roi_out)


def test_faster_rcnn_fpn():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50')
    x = backbone.forwardMS(x)
    fpn = FasterrcnnFPN(backbone.out_channels)
    p2, p3, p4, p5, p6 = fpn(x)
    print(p5.shape)
    print(p6.shape)

    rpn = RpnHead(256, 3)
    rpn_out = rpn((p2, p3, p4, p5, p6))

    # 第一列 对应 不同的特征map层
    proposal = torch.tensor([[10, 20, 100, 100], [15, 40, 130, 170]]).float()
    roihead = RoiHeadFPN()
    roi_out = roihead((p2, p3, p4, p5), [[proposal], [proposal], [proposal], [proposal]])

    bboxhead = FasterRcnnBBoxHeadC5(256 * 7 * 7, 1024, 81)
    rcnn_out = bboxhead(roi_out)
    print()


def test_mask_rcnn_fpn():
    x = torch.rand([1, 3, 320, 320])
    backbone = Backbone('resnet50')
    x = backbone.forwardMS(x)
    fpn = FasterrcnnFPN(backbone.out_channels)
    p2, p3, p4, p5, p6 = fpn(x)
    print(p5.shape)
    print(p6.shape)

    rpn = RpnHead(256, 3)
    rpn_out = rpn((p2, p3, p4, p5, p6))

    # 第一列 对应 不同的特征map层
    proposal = torch.tensor([[10, 20, 100, 100], [15, 40, 130, 170]]).float()
    roihead = RoiHeadFPN()
    roi_out = roihead((p2, p3, p4, p5), [[proposal], [proposal], [proposal], [proposal]])

    bboxhead = FasterRcnnBBoxHeadC5(256 * 7 * 7, 1024, 81)
    rcnn_out = bboxhead(roi_out)

    maskroihead = MaskRoiHeadFPN()
    mroi_out = maskroihead((p2, p3, p4, p5), [[proposal], [proposal], [proposal], [proposal]])

    mhead = MaskHead(256, 80)
    m_out = mhead(mroi_out)
    print()


def viewMMdet():
    from toolcv.api.define.utils.model.mmdet import load_model

    configs = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
    model = load_model(configs, None, 0)
    print(model)


if __name__ == "__main__":
    """
    # 查看模型结构(mmdetection)
    from toolcv.api.define.utils.model.mmdet import load_model

    configs = 'configs/yolo/yolov3_d53_320_273e_coco.py'
    model = load_model(configs,None,0)
    print(model)
    """

    # test_centernet()
    # test_fcos()
    # test_yolov3()
    # test_yolov1()
    # test_yolof()
    # test_yolox()

    # test_ssd()
    # test_retinanet()

    # test_faster_rcnn_c4()
    # test_faster_rcnn_c5()
    # test_faster_rcnn_fpn()
    # test_mask_rcnn_fpn()

    # backbone = Backbone('dla34', stride=16)
    backbone = BackboneDilate('resnet50', True, stride=16)
    x = torch.rand([1, 3, 224, 224])
    out = backbone(x)
    print(out.shape)
