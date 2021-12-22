import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from collections import OrderedDict

from toolcv.models.pytorch import dla


def _initParmas(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.01)
            # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                # nn.init.zeros_(m.bias)
                nn.init.constant_(m.bias, 0)


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
            CBL(in_channels * 4, in_channels, 1, 1, 0)
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


# 单分支
class Yolov1simple(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, stride=16, dropout=0.5, boxes_coord_nums=5,freeze_at=4):
        super().__init__()
        self.boxes_coord_nums = boxes_coord_nums
        # out_filler = num_anchors*5+num_classes
        out_filler = num_anchors * (boxes_coord_nums + num_classes)  # num_classes 不包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        if 'resnet' in model_name or 'resnext' in model_name:
            # model = resnet18(True)
            # model = resnet34(True)
            # model = resnet50(True)
            model = resnet.__dict__[model_name](pretrained)

            if stride == 16:
                # stride = 16
                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

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
            if stride == 16:
                _model.level5.tree1.conv1.stride = (1, 1)
                _model.level5.tree1.conv2.stride = (1, 1)
                # _model.level5.downsample.stride = 1
                _model.level5.downsample = nn.Sequential()

            # self.backbone = nn.ModuleDict(OrderedDict([  # nn.Sequential
            #     ("res1", nn.Sequential(_model.base_layer, _model.level0, _model.level1)),
            #     ("res2", _model.level2),
            #     ("res3", _model.level3),
            #     ("res4", _model.level4),
            #     ("res5", _model.level5)
            # ]))

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

        # self.logit = nn.Sequential(
        #     SPPNet(model.inplanes),
        #     # CBL(model.inplanes, model.inplanes, 3, 1, 1),
        #     nn.Conv2d(model.inplanes, model.inplanes, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(model.inplanes, out_filler, 1, 1, 0)
        # )

        self.logit = nn.Sequential(
            nn.Dropout(dropout),
            # SPPNet(model.inplanes),
            _make_detnet_layer(in_channels=out_channels),
            nn.Conv2d(256, out_filler, 3, 1, 1),
            # nn.BatchNorm2d(out_filler)
        )

        _initParmas(self.logit.modules())

    def forward(self, x):
        x = self.backbone(x)
        x = self.logit(x)
        x = x.permute(0, 2, 3, 1)
        bs, h, w, c = x.shape
        x = x.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)
        # x = x.clamp(0,1)

        return x


# 单分支
class Yolov1simpleV2(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=16, dropout=0.5, boxes_coord_nums=5,freeze_at=4):
        super().__init__()
        self.boxes_coord_nums = boxes_coord_nums
        out_filler = num_anchors * (boxes_coord_nums + num_classes)  # num_classes 不包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.strides = strides

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

        self.spp = SPPNet(out_channels)

        if self.strides == 32:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            _initParmas(self.logit_32.modules())

        if self.strides == 16:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_16 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 2 + 256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.logit_16.modules())
            _initParmas(self.logit_32.modules())

        if self.strides == 8:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_16 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 2 + 256),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_8 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 4 + 256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.logit_8.modules())
            _initParmas(self.logit_16.modules())
            _initParmas(self.logit_32.modules())

        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        if self.strides == 32:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            x32 = self.logit_32[2](_x32)

            x32 = x32.permute(0, 2, 3, 1)
            bs, h, w, c = x32.shape
            x32 = x32.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

            return x32

        if self.strides == 16:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            # x32 = self.logit_32[2](_x32)

            x16 = torch.cat((self.upsample(_x32), x16), 1)
            _x16 = self.logit_16[:2](x16)
            x16 = self.logit_16[2](_x16)

            x16 = x16.permute(0, 2, 3, 1)
            bs, h, w, c = x16.shape
            x16 = x16.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

            return x16

        if self.strides == 8:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            # x32 = self.logit_32[2](_x32)

            x16 = torch.cat((self.upsample(_x32), x16), 1)
            _x16 = self.logit_16[:2](x16)
            # x16 = self.logit_16[2](_x16)

            x8 = torch.cat((self.upsample(_x16), x8), 1)
            x8 = self.logit_8(x8)

            x8 = x8.permute(0, 2, 3, 1)
            bs, h, w, c = x8.shape
            x8 = x8.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

            return x8


# 多分支 s=8,16,32
class Yolov3simple(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=[8, 16, 32], dropout=0.5, boxes_coord_nums=5,freeze_at=4):
        super().__init__()
        self.boxes_coord_nums = boxes_coord_nums
        # out_filler = num_anchors*5+num_classes
        out_filler = num_anchors * (boxes_coord_nums + num_classes)  # num_classes 不包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes

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

        self.spp = SPPNet(out_channels)

        self.logit_8 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 4 + 256),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_16 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 2 + 256),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_32 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )


        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())
        _initParmas(self.logit_8.modules())
        _initParmas(self.logit_16.modules())
        _initParmas(self.logit_32.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        x32 = self.spp(x32)
        _x32 = self.logit_32[:2](x32)
        x32 = self.logit_32[2](_x32)

        x16 = torch.cat((self.upsample(_x32), x16), 1)
        _x16 = self.logit_16[:2](x16)
        x16 = self.logit_16[2](_x16)

        x8 = torch.cat((self.upsample(_x16), x8), 1)
        x8 = self.logit_8(x8)


        x8 = x8.permute(0, 2, 3, 1)
        bs, h, w, c = x8.shape
        x8 = x8.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)
        bs, h, w, c = x16.shape
        x16 = x16.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)
        bs, h, w, c = x32.shape
        x32 = x32.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        return x8, x16, x32

class Yolov3simpleV2(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=[8, 16, 32], dropout=0.5, boxes_coord_nums=5,freeze_at=4,share_head=True):
        super().__init__()
        self.boxes_coord_nums = boxes_coord_nums
        # out_filler = num_anchors*5+num_classes
        out_filler = num_anchors * (boxes_coord_nums + num_classes)  # num_classes 不包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.share_head = share_head

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

        self.spp = SPPNet(out_channels)

        self.logit_8 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 4 + 256),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_16 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 2 + 256),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_32 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )

        if self.share_head:
            self.head = nn.Sequential(
                _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3,1,1)
            )
            _initParmas(self.head.modules())
        else:
            self.head8 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.head16 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.head32 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.head8.modules())
            _initParmas(self.head16.modules())
            _initParmas(self.head32.modules())

        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())
        _initParmas(self.logit_8.modules())
        _initParmas(self.logit_16.modules())
        _initParmas(self.logit_32.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        x32 = self.spp(x32)
        _x32 = self.logit_32[:2](x32)
        # x32 = self.logit_32[2](_x32)

        x16 = torch.cat((self.upsample(_x32), x16), 1)
        _x16 = self.logit_16[:2](x16)
        # x16 = self.logit_16[2](_x16)

        x8 = torch.cat((self.upsample(_x16), x8), 1)
        x8 = self.logit_8(x8)

        if self.share_head:
            x32 = self.head(_x32)
            x16 = self.head(_x16)
            x8 = self.head(x8)
        else:
            x32 = self.head32(_x32)
            x16 = self.head16(_x16)
            x8 = self.head8(x8)

        x8 = x8.permute(0, 2, 3, 1)
        bs, h, w, c = x8.shape
        x8 = x8.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)
        bs, h, w, c = x16.shape
        x16 = x16.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)
        bs, h, w, c = x32.shape
        x32 = x32.contiguous().view(bs, h, w, self.num_anchors, self.boxes_coord_nums + self.num_classes)

        return x8, x16, x32

# 单分支
class SSDsimple(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1, num_classes=1,
                 stride=16, dropout=0.5,freeze_at=4):
        super().__init__()
        # out_filler = num_anchors*5+num_classes
        out_filler = num_anchors * (4 + num_classes)  # num_classes 包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        if 'resnet' in model_name or 'resnext' in model_name:
            # model = resnet18(True)
            # model = resnet34(True)
            # model = resnet50(True)
            model = resnet.__dict__[model_name](pretrained)

            if stride == 16:
                # stride = 16
                model.layer4[0].conv1.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].conv2.stride = (1, 1)  # (2,2)->(1,1)
                model.layer4[0].downsample[0].stride = (1, 1)  # (2,2)->(1,1)

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
            if stride == 16:
                _model.level5.tree1.conv1.stride = (1, 1)
                _model.level5.tree1.conv2.stride = (1, 1)
                # _model.level5.downsample.stride = 1
                _model.level5.downsample = nn.Sequential()

            # self.backbone = nn.ModuleDict(OrderedDict([  # nn.Sequential
            #     ("res1", nn.Sequential(_model.base_layer, _model.level0, _model.level1)),
            #     ("res2", _model.level2),
            #     ("res3", _model.level3),
            #     ("res4", _model.level4),
            #     ("res5", _model.level5)
            # ]))

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

        # self.logit = nn.Sequential(
        # nn.Dropout(dropout),
        #     SPPNet(model.inplanes),
        #     # CBL(model.inplanes, model.inplanes, 3, 1, 1),
        #     nn.Conv2d(model.inplanes, model.inplanes, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(model.inplanes, out_filler, 1, 1, 0)
        # )

        self.logit = nn.Sequential(
            nn.Dropout(dropout),
            # SPPNet(model.inplanes),
            _make_detnet_layer(in_channels=out_channels),
            nn.Conv2d(256, out_filler, 3, 1, 1),
            # nn.BatchNorm2d(out_filler)
        )

        _initParmas(self.logit.modules())

    def forward(self, x):
        x = self.backbone(x)
        x = self.logit(x)
        x = x.permute(0, 2, 3, 1)
        # bs, h, w, c = x.shape
        # x = x.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        # x = x.contiguous().view(bs, -1, 4 + self.num_classes)
        x = x.contiguous().view(-1, 4 + self.num_classes)
        # x = x.clamp(0,1)

        return x


# 单分支
class SSDsimpleV2(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=16, dropout=0.5,freeze_at=4):
        super().__init__()
        out_filler = num_anchors * (4 + num_classes)  # num_classes 包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.strides = strides

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

        self.spp = SPPNet(out_channels)

        if self.strides == 32:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            _initParmas(self.logit_32.modules())

        if self.strides == 16:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_16 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 2 + 256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.logit_16.modules())
            _initParmas(self.logit_32.modules())

        if self.strides == 8:
            self.logit_32 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_16 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 2 + 256),
                # nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.logit_8 = nn.Sequential(
                nn.Dropout(dropout),
                _make_detnet_layer(in_channels=out_channels // 4 + 256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.logit_8.modules())
            _initParmas(self.logit_16.modules())
            _initParmas(self.logit_32.modules())

        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        if self.strides == 32:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            x32 = self.logit_32[2](_x32)

            x32 = x32.permute(0, 2, 3, 1)
            # bs, h, w, c = x32.shape
            # x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
            # x32 = x32.contiguous().view(bs, -1, 4 + self.num_classes)
            x32 = x32.contiguous().view(-1, 4 + self.num_classes)

            return x32

        if self.strides == 16:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            # x32 = self.logit_32[2](_x32)

            x16 = torch.cat((self.upsample(_x32), x16), 1)
            _x16 = self.logit_16[:2](x16)
            x16 = self.logit_16[2](_x16)

            x16 = x16.permute(0, 2, 3, 1)
            # bs, h, w, c = x16.shape
            # x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
            # x16 = x16.contiguous().view(bs, -1, 4 + self.num_classes)
            x16 = x16.contiguous().view(-1, 4 + self.num_classes)

            return x16

        if self.strides == 8:
            x32 = self.spp(x32)
            _x32 = self.logit_32[:2](x32)
            # x32 = self.logit_32[2](_x32)

            x16 = torch.cat((self.upsample(_x32), x16), 1)
            _x16 = self.logit_16[:2](x16)
            # x16 = self.logit_16[2](_x16)

            x8 = torch.cat((self.upsample(_x16), x8), 1)
            x8 = self.logit_8(x8)

            x8 = x8.permute(0, 2, 3, 1)
            # bs, h, w, c = x8.shape
            # x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
            # x8 = x8.contiguous().view(bs, -1, 4 + self.num_classes)
            x8 = x8.contiguous().view(-1, 4 + self.num_classes)

            return x8


# 多分支 s=8,16,32
class SSDsimpleMS(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=[8, 16, 32], dropout=0.5,freeze_at=4):
        super().__init__()
        out_filler = num_anchors * (4 + num_classes)  # num_classes 包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes

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

        self.spp = SPPNet(out_channels)

        self.logit_8 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 4 + 256),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_16 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 2 + 256),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_32 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels),
            nn.Conv2d(256, out_filler, 3, 1, 1)
        )


        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())
        _initParmas(self.logit_8.modules())
        _initParmas(self.logit_16.modules())
        _initParmas(self.logit_32.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        x32 = self.spp(x32)
        _x32 = self.logit_32[:2](x32)
        x32 = self.logit_32[2](_x32)

        x16 = torch.cat((self.upsample(_x32), x16), 1)
        _x16 = self.logit_16[:2](x16)
        x16 = self.logit_16[2](_x16)

        x8 = torch.cat((self.upsample(_x16), x8), 1)
        x8 = self.logit_8(x8)


        x8 = x8.permute(0, 2, 3, 1)
        bs, h, w, c = x8.shape
        # x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x8 = x8.contiguous().view(bs, -1, 4 + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)
        bs, h, w, c = x16.shape
        # x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x16 = x16.contiguous().view(bs, -1, 4 + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)
        bs, h, w, c = x32.shape
        # x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x32 = x32.contiguous().view(bs, -1, 4 + self.num_classes)

        return torch.cat((x8, x16, x32), 1).contiguous().view(-1, 4 + self.num_classes)

class SSDsimpleMSV2(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, num_anchors=1,
                 num_classes=1, strides=[8, 16, 32], dropout=0.5,freeze_at=4,share_head=True):
        super().__init__()
        out_filler = num_anchors * (4 + num_classes)  # num_classes 包括背景
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.share_head = share_head

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

        self.spp = SPPNet(out_channels)

        self.logit_8 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 4 + 256),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_16 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels // 2 + 256),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )
        self.logit_32 = nn.Sequential(
            nn.Dropout(dropout),
            _make_detnet_layer(in_channels=out_channels),
            # nn.Conv2d(256, out_filler, 3, 1, 1)
        )


        if self.share_head:
            self.head = nn.Sequential(
                _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3,1,1)
            )
            _initParmas(self.head.modules())
        else:
            self.head8 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.head16 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )
            self.head32 = nn.Sequential(
                # _make_detnet_layer(in_channels=256),
                nn.Conv2d(256, out_filler, 3, 1, 1)
            )

            _initParmas(self.head8.modules())
            _initParmas(self.head16.modules())
            _initParmas(self.head32.modules())

        self.upsample = nn.Upsample(None, 2, 'bilinear', align_corners=True)

        _initParmas(self.spp.modules())
        _initParmas(self.logit_8.modules())
        _initParmas(self.logit_16.modules())
        _initParmas(self.logit_32.modules())

    def forward(self, x):
        x8 = self.backbone[:3](x)
        x16 = self.backbone[3](x8)
        x32 = self.backbone[4](x16)

        x32 = self.spp(x32)
        _x32 = self.logit_32[:2](x32)
        # x32 = self.logit_32[2](_x32)

        x16 = torch.cat((self.upsample(_x32), x16), 1)
        _x16 = self.logit_16[:2](x16)
        # x16 = self.logit_16[2](_x16)

        x8 = torch.cat((self.upsample(_x16), x8), 1)
        x8 = self.logit_8(x8)


        if self.share_head:
            x32 = self.head(_x32)
            x16 = self.head(_x16)
            x8 = self.head(x8)
        else:
            x32 = self.head32(_x32)
            x16 = self.head16(_x16)
            x8 = self.head8(x8)

        x8 = x8.permute(0, 2, 3, 1)
        bs, h, w, c = x8.shape
        # x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x8 = x8.contiguous().view(bs, -1, 4 + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)
        bs, h, w, c = x16.shape
        # x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x16 = x16.contiguous().view(bs, -1, 4 + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)
        bs, h, w, c = x32.shape
        # x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 4 + self.num_classes)
        x32 = x32.contiguous().view(bs, -1, 4 + self.num_classes)

        return torch.cat((x8, x16, x32), 1).contiguous().view(-1, 4 + self.num_classes)