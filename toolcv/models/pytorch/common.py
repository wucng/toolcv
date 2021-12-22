# @Time: 2021/4/13 13:35 
# @Author: wucong
# @File: common.py 
# @Software: PyCharm
# @Version: win10 Python 3.7.6
# @Version: torch 1.8.1+cu111 torchvision 0.9.1+cu111
# @Version: tensorflow 2.4.1+cu111  keras 2.4.3
# @Describe:

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from . import ACTIVATE_REGISTRY,SEBLOCK_REGISTRY,CONV_REGISTRY,\
    BOTTLEBLOCK_REGISTRY,BASICBLOCK_REGISTRY,BACKBONE_REGISTRY

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@ACTIVATE_REGISTRY.register()
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

@ACTIVATE_REGISTRY.register()
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

@SEBLOCK_REGISTRY.register()
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

@SEBLOCK_REGISTRY.register()
class SELayerV2(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayerV2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

@CONV_REGISTRY.register()
class CBA(nn.Module):
    def __init__(self,in_c,out_c,ksize=3,stride=1,padding="same",dilation=1,groups=1,bias=False,
                 use_bn=True,activate=True):
        super().__init__()
        if padding=="same":
            padding = (ksize+2*(dilation-1))//2
        else:
            padding = 0
        bias = False if (isinstance(use_bn,bool) and use_bn) or isinstance(use_bn,nn.Module) else True
        self.conv = nn.Conv2d(in_c,out_c,ksize,stride,padding,dilation,groups,bias)
        # self.bn = nn.BatchNorm2d(out_c) if use_bn else nn.Sequential()
        if isinstance(use_bn,bool) and use_bn:
            self.bn = nn.BatchNorm2d(out_c)
        elif isinstance(use_bn,nn.Module):
            self.bn = use_bn
        else:
            self.bn = nn.Sequential()

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

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.flatten(x,1)

@CONV_REGISTRY.register()
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return CBA(inp,oup,kernel_size,stride,groups=inp,activate=relu)

@BASICBLOCK_REGISTRY.register()
class BasicBlock(nn.Module):
    """resnet 基础 block"""
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.l1 = CBA(in_c,hide_c,ksize,1)
        self.l2 = CBA(hide_c,out_c,ksize,stride,activate=False)
        self.downsample = CBA(in_c,out_c,1,stride,activate=False) if stride>1 or in_c!=out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self,x):
        x1 = self.l1(x)
        x1 = self.l2(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x+x1)

@BOTTLEBLOCK_REGISTRY.register()
class BottleBlock(nn.Module):
    """resnet bottle block"""
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.l1 = CBA(in_c,hide_c,1,1)
        self.l2 = CBA(hide_c,hide_c,ksize,stride)
        self.l3 = CBA(hide_c, out_c, 1, 1,activate=False)
        self.downsample = CBA(in_c,out_c,1,stride,activate=False) if stride>1 or in_c!=out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self,x):
        x1 = self.l1(x)
        x1 = self.l2(x1)
        x1 = self.l3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x+x1)

@BOTTLEBLOCK_REGISTRY.register()
class BottleBlockX(nn.Module):
    """resneXt bottle block 采用分组卷积 论文最终采用方式"""
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False,cardinality=32):
        super().__init__()
        self.layers = nn.Sequential(
            CBA(in_c,hide_c,1,1),
            CBA(hide_c,hide_c,ksize,stride,groups=cardinality),
            CBA(hide_c, out_c, 1, 1, activate=False)
        )

        self.downsample = CBA(in_c,out_c,1,stride,activate=False) if stride>1 or in_c!=out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self,x):
        x1 = self.layers(x)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x+x1)

@BOTTLEBLOCK_REGISTRY.register()
class Bottle2Block(nn.Module):
    """res2net bottle block
    https://github.com/Res2Net/Res2Net-PretrainedModels
    """
    def __init__(self, in_c, hide_c, out_c, ksize=3, stride=1,use_se=False,nums=4):
        super().__init__()
        self.nums = nums
        self.stride = stride
        self.conv1x1_1 = CBA(in_c,hide_c,1,1)
        layers = []
        for i in range(nums):
            if i==0:
                layers.append(nn.Sequential() if stride==1 else nn.AvgPool2d(3,2,1))
            else:
                layers.append(CBA(hide_c // nums, hide_c // nums, ksize, stride))
        self.layers = nn.Sequential(*layers)
        self.conv1x1_2 = CBA(hide_c,out_c,1,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self,x):
        x1 = self.conv1x1_1(x)
        x_list = torch.chunk(x1,self.nums,1)
        result = []
        for i in range(self.nums):
            if i==0:
                result.append(self.layers[i](x_list[i]))
            else:
                tmp = x_list[i] if self.stride==2 else x_list[i]+result[i-1]
                result.append(self.layers[i-1](tmp))

        x1 = torch.cat(result,1)
        x1 = self.conv1x1_2(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)

@BOTTLEBLOCK_REGISTRY.register()
class Bottle2BlockX(nn.Module):
    """res2neXt bottle block
    https://github.com/Res2Net/Res2Net-PretrainedModels
    """

    def __init__(self, in_c, hide_c, out_c, ksize=3, stride=1,use_se=False,nums=4,cardinality=32):
        super().__init__()
        self.nums = nums
        self.stride = stride
        self.conv1x1_1 = CBA(in_c,hide_c,1,1)
        layers = []
        for i in range(nums):
            if i==0:
                layers.append(nn.Sequential() if stride==1 else nn.AvgPool2d(3,2,1))
            else:
                tmp = hide_c // nums
                if tmp % cardinality != 0: cardinality = tmp//4
                layers.append(CBA(tmp, tmp, ksize, stride,groups=cardinality))
        self.layers = nn.Sequential(*layers)
        self.conv1x1_2 = CBA(hide_c,out_c,1,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self,x):
        x1 = self.conv1x1_1(x)
        x_list = torch.chunk(x1,self.nums,1)
        result = []
        for i in range(self.nums):
            if i==0:
                result.append(self.layers[i](x_list[i]))
            else:
                tmp = x_list[i] if self.stride==2 else x_list[i]+result[i-1]
                result.append(self.layers[i-1](tmp))

        x1 = torch.cat(result,1)
        x1 = self.conv1x1_2(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)

@CONV_REGISTRY.register()
class GhostModule(nn.Module):
    """
    https://github.com/d-li14/ghostnet.pytorch
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = CBA(inp,init_channels,kernel_size,stride,activate=relu)
        self.cheap_operation = CBA(init_channels,new_channels,dw_size,1,groups=init_channels,activate=relu)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

@BOTTLEBLOCK_REGISTRY.register()
class GhostBottleneck(nn.Module):
    """
    https://github.com/d-li14/ghostnet.pytorch
    """
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        self.shortcut = nn.Sequential() if stride == 1 and inp == oup else \
            nn.Sequential(
            depthwise_conv(inp, inp, kernel_size, stride, relu=False),
            CBA(inp,oup,1,activate=False)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

@BOTTLEBLOCK_REGISTRY.register()
class MobileV3Bottleneck(nn.Module):
    """
    https://github.com/d-li14/mobilenetv3.pytorch
    """
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False):
        super(MobileV3Bottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            CBA(inp,hidden_dim,1),
            # dw
            CBA(hidden_dim,hidden_dim,kernel_size,stride,groups=hidden_dim,activate=False),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            nn.ReLU(inplace=True),
            # pw-linear
            CBA(hidden_dim,oup,1,activate=False)
        )

        self.shortcut = nn.Sequential() if stride == 1 and inp == oup else \
            nn.Sequential(
            depthwise_conv(inp, inp, kernel_size, stride, relu=False),
            CBA(inp,oup,1,activate=False)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

@CONV_REGISTRY.register()
class involution(nn.Module):
    """
    https://arxiv.org/pdf/2103.06255.pdf
    https://github.com/d-li14/involution
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,use_bn=True,activate=True):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = CBA(channels,channels//reduction_ratio,1)
        self.conv2 = CBA(channels//reduction_ratio,kernel_size**2 * self.groups,1,
                         bias=True,use_bn=False,activate=False)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

        self.bn = nn.BatchNorm2d(channels) if use_bn else nn.Sequential()
        if isinstance(activate, bool) and activate:
            self.act = nn.ReLU(inplace=True)
        elif isinstance(activate, nn.Module):
            self.act = activate
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return self.act(self.bn(out))

@BOTTLEBLOCK_REGISTRY.register()
class RedNetBottleneck(nn.Module):
    """
    https://arxiv.org/pdf/2103.06255.pdf
    https://github.com/d-li14/involution

    # https://github.com/d-li14/involution/blob/main/cls/mmcls/models/utils/involution_naive.py
    # https://github.com/d-li14/involution/blob/main/cls/mmcls/models/backbones/rednet.py
    # https://github.com/open-mmlab
    # from mmcv.cnn import ConvModule # pip install mmcv
    # from mmcls.models import ResNet
    """
    def __init__(self,in_c,hide_c,out_c,ksize=7,stride=1,use_se=False):
        super().__init__()
        self.conv1 = CBA(in_c,hide_c,1)
        self.conv2 = involution(hide_c,ksize,stride)
        self.conv3 = CBA(hide_c,out_c,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)

@CONV_REGISTRY.register()
class PSConv2dS(nn.Module):
    """
    https://arxiv.org/pdf/2007.06191.pdf
    https://github.com/d-li14/PSConv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, parts=4, bias=False):
        super().__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift

@BOTTLEBLOCK_REGISTRY.register()
class PSConvBottleneck(nn.Module):
    """
    https://arxiv.org/pdf/2007.06191.pdf
    https://github.com/d-li14/PSConv
    """
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.conv1 = CBA(in_c,hide_c,1)
        self.conv2 = nn.Sequential(PSConv2dS(hide_c,hide_c,ksize,stride),nn.BatchNorm2d(hide_c),nn.ReLU(inplace=True))
        self.conv3 = CBA(hide_c,out_c,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)

@CONV_REGISTRY.register()
class Mixconv(nn.Module):
    """
    - https://arxiv.org/pdf/1907.09595.pdf
    - https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    """
    def __init__(self,in_c,out_c,stride=1):
        super().__init__()
        temp_in_c = in_c//4
        temp_out_c = out_c//4
        self.conv = nn.Sequential(
            depthwise_conv(temp_in_c, temp_out_c, 3, stride),
            depthwise_conv(temp_in_c, temp_out_c, 5, stride),
            depthwise_conv(temp_in_c, temp_out_c, 7, stride),
            depthwise_conv(temp_in_c, temp_out_c, 9, stride),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        x_list = torch.chunk(x,4,1)
        return self.act(torch.cat([self.conv[i](x_list[i]) for i in range(4)],1))

@BOTTLEBLOCK_REGISTRY.register()
class MixConvBottleneck(nn.Module):
    """
    - https://arxiv.org/pdf/1907.09595.pdf
    - https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    """
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.conv1 = CBA(in_c,hide_c,1)
        self.conv2 = Mixconv(hide_c,hide_c,stride)
        self.conv3 = CBA(hide_c,out_c,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)

@CONV_REGISTRY.register()
class LambdaLayer(nn.Module):
    """
    - https://github.com/d-li14/lambda.pytorch
    - https://openreview.net/forum?id=xTJEN-ggl1b
    """
    def __init__(self, d, dk=16, du=1, Nh=4, m=None, r=23, stride=1):
        super(LambdaLayer, self).__init__()
        self.d = d
        self.dk = dk
        self.du = du
        self.Nh = Nh
        assert d % Nh == 0, 'd should be divided by Nh'
        dv = d // Nh
        self.dv = dv
        assert stride in [1, 2]
        self.stride = stride

        self.conv_qkv = nn.Conv2d(d, Nh * dk + dk * du + dv * du, 1, bias=False)
        self.norm_q = nn.BatchNorm2d(Nh * dk)
        self.norm_v = nn.BatchNorm2d(dv * du)
        self.softmax = nn.Softmax(dim=-1)
        self.lambda_conv = nn.Conv3d(du, dk, (1, r, r), padding = (0, (r - 1) // 2, (r - 1) // 2))

        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape

        qkv = self.conv_qkv(x)
        q, k, v = torch.split(qkv, [self.Nh * self.dk, self.dk * self.du, self.dv * self.du], dim=1)
        q = self.norm_q(q).view(N, self.Nh, self.dk, H*W)
        v = self.norm_v(v).view(N, self.du, self.dv, H*W)
        k = self.softmax(k.view(N, self.du, self.dk, H*W))

        lambda_c = torch.einsum('bukm,buvm->bkv', k, v)
        yc = torch.einsum('bhkm,bkv->bhvm', q, lambda_c)
        lambda_p = self.lambda_conv(v.view(N, self.du, self.dv, H, W)).view(N, self.dk, self.dv, H*W)
        yp = torch.einsum('bhkm,bkvm->bhvm', q, lambda_p)
        out = (yc + yp).reshape(N, C, H, W)

        if self.stride > 1:
            out = self.avgpool(out)

        return out

@BOTTLEBLOCK_REGISTRY.register()
class LambdaBottleneck(nn.Module):
    """
    - https://github.com/d-li14/lambda.pytorch
    - https://openreview.net/forum?id=xTJEN-ggl1b
    """
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.conv1 = CBA(in_c,hide_c,1)
        self.conv2 = nn.Sequential(LambdaLayer(hide_c,stride=stride),nn.BatchNorm2d(hide_c),nn.ReLU(inplace=True))
        self.conv3 = CBA(hide_c,out_c,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1)


# ----------------------------------------
"""
https://github.com/d-li14/dgconv.pytorch
https://arxiv.org/abs/1908.05867
"""
def aggregate(gate, D, I, K, sort=False):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))

@CONV_REGISTRY.register()
class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=False):
        super(DGConv2d, self).__init__()
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))
        self.K = int(math.log2(in_channels))
        eps = 1e-8
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        U_regularizer =  2 ** (self.K  + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org # Straight-Through Estimator
        U, gate = aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x, U_regularizer

@BOTTLEBLOCK_REGISTRY.register()
class DGBottleneck(nn.Module):
    """
    https://github.com/d-li14/dgconv.pytorch
    https://arxiv.org/abs/1908.05867
    """
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False):
        super().__init__()
        self.conv1 = CBA(in_c,hide_c,1)
        self.conv2 = DGConv2d(hide_c,hide_c,ksize,stride,padding=1,bias=False)
        self.BA = nn.Sequential(nn.BatchNorm2d(hide_c),nn.ReLU(inplace=True))
        self.conv3 = CBA(hide_c,out_c,1,activate=False)
        self.downsample = CBA(in_c, out_c, 1, stride, activate=False) if stride > 1 or in_c != out_c else \
            nn.Sequential()
        self.act = nn.ReLU()

        self.seblock = SELayer(out_c) if use_se else nn.Sequential()

    def forward(self, x):
        U_regularizer_sum = 0
        if isinstance(x, tuple):
            x, U_regularizer_sum = x[0], x[1]
        # identity = x

        x1 = self.conv1(x)
        x1,U_regularizer = self.conv2(x1)
        x1 = self.BA(x1)
        x1 = self.conv3(x1)
        x1 = self.seblock(x1)
        x = self.downsample(x)

        return self.act(x + x1),U_regularizer_sum + U_regularizer


def _make_layer(self, block, in_c,hide_c,out_c, blocks, stride=1):
    layers = []
    for i in range(blocks):
        if i == 0:
            layers.append(block(in_c,hide_c,out_c,3,stride,use_se=self.use_se))
        else:
            layers.append(block(out_c, hide_c, out_c, 3, 1,use_se=self.use_se))

    return nn.Sequential(*layers)

def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    """
    layers = [2,2,2,2] 18
    layers = [3,4,6,3] 34
    layers = [3,4,6,3] 50
    layers = [3,4,23,3] 101
    layers = [3,8,36,3] 152
    """
    def __init__(self, block,param={}, num_classes=1000,dropout=0.2,use_se=False):
        super().__init__()
        layers = param["layers"]
        in_c = param["in_c"]
        hide_c = param["hide_c"]
        out_c = param["out_c"]
        self.use_se = use_se

        self.stem = nn.Sequential(
            CBA(3,64,7,2),
            nn.MaxPool2d(3,2,1)
        )

        # self.stem = nn.Sequential(
        #     CBA(3, 64//2, 3, 2),
        #     involution(64//2,3,1),
        #     CBA(64//2,64,3,1),
        #     nn.MaxPool2d(3, 2, 1)
        # )


        self.layer1 = _make_layer(self,block, in_c[0],hide_c[0],out_c[0], layers[0])
        self.layer2 = _make_layer(self,block, in_c[1],hide_c[1],out_c[1], layers[1], 2)
        self.layer3 = _make_layer(self,block, in_c[2],hide_c[2],out_c[2], layers[2], 2)
        self.layer4 = _make_layer(self,block, in_c[3],hide_c[3],out_c[3], layers[3], 2)

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_c[-1], num_classes)
        )

        _init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if isinstance(x,tuple):x=x[0]
        x = self.classify(x)

        return x

def resnet34(block=None,num_classes=1000,dropout=0.2,use_se=False):
    """
    layers = [2,2,2,2] 18
    layers = [3,4,6,3] 34
    """
    param = {
        "layers": [3, 4, 6, 3],
        "in_c": [64, 64, 128, 256],
        "hide_c": [64, 128, 256, 512],
        "out_c": [64, 128, 256, 512],
    }
    return ResNet(BasicBlock,param,num_classes,dropout,use_se)

def resnet50(block,num_classes=1000,dropout=0.2,use_se=False):
    """
    26: (Bottleneck, (1, 2, 4, 1)),
    38: (Bottleneck, (2, 3, 5, 2)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))

    layers = [3,4,6,3] 50
    layers = [3,4,23,3] 101
    layers = [3,8,36,3] 152

    BottleBlock # resnet50
    BottleBlockX # resneXt50
    Bottle2Block # res2net50
    Bottle2BlockX # res2neXt50
    GhostBottleneck # GhostNet_50
    MobileV3Bottleneck # MobileNetv3_50
    RedNetBottleneck # RedNet50 可以不使用 se
    PSConvBottleneck # Ps-resnet50
    MixConvBottleneck # Mix-resnet50
    LambdaBottleneck # lambda-resnet50  可以不使用 se
    DGBottleneck # DGconv-resnet50
    """

    param = {
        "layers": [3, 4, 6, 3],
        "in_c": [64, 256, 512, 1024],
        "hide_c": [64, 128, 256, 512],
        "out_c": [256, 512, 1024, 2048],
    }
    return ResNet(block, param, num_classes, dropout, use_se)

if __name__ == "__main__":

    x = torch.rand([1,3,224,224])
    m = resnet50(DGBottleneck,use_se=False)
    print(m)
    pred = m(x)
    print(pred.shape)
    # torch.save(m.state_dict(),"DGBottleneck.pth")