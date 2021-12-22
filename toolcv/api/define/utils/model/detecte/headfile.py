import os
import torch
from torch import nn
# from torchvision.ops import DeformConv2d
from torchvision.models.resnet import Bottleneck, BasicBlock
import random
from collections import OrderedDict
import numpy as np
import json

from mmcv.ops import FusedBiasLeakyReLU, DeformConv2dPack, DeformRoIPool
# from mmcv.ops import DeformConv2d, DeformConv2dPack, FusedBiasLeakyReLU, Conv2d, ConvTranspose2d, \
#     SAConv2d, ModulatedDeformConv2d, ModulatedDeformConv2dPack, CrissCrossAttention, MaxPool2d, MaskedConv2d, \
#     MultiScaleDeformableAttention, SyncBatchNorm

from toolcv.api.define.utils.model.yolov5.experimental import CrossConv  # , MixConv2d
from toolcv.api.define.utils.model.yolov5.common import Focus, TransformerBlock, BottleneckCSP, \
    GhostBottleneck, C3, C3Ghost, C3TR, C3SPP, SPPF, SPP, Contract, Expand, GhostConv  # ,Bottleneck
from toolcv.api.define.utils.model.yolov5.activations import Mish, Hardswish, SiLU, FReLU, MemoryEfficientMish
from toolcv.api.define.utils.model.net import CBA
from toolcv.api.define.utils.model.advance.attention import Self_Attn, SEBlock
from toolcv.api.define.utils.model.advance.res2net.res2next import Bottle2neckX
from toolcv.api.define.utils.model.advance.res2net.res2net import Bottle2neck
from toolcv.api.define.utils.model.advance.cspdarknet.cslayers import Stage3, Stage
from toolcv.api.define.utils.model.advance.senet.se_resnet import SEBasicBlock, SEBottleneck
from toolcv.api.define.utils.model.advance.sknet.sknet import SKUnit
from toolcv.api.define.utils.model.advance.cbam.resnet_cbam import BasicBlock as CbamBasicBlock, \
    Bottleneck as CbamBottleneck
from toolcv.api.define.utils.model.advance.coordAttention.mbv2_ca import InvertedResidual
from toolcv.api.define.utils.model.advance.ecanet.eca_resnet import ECABasicBlock, ECABottleneck
from toolcv.api.define.utils.model.advance.epsanet.epsanet import EPSABlock
from toolcv.api.define.utils.model.advance.ghostnet.ghost_net import GhostBottleneck as GhostBottleneck2
from toolcv.api.define.utils.model.advance.mixnet.mixnet import MixNetBlock
from toolcv.api.define.utils.model.advance.mobilenet.mobilenetv3 import Block as MBv3Block, SeModule

# from toolcv.api.define.utils.model.advance.dropblock import DropBlock2D
# from toolcv.api.define.utils.loss.loss import LabelSmoothingCrossEntropy
# from apex.optimizers.fused_adam import FusedAdam
# from toolcv.api.mobileNeXt.codebase.loss.cross_entropy import LabelSmoothingCrossEntropy
# from toolcv.api.mobileNeXt.codebase.optim.radam import RAdam, PlainRAdam

from timm.optim.radam import RAdam, PlainRAdam
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from toolcv.api.mobileNeXt.codebase.models.gluon_resnet import Bottleneck as gluonBottleneck, \
    BasicBlock as gluonBasicBlock
from toolcv.api.mobileNeXt.codebase.models.hrnet import BasicBlock as hrBasicBlock, Bottleneck as hrBottleneck
from toolcv.api.mobileNeXt.codebase.models.i2rnet import I2RBlock
from toolcv.api.mobileNeXt.codebase.models.i2rnetv3_fbn import I2RBlock as I2RBlockv3_fbn
from toolcv.api.mobileNeXt.codebase.models.efficientnet_blocks import EdgeResidual

from timm.models.byobnet import BasicBlock as AttBasicBlock, BottleneckBlock as AttBottleneckBlock, \
    DarkBlock, EdgeBlock, RepVggBlock, SelfAttnBlock, LayerFn
from timm.models.cait import LayerScaleBlock, LayerScaleBlockClassAttn
from timm.models.coat import SerialBlock, ParallelBlock
from timm.models.convit import Block as convitBlock
from timm.models.cspnet import ResBottleneck, DarkBlock as cspDarkBlock, CrossStage, DarkStage
from timm.models.dla import DlaBasic, DlaBottleneck, DlaBottle2neck
from timm.models.dpn import DualPathBlock
from timm.models.efficientnet_blocks import InvertedResidual as timmInvertedResidual, CondConvResidual  # ,EdgeResidual
from timm.models.ghostnet import GhostBottleneck as GhostBottleneck3
from timm.models.gluon_xception import Block as gluonBlock
from timm.models.mlp_mixer import MixerBlock, SpatialGatingBlock, ResBlock
from timm.models.layers.create_attn import get_attn
from timm.models.layers.drop import DropPath, DropBlock2d

# from toolcv.api.define.utils.model.mmdet import load_weights, freeze_model, freeze_bn, unfreeze_model, \
#     statistical_parameter
from toolcv.api.define.utils.model.net import _initParmas


# from apex.normalization import FusedLayerNorm


# from timm.models.layers.activations import *
# from timm.models.layers.mixed_conv2d import MixedConv2d


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.downsample = nn.Sequential()
        self.do_downsample = c1 != c2
        if self.do_downsample:
            self.downsample = nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2))

    def forward(self, x):
        if self.do_downsample:
            return self.act(self.downsample(x) + self.bn(torch.cat([m(x) for m in self.m], 1)))
        else:
            return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Reduction(nn.Module):
    """代替maxpool2d模块"""

    def __init__(self, in_c, groups=16, act='relu', stride=2, e=1 / 2):
        super().__init__()
        c_ = int(in_c * e)
        self.stride = stride
        self.m1 = nn.MaxPool2d(3, self.stride, 1)
        self.m2 = CBA(in_c, c_, 3, self.stride, 'same', groups=groups, activate=False)
        self.m3 = CBA(in_c, c_, 3, self.stride, 'same', dilation=2, groups=groups, activate=False)
        """
        self.m3 = nn.Sequential(
        CBA(in_c, in_c // 4, 1, 1, 'same'),
        CBA(in_c // 4, in_c // 4, 3, 1, 'same'),
        CBA(in_c // 4, in_c, 3, 2, 'same')
        )
        """
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = FusedBiasLeakyReLU(in_c)

    def forward(self, x):
        return self.act(torch.cat((self.m2(x), self.m3(x)), 1) + self.m1(x))


def computeFlops(model, input=torch.randn(1, 3, 224, 224)):
    """
    FLOPS（即“每秒浮点运算次数”，“每秒峰值速度”）是“每秒所执行的浮点运算次数”
    （floating-point operations per second）的缩写。

    !pip install thop
    https://github.com/Lyken17/pytorch-OpCounter
    https://github.com/Swall0w/torchstat
    """
    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")

    print('flops: ', flops, 'params: ', params, 'memory(about):', str(float(params[:-1]) * 4) + params[-1])

    return flops, params


def summary(model, input_size=(3, 224, 224), device='cpu'):
    """
    !pip install torchsummary
    from torchsummary import summary
    summary(model.cuda(), input_size=(3, 512, 512))
    """
    from torchsummary import summary
    summary(model, input_size=input_size, device=device)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print('Size (MB):', size)
    os.remove('temp.p')

    return size


# -------------------------------RandomModel----------------------------
class Module:
    def __init__(self, device='cpu'):
        # self.cfg = {}
        self.device = device

    def get_activation(self, **kwargs):
        assert 'in_c' in kwargs
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice(['relu', 'prelu', 'LeakyReLU', 'FusedBiasLeakyReLU', "Mish",
                                    "Hardswish", "SiLU", "FReLU", "MemoryEfficientMish"])
            kwargs['typing'] = typing

        if typing.lower() == 'relu':
            act = nn.ReLU(inplace=True)
        elif typing.lower() == 'prelu':
            act = nn.PReLU()
        elif typing.lower() == 'LeakyReLU'.lower():
            act = nn.LeakyReLU(0.2, inplace=True)
        elif typing.lower() == 'FusedBiasLeakyReLU'.lower():
            in_c = kwargs['in_c']
            act = FusedBiasLeakyReLU(in_c)
        elif typing.lower() == "FReLU".lower():
            in_c = kwargs['in_c']
            act = FReLU(in_c)
        elif typing.lower() == "Mish".lower():
            act = Mish()
        elif typing.lower() == "Hardswish".lower():
            act = Hardswish()
        elif typing.lower() == "SiLU".lower():
            act = SiLU()
        elif typing.lower() == "MemoryEfficientMish".lower():
            act = MemoryEfficientMish()

        return act, kwargs

    def get_norm(self, **kwargs):
        assert 'in_c' in kwargs
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice(['BatchNorm2d', 'GroupNorm'])  # 'sbn'
            kwargs['typing'] = typing
        if typing.lower() == 'BatchNorm2d'.lower():
            norm = nn.BatchNorm2d(kwargs['in_c'])
        elif typing.lower() == 'SyncBatchNorm'.lower():
            norm = nn.SyncBatchNorm(kwargs['in_c'])
        elif typing.lower() == 'GroupNorm'.lower():
            in_c = kwargs['in_c']
            if 'groups' not in kwargs:
                kwargs['groups'] = random.choice(
                    [in_c // 2, in_c // 4, in_c // 8, in_c // 16, in_c // 32])
            norm = nn.GroupNorm(kwargs['groups'], in_c)
        elif typing.lower() == 'FusedLayerNorm'.lower():
            from apex.normalization import FusedLayerNorm
            """
            >>> input = torch.randn(20, 5, 10, 10)
            >>> # With Learnable Parameters
            >>> m = apex.normalization.FusedLayerNorm(input.size()[1:])
            """
            normalized_shape = kwargs['normalized_shape']
            norm = FusedLayerNorm(normalized_shape)

        _initParmas(norm.modules())

        return norm, kwargs

    def get_conv2d(self, **kwargs):
        assert 'in_c' in kwargs and 'out_c' in kwargs and 'stride' in kwargs
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            if self.device == 'cpu':
                typing = 'conv2d'
            else:
                typing = random.choice(['conv2d', 'DeformConv2dPack'])
            kwargs['typing'] = typing

        in_c = kwargs['in_c']
        out_c = kwargs['out_c']
        if 'ksize' not in kwargs:
            kwargs['ksize'] = random.choice([1, 3, 5])
        if 'bias' not in kwargs: kwargs['bias'] = random.choice([True, False])
        if 'dilation' not in kwargs:
            kwargs['dilation'] = random.choice([1, 2, 3])
        if 'groups' not in kwargs:
            # kwargs['groups'] = random.choice([in_c, in_c // 2, in_c // 4, in_c // 8, in_c // 16, in_c // 32])
            c_ = min(in_c, out_c)
            kwargs['groups'] = random.choice(
                [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1), max(c_ // 32, 1)])
        padding = (kwargs['ksize'] + (kwargs['ksize'] - 1) * (kwargs['dilation'] - 1)) // 2
        if typing.lower() == "conv2d":
            conv2d = nn.Conv2d(in_c, out_c, kwargs['ksize'], kwargs['stride'], padding,
                               kwargs['dilation'], kwargs['groups'], kwargs['bias'])
        elif typing.lower() == 'DeformConv2dPack'.lower():
            kwargs['bias'] = True
            conv2d = DeformConv2dPack(in_c, out_c, kwargs['ksize'], kwargs['stride'], padding,
                                      kwargs['dilation'], kwargs['groups'], kwargs['bias'])

        if 'mode' not in kwargs:
            kwargs['mode'] = random.choice(['normal', 'kaiming', 'xavier'])
        _initParmas(conv2d.modules(), mode=kwargs['mode'])

        return conv2d, kwargs

    def get_pooling(self, **kwargs):
        assert 'in_c' in kwargs
        in_c = kwargs['in_c']
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice(['MaxPool2d', 'AvgPool2d', 'Reduction'])
            kwargs['typing'] = typing

        if 'ksize' not in kwargs:
            kwargs['ksize'] = random.choice([2, 3, 4, 5])

        if 'stride' not in kwargs: kwargs['stride'] = 2

        padding = (kwargs['ksize'] - 1) // 2
        if typing.lower() == "MaxPool2d".lower():
            pool = nn.MaxPool2d(kwargs['ksize'], kwargs['stride'], padding)
        elif typing.lower() == "AvgPool2d".lower():
            pool = nn.AvgPool2d(kwargs['ksize'], kwargs['stride'], padding)
        # elif typing.lower() == 'DeformRoIPool'.lower():
        #     DeformRoIPool()
        elif typing.lower() == 'Reduction'.lower():
            if 'e' not in kwargs: kwargs['e'] = 1 / 2
            if 'groups' not in kwargs:
                c_ = int(in_c * kwargs['e'])
                kwargs['groups'] = random.choice(
                    [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1), max(c_ // 32, 1)])
            if 'act' not in kwargs:
                kwargs['act'] = random.choice(['relu', 'prelu'])

            pool = Reduction(in_c, kwargs['groups'], kwargs['act'], kwargs['stride'], kwargs['e'])

        return pool, kwargs

    def get_basicBlock(self, **kwargs):
        pass

    def get_Bottleneck(self, **kwargs):
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice([
                # attention
                'ECABottleneck', 'SEBottleneck', 'CbamBottleneck', 'GhostBottleneck2',
                'gluonBottleneck', 'hrBottleneck', 'MBv3Block', 'SKUnit', 'TransformerBlock', 'C3TR',
                'AttBottleneckBlock', 'DarkBlock', 'EdgeResidual',
                'ResBottleneck', 'timmInvertedResidual',
                'cspDarkBlock', 'EdgeBlock', 'RepVggBlock',
                # 'SelfAttnBlock',
                # 'CondConvResidual',

                'Bottle2neckX', 'Bottle2neck', 'Bottleneck', 'BottleneckCSP',
                'GhostBottleneck', 'I2RBlock', 'I2RBlockv3_fbn', 'MixNetBlock',
                'InvertedResidual', 'C3Ghost', 'DlaBottleneck', 'DlaBottle2neck', 'gluonBlock'
            ])
            kwargs['typing'] = typing

        layers = LayerFn()
        if 'attn' not in kwargs:
            kwargs['attn'] = random.choice(
                ['se', 'ese', 'eca', 'ecam', 'ceca', 'ge', 'gc', 'cbam', 'lcbam', 'sk', 'splat'])
        if 'self_attn' not in kwargs:
            kwargs['self_attn'] = random.choice(
                ['lambda', 'bottleneck', 'halo', 'involution', 'nl', 'bat'])  # 'swin',
        layers.attn = get_attn(kwargs['attn'])
        layers.self_attn = get_attn(kwargs['self_attn'])

        # if 'g' not in kwargs:
        #     kwargs['g'] = random.choice([4, 8, 16, 32])
        if 'e' not in kwargs:
            kwargs['e'] = random.choice([1 / 2, 1 / 4, 1 / 8])

        if 'use_se' not in kwargs:
            kwargs['use_se'] = random.choice([True, False])

        in_c = kwargs['in_c']

        c_ = int(in_c * kwargs['e'])
        if 'g' not in kwargs:
            kwargs['g'] = random.choice(
                [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1), max(c_ // 32, 1)])

        if typing.lower() == 'Bottle2neckX'.lower():
            m = Bottle2neckX(in_c, in_c // 4, 4, 8, 1, None, 4)
        elif typing.lower() == 'Bottle2neck'.lower():
            m = Bottle2neck(in_c, in_c // 4)
        elif typing.lower() == 'Bottleneck'.lower():
            m = Bottleneck(in_c, in_c // 4, 1, None, groups=32, base_width=4)
        elif typing.lower() == 'BottleneckCSP'.lower():
            m = BottleneckCSP(in_c, in_c, g=kwargs['g'], e=kwargs['e'])
        elif typing.lower() == 'ECABottleneck'.lower():
            m = ECABottleneck(in_c, in_c // 4)
        elif typing.lower() == 'SEBottleneck'.lower():
            m = SEBottleneck(in_c, in_c // 4)
        elif typing.lower() == 'GhostBottleneck'.lower():
            m = GhostBottleneck(in_c, in_c)
        elif typing.lower() == 'CbamBottleneck'.lower():
            m = CbamBottleneck(in_c, in_c // 4)
        elif typing.lower() == 'GhostBottleneck2'.lower():
            m = GhostBottleneck2(in_c, in_c // 4, in_c, 3, 1, kwargs['use_se'])
        elif typing.lower() == 'gluonBottleneck'.lower():
            m = gluonBottleneck(in_c, in_c // 4, cardinality=32, base_width=4, use_se=kwargs['use_se'])
        elif typing.lower() == 'hrBottleneck'.lower():
            m = hrBottleneck(in_c, in_c // 4, cardinality=32, base_width=4, use_se=kwargs['use_se'])
        elif typing.lower() == 'I2RBlock'.lower():
            m = I2RBlock(in_c, in_c, 1, 4, False)
        elif typing.lower() == 'I2RBlockv3_fbn'.lower():
            m = I2RBlockv3_fbn(in_c, in_c, 1, 4, False)
        elif typing.lower() == 'MBv3Block'.lower():
            semodule = None
            if kwargs['use_se']: semodule = layers.attn(in_c)  # SEBlock(in_c)
            m = MBv3Block(3, in_c, in_c // 4, in_c, nn.ReLU(inplace=True), semodule, 1)
        elif typing.lower() == 'MixNetBlock'.lower():
            m = MixNetBlock(in_c, in_c, [3, 5], [1, 1])
        elif typing.lower() == 'InvertedResidual'.lower():
            m = InvertedResidual(in_c, in_c, 1, 1 / 4)
        elif typing.lower() == 'SKUnit'.lower():
            m = SKUnit(in_c, in_c, 32, 2, kwargs['g'], 2, mid_features=c_)
        elif typing.lower() == 'TransformerBlock'.lower():
            m = TransformerBlock(in_c, in_c, 4, 1)
        elif typing.lower() == 'C3Ghost'.lower():
            m = C3Ghost(in_c, in_c, 1, g=kwargs['g'], e=kwargs['e'])
        elif typing.lower() == 'C3TR'.lower():
            m = C3TR(in_c, in_c, 1, g=kwargs['g'], e=kwargs['e'])
        elif typing.lower() == 'AttBottleneckBlock'.lower():
            m = AttBottleneckBlock(in_c, in_c, bottle_ratio=kwargs['e'], group_size=kwargs['g'], attn_last=True,
                                   layers=layers, drop_block=DropBlock2d, drop_path_rate=0.1)
        elif typing.lower() == 'DarkBlock'.lower():
            m = DarkBlock(in_c, in_c, bottle_ratio=kwargs['e'], group_size=kwargs['g'], attn_last=True, layers=layers,
                          drop_block=DropBlock2d, drop_path_rate=0.1)
        elif typing.lower() == 'EdgeBlock'.lower():
            m = EdgeBlock(in_c, in_c, bottle_ratio=kwargs['e'], group_size=kwargs['g'], attn_last=True, layers=layers,
                          drop_block=DropBlock2d, drop_path_rate=0.1)
        elif typing.lower() == 'RepVggBlock'.lower():
            m = RepVggBlock(in_c, in_c, bottle_ratio=kwargs['e'], group_size=kwargs['g'], layers=layers,
                            drop_block=DropBlock2d, drop_path_rate=0.1)
        elif typing.lower() == 'SelfAttnBlock'.lower():
            m = SelfAttnBlock(in_c, in_c, bottle_ratio=kwargs['e'], group_size=kwargs['g'], layers=layers,
                              drop_block=DropBlock2d, drop_path_rate=0.1)
        elif typing.lower() == 'ResBottleneck'.lower():
            m = ResBottleneck(in_c, in_c, bottle_ratio=kwargs['e'], groups=kwargs['g'], attn_last=True,
                              attn_layer=layers.attn, aa_layer=None, drop_block=DropBlock2d, drop_path=None)
        elif typing.lower() == 'cspDarkBlock'.lower():
            m = cspDarkBlock(in_c, in_c, bottle_ratio=kwargs['e'], groups=kwargs['g'], attn_layer=layers.attn,
                             drop_block=DropBlock2d, drop_path=None)
        elif typing.lower() == 'DlaBottleneck'.lower():
            m = DlaBottleneck(in_c, in_c, 1, cardinality=32, base_width=4)
        elif typing.lower() == 'DlaBottle2neck'.lower():
            m = DlaBottle2neck(in_c, in_c, 1, scale=1, cardinality=32, base_width=4)
        elif typing.lower() == 'timmInvertedResidual'.lower():
            m = timmInvertedResidual(in_c, in_c, exp_ratio=kwargs['e'], se_layer=layers.attn, drop_path_rate=0.1)
        elif typing.lower() == 'EdgeResidual'.lower():
            # 需注释掉 timm.models.efficientnet_blocks.py 208行
            # m = EdgeResidual(in_c, in_c, exp_ratio=kwargs['e'], se_layer=layers.attn, drop_path_rate=0.1)
            if 'se_ratio' not in kwargs: kwargs['se_ratio'] = random.choice([0, 1 / 4, 1 / 8])
            m = EdgeResidual(in_c, in_c, exp_ratio=kwargs['e'], se_ratio=kwargs['se_ratio'], drop_connect_rate=0.1)
        elif typing.lower() == 'CondConvResidual'.lower():
            m = CondConvResidual(in_c, in_c, exp_ratio=kwargs['e'], se_layer=layers.attn, drop_path_rate=0.1)
        elif typing.lower() == 'gluonBlock'.lower():
            m = gluonBlock(in_c, in_c, norm_layer=nn.BatchNorm2d)

        if 'mode' not in kwargs:
            kwargs['mode'] = random.choice(['normal', 'kaiming', 'xavier'])
        _initParmas(m.modules(), mode=kwargs['mode'])

        return m, kwargs

    def get_CBA(self, **kwargs):
        assert 'in_c' in kwargs and 'out_c' in kwargs and 'stride' in kwargs
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice(['CBA', 'GhostConv', 'CrossConv', 'MixConv2d'])
            kwargs['typing'] = typing

        in_c = kwargs['in_c']
        if 'ksize' not in kwargs:
            kwargs['ksize'] = random.choice([1, 3, 5])
        if 'bias' not in kwargs: kwargs['bias'] = random.choice([True, False])
        if 'dilation' not in kwargs:
            kwargs['dilation'] = random.choice([1, 2, 3])

        padding = (kwargs['ksize'] + (kwargs['ksize'] - 1) * (kwargs['dilation'] - 1)) // 2
        if typing.lower() == 'CBA'.lower():
            if 'groups' not in kwargs:
                c_ = min(in_c, kwargs['out_c'])
                kwargs['groups'] = random.choice(
                    [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1)])

            m = CBA(in_c, kwargs['out_c'], kwargs['ksize'], kwargs['stride'], padding, kwargs['dilation'],
                    kwargs['groups'])
        elif typing.lower() == 'GhostConv'.lower():
            if 'groups' not in kwargs:
                c_ = min(in_c, kwargs['out_c'] // 2)
                kwargs['groups'] = random.choice(
                    [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1)])

            m = GhostConv(in_c, kwargs['out_c'], kwargs['ksize'], kwargs['stride'], kwargs['groups'])
        elif typing.lower() == 'CrossConv'.lower():
            if 'e' not in kwargs:
                kwargs['e'] = random.choice([1 / 2, 1 / 4, 1 / 8])
            c_ = min(int(kwargs['out_c'] * kwargs['e']), kwargs['out_c'])
            if 'groups' not in kwargs:
                kwargs['groups'] = random.choice(
                    [c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1)])
            m = CrossConv(in_c, kwargs['out_c'], kwargs['ksize'], kwargs['stride'], kwargs['groups'], kwargs['e'], True)
        elif typing.lower() == 'MixConv2d'.lower():
            m = MixConv2d(in_c, kwargs['out_c'], s=kwargs['stride'])

        if 'mode' not in kwargs:
            kwargs['mode'] = random.choice(['normal', 'kaiming', 'xavier'])
        _initParmas(m.modules(), mode=kwargs['mode'])

        return m, kwargs

    def get_Focus(self, **kwargs):
        assert 'in_c' in kwargs and 'out_c' in kwargs and 'stride' in kwargs
        if 'typing' in kwargs:
            typing = kwargs['typing']
        else:
            typing = random.choice(['Focus'])
            kwargs['typing'] = typing

        in_c = kwargs['in_c']
        if 'ksize' not in kwargs:
            kwargs['ksize'] = random.choice([1, 3, 5])
        # if 'bias' not in kwargs: kwargs['bias'] = random.choice([True, False])
        if 'dilation' not in kwargs:
            kwargs['dilation'] = random.choice([1, 2, 3])
        if 'groups' not in kwargs:
            c_ = min(in_c * 4, kwargs['out_c'])
            kwargs['groups'] = random.choice([c_, max(c_ // 2, 1), max(c_ // 4, 1), max(c_ // 8, 1), max(c_ // 16, 1)])

        # padding = (kwargs['ksize'] + (kwargs['ksize'] - 1) * (kwargs['dilation'] - 1)) // 2

        if typing.lower() == 'Focus'.lower():
            if in_c in [1, 3]: kwargs['groups'] = 1
            m = Focus(in_c, kwargs['out_c'], kwargs['ksize'], kwargs['stride'], None, kwargs['groups'], True)

        if 'mode' not in kwargs:
            kwargs['mode'] = random.choice(['normal', 'kaiming', 'xavier'])
        _initParmas(m.modules(), mode=kwargs['mode'])

        return m, kwargs


Module.get_version = lambda self: print('0.0.1')


class RandomModel(Module):
    def __init__(self, device='cpu'):
        super().__init__(device)
        # self.get_version()

    def run(self, config):
        layers = []
        for i, cfg in enumerate(config):
            module_type = cfg['module_type']
            if module_type == "conv2d":
                m, cfg = self.get_conv2d(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type == "norm":
                m, cfg = self.get_norm(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type == "act":
                m, cfg = self.get_activation(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type == "pooling":
                m, cfg = self.get_pooling(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type == "bottleneck":
                m, cfg = self.get_Bottleneck(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type.lower() == "CBA".lower():  # conv2d+batchnorm+activation
                m, cfg = self.get_CBA(**cfg)
                layers.append(m)
                config[i] = cfg
            elif module_type.lower() == 'Focus'.lower():
                m, cfg = self.get_Focus(**cfg)
                layers.append(m)
                config[i] = cfg

        return nn.Sequential(*layers), config


def get_basic_config(in_c=3, basewidth=64, expand=2, layers=[3, 4, 6, 3]):
    config = {'model': [
        {"stage1": [dict(module_type='Focus', in_c=in_c, out_c=basewidth, stride=1)]
         },
        {"stage2": [
            dict(module_type='pooling', in_c=basewidth),
            dict(module_type='CBA', in_c=basewidth, out_c=basewidth, stride=1),
            *[dict(module_type='bottleneck', in_c=basewidth, out_c=basewidth, stride=1) for _ in range(layers[0])]
        ]},
        {"stage3": [
            dict(module_type='pooling', in_c=basewidth),
            dict(module_type='CBA', in_c=basewidth, out_c=basewidth * expand, stride=1),
            *[dict(module_type='bottleneck', in_c=basewidth * expand, out_c=basewidth * expand, stride=1)
              for _ in range(layers[1])]
        ]},
        {"stage4": [
            dict(module_type='pooling', in_c=basewidth * expand),
            dict(module_type='CBA', in_c=basewidth * expand, out_c=basewidth * expand * 2, stride=1),
            *[dict(module_type='bottleneck', in_c=basewidth * expand * 2, out_c=basewidth * expand * 2,
                   stride=1) for _ in range(layers[2])]
        ]},
        {"stage5": [
            dict(module_type='pooling', in_c=basewidth * expand * 2),
            dict(module_type='CBA', in_c=basewidth * expand * 2, out_c=basewidth * expand * 4, stride=1),
            *[dict(module_type='bottleneck', in_c=basewidth * expand * 4,
                   out_c=basewidth * expand * 4, stride=1) for _ in range(layers[3])]
        ]}
    ], 'describe': {}}

    return config


def get_backbone(config=None, device='cpu', test=True, save_config='./config.json'):
    if config is None:
        config = {'model': [
            {"stage1": [dict(module_type='Focus', in_c=3, out_c=64, stride=1)]
             },
            {"stage2": [
                dict(module_type='pooling', in_c=64),
                dict(module_type='CBA', in_c=64, out_c=64, stride=1),
                *[dict(module_type='bottleneck', in_c=64, out_c=64, stride=1) for _ in range(3)]
            ]},
            {"stage3": [
                dict(module_type='pooling', in_c=64),
                dict(module_type='CBA', in_c=64, out_c=128, stride=1),
                *[dict(module_type='bottleneck', in_c=128, out_c=128, stride=1) for _ in range(4)]
            ]},
            {"stage4": [
                dict(module_type='pooling', in_c=128),
                dict(module_type='CBA', in_c=128, out_c=256, stride=1),
                *[dict(module_type='bottleneck', in_c=256, out_c=256, stride=1) for _ in range(6)]
            ]},
            {"stage5": [
                dict(module_type='pooling', in_c=256),
                dict(module_type='CBA', in_c=256, out_c=512, stride=1),
                *[dict(module_type='bottleneck', in_c=512, out_c=512, stride=1) for _ in range(3)]
            ]}
        ], 'describe': {}}
    # layers = OrderedDict()
    # layers.update({name: m})
    # model = nn.Sequential(layers)
    layers = []
    _config = config['model']
    for i, cfg in enumerate(_config):
        name = 'stage' + str(i + 1)
        m, _cfg = RandomModel(device).run(cfg[name])
        _config[i][name] = _cfg
        layers.append(m)

    config['model'] = _config
    backbone = nn.Sequential(*layers)

    if test:
        try:
            x = torch.randn([1, 3, 224, 224]).to(device)
            backbone.to(device).eval()
            print(config)
            pred = backbone(x)
            print(pred.shape)
            # summary(backbone)
            size = print_size_of_model(backbone)
            flops, params = computeFlops(backbone, x)
            config['describe'].update(dict(size=str(size) + '(MB)', flops=flops, params=params))

            if len(save_config) > 0 and not os.path.exists(save_config):
                json.dump(config, open(save_config, 'w'), indent=4)
        except Exception as e:
            print(e)
            return None, None

    return backbone, config


def loop_get_backbone(in_c=3, basewidth=64, expand=2, layers=[3, 4, 6, 3], save_config='./config.json'):
    while True:
        basic_config = get_basic_config(in_c, basewidth, expand, layers)
        backbone, config = get_backbone(basic_config, save_config=save_config)
        if backbone is not None:
            return backbone, config


def modify_config(config):
    # 修改config
    config['model'][1]['stage2'][0]['typing'] = 'Reduction'
    config['model'][1]['stage2'][1]['typing'] = 'MixConv2d'

    config['model'][2]['stage3'][0]['typing'] = 'Reduction'
    config['model'][2]['stage3'][1]['typing'] = 'MixConv2d'

    config['model'][3]['stage4'][0]['typing'] = 'Reduction'
    config['model'][3]['stage4'][1]['typing'] = 'MixConv2d'

    config['model'][4]['stage5'][0]['typing'] = 'Reduction'
    config['model'][4]['stage5'][1]['typing'] = 'MixConv2d'

    return config


if __name__ == "__main__":
    """
    torch.backends.cudnn.benchmark = True
    backbone, config = loop_get_backbone(basewidth=64, expand=4, save_config='')
    # 修改config
    config = modify_config(config)
    backbone, config = get_backbone(config, save_config='./config.json')
    """
    config = json.load(open("config.json", 'r'))
    backbone, config = get_backbone(config, save_config='')
