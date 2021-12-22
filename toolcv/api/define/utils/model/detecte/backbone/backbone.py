import os
import torch
from torch import nn
import json

# from toolcv.api.define.utils.model.detecte.headfile import *
from toolcv.api.define.utils.model.detecte.headfile import Reduction, Focus, CBA, C3Ghost, \
    ResBottleneck, DropBlock2d, DropPath, get_attn, computeFlops, summary, print_size_of_model, _initParmas
from toolcv.api.define.utils.model.detecte.headfile import DlaBottle2neck, AttBottleneckBlock, ECABottleneck, \
    gluonBottleneck, hrBottleneck, MBv3Block, C3TR, cspDarkBlock, Bottle2neckX

from timm.models.byobnet import BasicBlock, BottleneckBlock, DarkBlock, EdgeBlock, RepVggBlock, SelfAttnBlock

import timm
from timm.models.byobnet import LayerFn

from toolcv.api.define.utils.model.mmdet import load_weights, freeze_model, freeze_bn, unfreeze_model, \
    statistical_parameter
from toolcv.api.define.utils.model.detecte.headfile import get_basic_config, get_backbone, loop_get_backbone


class Resnet50x(nn.Module):
    """
    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=2, n_layers=[3, 4, 6, 3], act='prelu')
    flops:  370.515M params:  2.720M memory(about): 10.88M
    Size (MB): 11.377343

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=2, n_layers=[2, 3, 5, 2], act='prelu')
    flops:  366.864M params:  2.704M memory(about): 10.816M
    Size (MB): 11.238847

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=32, exp=2, n_layers=[2, 2, 3, 2], act='prelu')
    flops:  114.959M params:  809.056K memory(about): 3236.224K
    Size (MB): 3.575007

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=4, n_layers=[2, 2, 3, 2], act='prelu')
    flops:  970.505M params:  9.516M memory(about): 38.064M
    Size (MB): 38.481951

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=4, n_layers=[3, 4, 6, 3], act='prelu')
    flops:  1.101G params:  8.606M memory(about): 34.424M
    Size (MB): 34.990367

    Resnet50x(3, 1000, g= 16, e=1 / 4, bw=128, exp=4, n_layers=[3, 4, 6, 3], act='prelu')
    flops:  4.197G params:  34.307M memory(about): 137.228M
    Size (MB): 137.907871

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=6, n_layers=[3, 4, 6, 3], act='prelu')
    flops:  2.083G params:  19.144M memory(about): 76.576M
    Size (MB): 77.195871

    Resnet50x(3, 1000, g=16, e=1 / 4, bw=64, exp=4, n_layers=[5, 6, 8, 5], act='prelu')
    flops:  1.117G params:  8.713M memory(about): 34.852M
    Size (MB): 35.573343
    """

    def __init__(self, in_c, num_classes, g=16, e=1 / 4, bw=64,
                 exp=2, n_layers=[3, 4, 6, 3], act='prelu',
                 freeze_at=5, stride=32, num_out=1, do_cls=False):
        super().__init__()
        self.num_out = num_out
        self.do_cls = do_cls
        n1, n2, n3, n4 = n_layers
        stage1 = nn.Sequential(Focus(in_c, bw, 3, 1, 1))

        stage2 = nn.Sequential(
            Reduction(bw, act=act), CBA(bw, bw),
            C3Ghost(bw, bw, n=n1, g=g // 2, e=e),
            ResBottleneck(bw, bw, bottle_ratio=e, groups=g // 2,
                          attn_last=True, attn_layer=get_attn('ecam'),
                          aa_layer=None, drop_block=DropBlock2d))

        stage3 = nn.Sequential(
            Reduction(bw, act=act, stride=1 if stride in [4] else 2), CBA(bw, bw * exp),
            C3Ghost(bw * exp, bw * exp, n=n2, g=g, e=e),
            ResBottleneck(bw * exp, bw * exp, bottle_ratio=e, groups=g,
                          attn_last=True, attn_layer=get_attn('ecam'),
                          aa_layer=None, drop_block=DropBlock2d))

        stage4 = nn.Sequential(
            Reduction(bw * exp, act=act, stride=1 if stride in [4, 8] else 2), CBA(bw * exp, bw * exp * 2),
            C3Ghost(bw * exp * 2, bw * exp * 2, n=n3, g=g, e=e),
            ResBottleneck(bw * exp * 2, bw * exp * 2, bottle_ratio=e, groups=g,
                          attn_last=True, attn_layer=get_attn('ecam'),
                          aa_layer=None, drop_block=DropBlock2d))

        stage5 = nn.Sequential(
            Reduction(bw * exp * 2, act=act, stride=1 if stride in [4, 8, 16] else 2), CBA(bw * exp * 2, bw * exp * 4),
            C3Ghost(bw * exp * 4, bw * exp * 4, n=n4, g=g, e=e),
            ResBottleneck(bw * exp * 4, bw * exp * 4, bottle_ratio=e, groups=g,
                          attn_last=True, attn_layer=get_attn('ecam'),
                          aa_layer=None, drop_block=DropBlock2d))

        self.backbone = nn.Sequential(stage1, stage2, stage3, stage4, stage5)
        self.out_channels = bw * exp * 4
        _initParmas(self.backbone.modules())

        if self.do_cls:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                      nn.Linear(self.out_channels, num_classes))
            _initParmas(self.head.modules())

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)

        if freeze_at > 0:
            # 默认冻结 BN中的参数 不更新
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    for parameter in m.parameters():
                        parameter.requires_grad_(False)

    def forward(self, x):
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


class Resnet50l(nn.Module):
    """
    Size (MB): 15.477679
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="DarkBlock")

    Size (MB): 14.737455
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="EdgeBlock")

    Size (MB): 12.152559
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="RepVggBlock")

    Size (MB): 14.756911
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="SelfAttnBlock")

    Size (MB): 21.048239
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="BasicBlock")

    Size (MB): 15.058607
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,first_layer="BottleneckBlock")

    Size (MB): 21.854831
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2, mid_layer='MBv3Block')

    Size (MB): 15.477679
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2, mid_layer='C3Ghost')

    Size (MB): 15.477679
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,last_layer="cspDarkBlock")

    Size (MB): 21.685327  (last_layer="C3TR" 不推荐 效果差)
    Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2,last_layer="C3TR")
    """

    def __init__(self, in_c, num_classes, g=16, e=1 / 4, bw=64,
                 exp=2, n_layers=[2, 3, 4, 6, 3], first_layer='DarkBlock', mid_layer='C3Ghost',
                 last_layer="cspDarkBlock",
                 attn_type='lcbam', freeze_at=5, stride=32, num_out=1, do_cls=False):
        super().__init__()

        if first_layer == 'DarkBlock':
            first_layer = DarkBlock
        elif first_layer == 'EdgeBlock':
            first_layer = EdgeBlock
        elif first_layer == 'RepVggBlock':
            first_layer = RepVggBlock
        elif first_layer == 'SelfAttnBlock':
            first_layer = SelfAttnBlock
        elif first_layer == 'BasicBlock':
            first_layer = BasicBlock
        elif first_layer == 'BottleneckBlock':
            first_layer = BottleneckBlock
        else:
            raise ('error!')

        layers = LayerFn()
        layers.attn = get_attn(attn_type)
        layers.self_attn = get_attn('involution')

        self.num_out = num_out
        self.do_cls = do_cls
        n0, n1, n2, n3, n4 = n_layers

        # ---------------------------stage1---------------------------------------
        stage1 = nn.Sequential(Focus(in_c, bw, 3, 1, 1), C3Ghost(bw, bw, n=n0, g=g // 2, e=e))

        # ---------------------------stage2---------------------------------------
        stage2 = nn.Sequential(
            first_layer(bw, bw * exp, 3, 2, bottle_ratio=e, group_size=bw // g, layers=layers,
                        drop_block=DropBlock2d, drop_path_rate=0.1),
            C3Ghost(bw * exp, bw * exp, n=n1, g=g, e=e) if mid_layer == 'C3Ghost' else \
                nn.Sequential(
                    *[MBv3Block(3, bw * exp, int(bw * exp * e), bw * exp, nn.ReLU(inplace=True), layers.attn(bw * exp),
                                1) for _ in range(n1)]),
            C3TR(bw * exp, bw * exp, n=1, g=g, e=e) if last_layer == 'C3TR' else \
                cspDarkBlock(bw * exp, bw * exp, 2, e, g, attn_layer=layers.attn, drop_block=DropBlock2d))

        # ---------------------------stage3---------------------------------------
        stride = 1 if stride in [4] else 2
        stage3 = nn.Sequential(
            first_layer(bw * exp, bw * exp * 2, 3, stride, bottle_ratio=e, group_size=bw * exp // g, layers=layers,
                        drop_block=DropBlock2d, drop_path_rate=0.1),
            C3Ghost(bw * exp * 2, bw * exp * 2, n=n2, g=g, e=e) if mid_layer == 'C3Ghost' else \
                nn.Sequential(
                    *[MBv3Block(3, bw * exp * 2, int(bw * exp * 2 * e), bw * exp * 2, nn.ReLU(inplace=True),
                                layers.attn(bw * exp * 2), 1) for _ in range(n2)]),
            C3TR(bw * exp * 2, bw * exp * 2, n=1, g=g, e=e) if last_layer == 'C3TR' else \
                cspDarkBlock(bw * exp * 2, bw * exp * 2, 2, e, g, attn_layer=layers.attn, drop_block=DropBlock2d))

        # ---------------------------stage4---------------------------------------
        stride = 1 if stride in [4, 8] else 2
        stage4 = nn.Sequential(
            first_layer(bw * exp * 2, bw * exp * 4, 3, stride, bottle_ratio=e, group_size=bw * exp * 2 // g,
                        layers=layers, drop_block=DropBlock2d, drop_path_rate=0.1),
            C3Ghost(bw * exp * 4, bw * exp * 4, n=n3, g=g, e=e) if mid_layer == 'C3Ghost' else \
                nn.Sequential(
                    *[MBv3Block(3, bw * exp * 4, int(bw * exp * 4 * e), bw * exp * 4, nn.ReLU(inplace=True),
                                layers.attn(bw * exp * 4), 1) for _ in range(n3)]),
            C3TR(bw * exp * 4, bw * exp * 4, n=1, g=g, e=e) if last_layer == 'C3TR' else \
                cspDarkBlock(bw * exp * 4, bw * exp * 4, 2, e, g, attn_layer=layers.attn, drop_block=DropBlock2d)
        )

        # ---------------------------stage5---------------------------------------
        stride = 1 if stride in [4, 8, 16] else 2
        stage5 = nn.Sequential(
            first_layer(bw * exp * 4, bw * exp * 8, 3, stride, bottle_ratio=e, group_size=bw * exp * 4 // g,
                        layers=layers, drop_block=DropBlock2d, drop_path_rate=0.1),
            C3Ghost(bw * exp * 8, bw * exp * 8, n=n4, g=g, e=e) if mid_layer == 'C3Ghost' else \
                nn.Sequential(
                    *[MBv3Block(3, bw * exp * 8, int(bw * exp * 8 * e), bw * exp * 8, nn.ReLU(inplace=True),
                                layers.attn(bw * exp * 8), 1) for _ in range(n4)]),
            C3TR(bw * exp * 8, bw * exp * 8, n=1, g=g, e=e) if last_layer == 'C3TR' else \
                cspDarkBlock(bw * exp * 8, bw * exp * 8, 2, e, g, attn_layer=layers.attn, drop_block=DropBlock2d)
        )

        self.backbone = nn.Sequential(stage1, stage2, stage3, stage4, stage5)
        self.out_channels = bw * exp * 8
        _initParmas(self.backbone.modules())

        if self.do_cls:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                      nn.Linear(self.out_channels, num_classes))
            _initParmas(self.head.modules())

        for parme in self.backbone[:freeze_at].parameters():
            # for parme in self.backbone.parameters():
            parme.requires_grad_(False)

        if freeze_at > 0:
            # 默认冻结 BN中的参数 不更新
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    for parameter in m.parameters():
                        parameter.requires_grad_(False)

    def forward(self, x):
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


class RandomModel(nn.Module):
    def __init__(self, in_c, num_classes, basewidth=64,
                 expand=2, n_layers=[3, 4, 6, 3], save_config='./config.json',
                 freeze_at=5, stride=32, num_out=1, do_cls=False):
        super().__init__()
        self.num_out = num_out
        self.do_cls = do_cls

        if os.path.exists(save_config):
            config = json.load(open(save_config, 'r'))

            if stride == 4:
                config['model'][2]['stage3'][0]['stride'] = 1
            if stride in [4, 8]:
                config['model'][3]['stage4'][0]['stride'] = 1
            if stride in [4, 8, 16]:
                config['model'][4]['stage5'][0]['stride'] = 1
        else:
            _, config = loop_get_backbone(in_c, basewidth, expand, n_layers, '')

        self.backbone, config = get_backbone(config, 'cpu', False)
        self.out_channels = config['model'][4]['stage5'][-1]['out_c']

        if not os.path.exists(save_config):
            json.dump(config, open(save_config, 'w'), indent=4)

        if self.do_cls:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                      nn.Linear(self.out_channels, num_classes))

            _initParmas(self.head.modules())

        for parme in self.backbone[:freeze_at].parameters():
            parme.requires_grad_(False)

        if freeze_at > 0:
            # 默认冻结 BN中的参数 不更新
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    for parameter in m.parameters():
                        parameter.requires_grad_(False)

    def forward(self, x):
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


def get_timm_model(model_name='repvgg_a2', pretrained=False, num_classes=1000, stride=32, test=True):
    model = timm.create_model(model_name, pretrained)

    # 冻结参数
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)

    # 对应分类
    model.head.fc.out_features = num_classes

    if stride == 16:
        model.stages[3][0].conv_kxk.conv.stride = (1, 1)
        model.stages[3][0].conv_1x1.conv.stride = (1, 1)
        model.head = nn.Sequential()
    model.out_channels = 1408

    # model.forward = lambda x:model.stages(model.stem(x))
    def forward(x):
        x = model.stem(x)
        c2 = model.stages[0](x)
        c3 = model.stages[1](c2)
        c4 = model.stages[2](c3)
        c5 = model.stages[3](c4)

        return c2, c3, c4, c5

    model.forward = forward

    if test:
        model.eval()
        x = torch.rand([1, 3, 224, 224])
        print(model(x)[-1].shape)

    return model


if __name__ == "__main__":
    x = torch.rand([1, 3, 224, 224])
    model = Resnet50l(3, 1000, g=16, e=1 / 4, bw=64, exp=2, last_layer="C3TR")
    """
    model = RandomModel(3, 1000, freeze_at=0, stride=32, do_cls=False)
    if os.path.exists('weight.pth'):
        model.load_state_dict(torch.load('weight.pth'))
        print("-------load weight successful---------")
    model.eval()
    pred = model(x)
    print(pred.shape)
    torch.save(model.state_dict(), 'weight.pth')
    """
    # summary(model)
    # computeFlops(model, x)
    print_size_of_model(model)
    # print(model)
    # get_timm_model(stride=16)
