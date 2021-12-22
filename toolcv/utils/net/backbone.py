import torch
from torch import nn
import torch.nn.functional as F

# from timm.models.cspnet import cspdarknet53
from timm.models import cspnet, resnet, resnest, res2net, tresnet, senet, nfnet, efficientnet, dla

_BatchNorm = (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d, nn.BatchNorm1d)


class BaseNet(nn.Module):
    def __init__(self, freeze_at=2, norm_eval=True):
        super().__init__()
        self.freeze_at = freeze_at
        self.norm_eval = norm_eval

    def train(self, mode: bool = True):
        # 这行代码会导致 BN 进入 train 模式
        super().train(mode)
        # 再次调用，固定 stem 和 前 n 个 stage 的 BN
        self._freeze_at()
        # 如果所有 BN 都采用全局均值和方差，则需要对整个网络的 BN 都开启 eval 模式
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    # 固定权重，需要两个步骤：1. 设置 eval 模式；2. requires_grad=False
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # 固定 stem 权重
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        # 固定 stage 权重
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _freeze_at(self):
        # for param in self.backbone[:self.freeze_at].parameters():
        #     param.requires_grad_(False)  # "Freezing Weight/Bias of BatchNorm2D、Conv2D."

        for m in self.backbone[:self.freeze_at].modules():
            # if isinstance(m, _BatchNorm):
            m.eval()  # "Freezing Mean/Var of BatchNorm2D."
            for param in m.parameters():
                param.requires_grad_(False)


class BackboneCspnet(BaseNet):
    def __init__(self, model_name='cspdarknet53', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        # m = cspdarknet53(pretrained)
        m = getattr(cspnet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.stem, m.stages[0]), m.stages[1], m.stages[2], m.stages[3], m.stages[4]
        )
        if stride == 16:
            self.backbone[-1].conv_down.conv.stride = (1, 1)
            if dilation:
                self.backbone[-1].conv_down.conv.dilation = (2, 2)
                self.backbone[-1].conv_down.conv.padding = (2, 2)
        elif stride == 8:
            self.backbone[-2].conv_down.conv.stride = (1, 1)
            self.backbone[-1].conv_down.conv.stride = (1, 1)
            if dilation:
                self.backbone[-2].conv_down.conv.dilation = (2, 2)
                self.backbone[-2].conv_down.conv.padding = (2, 2)
                self.backbone[-1].conv_down.conv.dilation = (4, 4)
                self.backbone[-1].conv_down.conv.padding = (4, 4)
        elif stride == 4:
            self.backbone[-3].conv_down.conv.stride = (1, 1)
            self.backbone[-2].conv_down.conv.stride = (1, 1)
            self.backbone[-1].conv_down.conv.stride = (1, 1)
            if dilation:
                self.backbone[-3].conv_down.conv.dilation = (2, 2)
                self.backbone[-3].conv_down.conv.padding = (2, 2)
                self.backbone[-2].conv_down.conv.dilation = (4, 4)
                self.backbone[-2].conv_down.conv.padding = (4, 4)
                self.backbone[-1].conv_down.conv.dilation = (8, 8)
                self.backbone[-1].conv_down.conv.padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneResnet(BaseNet):
    def __init__(self, model_name='resnet18', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(resnet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool),

            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )

        if stride == 16:
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-1][0].conv2.dilation = (2, 2)
                self.backbone[-1][0].conv2.padding = (2, 2)
        elif stride == 8:
            self.backbone[-2][0].conv1.stride = (1, 1)
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-2][0].conv2.dilation = (2, 2)
                self.backbone[-2][0].conv2.padding = (2, 2)
                self.backbone[-1][0].conv2.dilation = (4, 4)
                self.backbone[-1][0].conv2.padding = (4, 4)
        elif stride == 4:
            self.backbone[-3][0].conv1.stride = (1, 1)
            self.backbone[-3][0].conv2.stride = (1, 1)
            self.backbone[-3][0].downsample[0].stride = (1, 1)
            self.backbone[-2][0].conv1.stride = (1, 1)
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-3][0].conv2.dilation = (2, 2)
                self.backbone[-3][0].conv2.padding = (2, 2)
                self.backbone[-2][0].conv2.dilation = (4, 4)
                self.backbone[-2][0].conv2.padding = (4, 4)
                self.backbone[-1][0].conv2.dilation = (8, 8)
                self.backbone[-1][0].conv2.padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneResnest(BaseNet):
    def __init__(self, model_name='resnest14d', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(resnest, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool),

            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )

        if stride == 16:
            self.backbone[-1][0].avd_last.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-1][0].downsample[0].padding = (1, 1)
            if dilation:
                self.backbone[-1][0].conv2.conv.dilation = (2, 2)
                self.backbone[-1][0].conv2.conv.padding = (2, 2)
        elif stride == 8:
            self.backbone[-2][0].avd_last.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-2][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-2][0].downsample[0].padding = (1, 1)
            self.backbone[-1][0].avd_last.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-1][0].downsample[0].padding = (1, 1)
            if dilation:
                self.backbone[-2][0].conv2.conv.dilation = (2, 2)
                self.backbone[-2][0].conv2.conv.padding = (2, 2)
                self.backbone[-1][0].conv2.conv.dilation = (4, 4)
                self.backbone[-1][0].conv2.conv.padding = (4, 4)
        elif stride == 4:
            self.backbone[-3][0].avd_last.stride = (1, 1)
            self.backbone[-3][0].downsample[0].stride = (1, 1)
            self.backbone[-3][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-3][0].downsample[0].padding = (1, 1)
            self.backbone[-2][0].avd_last.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-2][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-2][0].downsample[0].padding = (1, 1)
            self.backbone[-1][0].avd_last.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].downsample[0].kernel_size = (3, 3)
            self.backbone[-1][0].downsample[0].padding = (1, 1)
            if dilation:
                self.backbone[-3][0].conv2.conv.dilation = (2, 2)
                self.backbone[-3][0].conv2.conv.padding = (2, 2)
                self.backbone[-2][0].conv2.conv.dilation = (4, 4)
                self.backbone[-2][0].conv2.conv.padding = (4, 4)
                self.backbone[-1][0].conv2.conv.dilation = (8, 8)
                self.backbone[-1][0].conv2.conv.padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneRes2net(BaseNet):
    def __init__(self, model_name='res2next50', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(res2net, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool),

            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )

        if stride == 16:
            for i in range(3):
                self.backbone[-1][0].convs[i].stride = (1, 1)
            self.backbone[-1][0].pool.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                for i in range(3):
                    self.backbone[-1][0].convs[i].dilation = (2, 2)
                    self.backbone[-1][0].convs[i].padding = (2, 2)
        elif stride == 8:
            for i in range(3):
                self.backbone[-2][0].convs[i].stride = (1, 1)
                self.backbone[-1][0].convs[i].stride = (1, 1)
            self.backbone[-2][0].pool.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].pool.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                for i in range(3):
                    self.backbone[-2][0].convs[i].dilation = (2, 2)
                    self.backbone[-2][0].convs[i].padding = (2, 2)
                    self.backbone[-1][0].convs[i].dilation = (4, 4)
                    self.backbone[-1][0].convs[i].padding = (4, 4)

        elif stride == 4:
            for i in range(3):
                self.backbone[-3][0].convs[i].stride = (1, 1)
                self.backbone[-2][0].convs[i].stride = (1, 1)
                self.backbone[-1][0].convs[i].stride = (1, 1)
            self.backbone[-3][0].pool.stride = (1, 1)
            self.backbone[-3][0].downsample[0].stride = (1, 1)
            self.backbone[-2][0].pool.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].pool.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                for i in range(3):
                    self.backbone[-3][0].convs[i].dilation = (2, 2)
                    self.backbone[-3][0].convs[i].padding = (2, 2)
                    self.backbone[-2][0].convs[i].dilation = (4, 4)
                    self.backbone[-2][0].convs[i].padding = (4, 4)
                    self.backbone[-1][0].convs[i].dilation = (8, 8)
                    self.backbone[-1][0].convs[i].padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneTresnet(BaseNet):
    def __init__(self, model_name='tresnet_m', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(tresnet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.body.SpaceToDepth, m.body.conv1),

            m.body.layer1,
            m.body.layer2,
            m.body.layer3,
            m.body.layer4,
        )

        if stride == 16:
            self.backbone[-1][0].conv2[1] = nn.Identity()
            self.backbone[-1][0].downsample[0] = nn.Identity()
            if dilation:
                self.backbone[-1][0].conv2[0][0].dilation = (2, 2)
                self.backbone[-1][0].conv2[0][0].padding = (2, 2)
        elif stride == 8:
            self.backbone[-2][0].conv2[1] = nn.Identity()
            self.backbone[-2][0].downsample[0] = nn.Identity()
            self.backbone[-1][0].conv2[1] = nn.Identity()
            self.backbone[-1][0].downsample[0] = nn.Identity()
            if dilation:
                self.backbone[-2][0].conv2[0][0].dilation = (2, 2)
                self.backbone[-2][0].conv2[0][0].padding = (2, 2)
                self.backbone[-1][0].conv2[0][0].dilation = (4, 4)
                self.backbone[-1][0].conv2[0][0].padding = (4, 4)
        elif stride == 4:
            self.backbone[-3][0].conv1[1] = nn.Identity()
            self.backbone[-3][0].downsample[0] = nn.Identity()
            self.backbone[-2][0].conv2[1] = nn.Identity()
            self.backbone[-2][0].downsample[0] = nn.Identity()
            self.backbone[-1][0].conv2[1] = nn.Identity()
            self.backbone[-1][0].downsample[0] = nn.Identity()
            if dilation:
                self.backbone[-3][0].conv2[0].dilation = (2, 2)
                self.backbone[-3][0].conv2[0].padding = (2, 2)
                self.backbone[-2][0].conv2[0][0].dilation = (4, 4)
                self.backbone[-2][0].conv2[0][0].padding = (4, 4)
                self.backbone[-1][0].conv2[0][0].dilation = (8, 8)
                self.backbone[-1][0].conv2[0][0].padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneSenet(BaseNet):
    def __init__(self, model_name='legacy_seresnet18', pretrained=False, freeze_at=5, stride=32, dilation=False,
                 num_out=1, norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(senet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.layer0, m.pool0),

            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )

        if stride == 16:
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-1][0].conv2.dilation = (2, 2)
                self.backbone[-1][0].conv2.padding = (2, 2)
        elif stride == 8:
            self.backbone[-2][0].conv1.stride = (1, 1)
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-2][0].conv2.dilation = (2, 2)
                self.backbone[-2][0].conv2.padding = (2, 2)
                self.backbone[-1][0].conv2.dilation = (4, 4)
                self.backbone[-1][0].conv2.padding = (4, 4)
        elif stride == 4:
            self.backbone[-3][0].conv1.stride = (1, 1)
            self.backbone[-3][0].conv2.stride = (1, 1)
            self.backbone[-3][0].downsample[0].stride = (1, 1)
            self.backbone[-2][0].conv1.stride = (1, 1)
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-2][0].downsample[0].stride = (1, 1)
            self.backbone[-1][0].conv1.stride = (1, 1)
            self.backbone[-1][0].conv2.stride = (1, 1)
            self.backbone[-1][0].downsample[0].stride = (1, 1)
            if dilation:
                self.backbone[-3][0].conv2.dilation = (2, 2)
                self.backbone[-3][0].conv2.padding = (2, 2)
                self.backbone[-2][0].conv2.dilation = (4, 4)
                self.backbone[-2][0].conv2.padding = (4, 4)
                self.backbone[-1][0].conv2.dilation = (8, 8)
                self.backbone[-1][0].conv2.padding = (8, 8)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneNfnet(BaseNet):
    def __init__(self, model_name='nf_seresnet26', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(nfnet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            m.stem,

            m.stages[0],
            m.stages[1],
            m.stages[2],
            nn.Sequential(m.stages[3], m.final_conv, m.final_act)
        )

        if stride == 16:
            self.backbone[-1][0][0].downsample.pool = nn.Identity()
            self.backbone[-1][0][0].conv2.stride = (1, 1)
            if dilation:
                self.backbone[-1][0][0].conv2.dilation = (2, 2)
                self.backbone[-1][0][0].conv2.padding = (2, 2)
        elif stride == 8:
            self.backbone[-2][0].downsample.pool = nn.Identity()
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-1][0][0].downsample.pool = nn.Identity()
            self.backbone[-1][0][0].conv2.stride = (1, 1)
            if dilation:
                self.backbone[-2][0].conv2.dilation = (2, 2)
                self.backbone[-2][0].conv2.padding = (2, 2)
                self.backbone[-1][0][0].conv2.dilation = (2, 2)
                self.backbone[-1][0][0].conv2.padding = (2, 2)

        elif stride == 4:
            self.backbone[-3][0].downsample.pool = nn.Identity()
            self.backbone[-3][0].conv2.stride = (1, 1)
            self.backbone[-2][0].downsample.pool = nn.Identity()
            self.backbone[-2][0].conv2.stride = (1, 1)
            self.backbone[-1][0][0].downsample.pool = nn.Identity()
            self.backbone[-1][0][0].conv2.stride = (1, 1)
            if dilation:
                self.backbone[-3][0].conv2.dilation = (2, 2)
                self.backbone[-3][0].conv2.padding = (2, 2)
                self.backbone[-2][0].conv2.dilation = (2, 2)
                self.backbone[-2][0].conv2.padding = (2, 2)
                self.backbone[-1][0][0].conv2.dilation = (2, 2)
                self.backbone[-1][0][0].conv2.padding = (2, 2)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneEfficientnet(BaseNet):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, freeze_at=5, stride=32, dilation=False,
                 num_out=1, norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(efficientnet, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.conv_stem, m.bn1, m.act1, m.blocks[0]),

            m.blocks[1],  # s=4
            m.blocks[2],  # s=8
            nn.Sequential(m.blocks[3], m.blocks[4]),  # s=16
            nn.Sequential(m.blocks[5], m.blocks[6]),  # s = 32
        )

        if stride == 16:
            self.backbone[-1][0][0].conv_dw.stride = (1, 1)
            if dilation:
                self.backbone[-1][0][0].conv_dw.dilation = (2, 2)
                self.backbone[-1][0][0].conv_dw.padding = (4, 4)
        elif stride == 8:
            self.backbone[-2][0][0].conv_dw.stride = (1, 1)
            self.backbone[-1][0][0].conv_dw.stride = (1, 1)
            if dilation:
                self.backbone[-2][0][0].conv_dw.dilation = (2, 2)
                self.backbone[-2][0][0].conv_dw.padding = (2, 2)
                self.backbone[-1][0][0].conv_dw.dilation = (4, 4)
                self.backbone[-1][0][0].conv_dw.padding = (8, 8)

        elif stride == 4:
            self.backbone[-3][0].conv_dw.stride = (1, 1)
            self.backbone[-2][0][0].conv_dw.stride = (1, 1)
            self.backbone[-1][0][0].conv_dw.stride = (1, 1)
            if dilation:
                self.backbone[-3][0].conv_dw.dilation = (2, 2)
                self.backbone[-3][0].conv_dw.padding = (4, 4)
                self.backbone[-2][0][0].conv_dw.dilation = (4, 4)
                self.backbone[-2][0][0].conv_dw.padding = (4, 4)
                self.backbone[-1][0][0].conv_dw.dilation = (8, 8)
                self.backbone[-1][0][0].conv_dw.padding = (16, 16)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


class BackboneDla(BaseNet):
    def __init__(self, model_name='dla34', pretrained=False, freeze_at=5, stride=32, dilation=False, num_out=1,
                 norm_eval=True):
        super().__init__(freeze_at, norm_eval)
        if num_out > 1: assert stride == 32
        self.num_out = num_out
        m = getattr(dla, model_name)(pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(m.base_layer, m.level0, m.level1),

            m.level2,  # s=4
            m.level3,  # s=8
            m.level4,  # s=16
            m.level5  # s =32
        )

        if stride == 16:
            self.backbone[-1].downsample = nn.Identity()
            self.backbone[-1].tree1.conv1.stride = (1, 1)

        elif stride == 8:
            self.backbone[-2].downsample = nn.Identity()
            self.backbone[-2].tree1.downsample = nn.Identity()
            self.backbone[-2].tree1.tree1.conv1.stride = (1, 1)
            self.backbone[-1].downsample = nn.Identity()
            self.backbone[-1].tree1.conv1.stride = (1, 1)

        elif stride == 4:
            self.backbone[-3].downsample = nn.Identity()
            self.backbone[-3].tree1.downsample = nn.Identity()
            self.backbone[-3].tree1.tree1.conv1.stride = (1, 1)
            self.backbone[-2].downsample = nn.Identity()
            self.backbone[-2].tree1.downsample = nn.Identity()
            self.backbone[-2].tree1.tree1.conv1.stride = (1, 1)
            self.backbone[-1].downsample = nn.Identity()
            self.backbone[-1].tree1.conv1.stride = (1, 1)

        self.out_channels = m.num_features

        # for param in self.backbone[:freeze_at].parameters():
        #     # for param in self.backbone.parameters():
        #     param.requires_grad_(False)
        #
        # if freeze_at > 0:
        #     # 默认冻结 BN中的参数 不更新
        #     for m in self.backbone.modules():
        #         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             for parameter in m.parameters():
        #                 parameter.requires_grad_(False)

    def forward(self, x):
        # 单分支
        # return self.backbone(x)  # c5
        out = self.forwardMS(x)

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


if __name__ == "__main__":
    x = torch.rand([2, 3, 416, 416])

    m = nn.Sequential(BackboneCspnet("darknet53", False, 2, stride=32, dilation=True, num_out=1))
    # m = BackboneEfficientnet("mixnet_m", False, stride=16, dilation=True, num_out=1)
    m.train()
    print(m(x).shape)
