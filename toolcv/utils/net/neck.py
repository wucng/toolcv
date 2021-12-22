"""
多尺度融合 spp ，aspp，PSPModule，DilatedParllelResidualBlockB 多尺度融合
多级语义融合  fpn pan
"""
import torch
from torch import nn
import torch.nn.functional as F

# spp or aspp 多尺度融合
from toolcv.tools.segment.net.espnet import DilatedParllelResidualBlockB as DRBlock
from toolcv.tools.segment.net.pspnet import PSPModule
from toolcv.tools.segment.net.YOLOP.common import SPP, BottleneckCSP, Conv
from toolcv.tools.segment.net.deeplabv3 import ASPP

from toolcv.utils.net.net import CBA


class FilterOutput(nn.Module):
    def __init__(self, index=[0, 1, 2]):
        super().__init__()
        self.index = index

    def forward(self, x):
        out = []
        for i in self.index:
            out.append(x[i])
        return out


class FPN(nn.Module):
    def __init__(self, in_c=[128, 256, 512], out_c=[256, 256, 256], activate=nn.ReLU(inplace=True), mode="add"):
        super().__init__()
        num_in = len(in_c)
        self.lateral = nn.ModuleList([CBA(in_c[i], out_c[i], 1, activate=activate) for i in range(num_in)])
        if mode == "add":
            self.fpn_convs = nn.ModuleList([CBA(out_c[i], out_c[i], 3, activate=activate) for i in range(num_in)])
        else:
            mid_c = []
            for i in out_c[::-1]:
                if len(mid_c) == 0:
                    mid_c.append(i)
                else:
                    mid_c.append(i + mid_c[-1])
            mid_c = mid_c[::-1]
            self.fpn_convs = nn.ModuleList([CBA(mid_c[i], out_c[i], 3, activate=activate) for i in range(num_in)])

        self.mode = mode
        self.num_in = num_in

        self.upsample = nn.Upsample(None, 2, 'nearest')

    def forward(self, x):
        out = []
        assert self.num_in == len(x)
        for i in range(self.num_in - 1, -1, -1):
            if len(out) == 0:
                m = self.lateral[i](x[i])
            else:
                if self.mode == "add":
                    m = self.upsample(out[-1]) + self.lateral[i](x[i])
                else:
                    m = torch.cat((self.upsample(out[-1]), self.lateral[i](x[i])), 1)

            out.append(m)

        out = out[::-1]
        for i in range(self.num_in):
            out[i] = self.fpn_convs[i](out[i])

        return out


class PAN(nn.Module):
    def __init__(self, in_c=[256, 256, 256], out_c=[256, 256, 256], activate=nn.ReLU(inplace=True), mode="add"):
        super().__init__()
        num_in = len(in_c)
        lateral = [nn.Identity()]
        for i in range(1, num_in):
            if mode == "add":
                lateral.append(CBA(in_c[i], out_c[i], 3, 2, activate=activate))
            else:
                mid_c = in_c[i - 1] + in_c[i] if i > 1 else in_c[i]
                lateral.append(CBA(mid_c, out_c[i], 3, 2, activate=activate))
        if mode == "add":
            fpn_convs = [nn.Identity()]
            for i in range(1, num_in): fpn_convs.append(CBA(out_c[i], out_c[i], 3, 1, activate=activate))
            self.fpn_convs = nn.ModuleList(fpn_convs)
        else:
            mid_c = []
            for i in out_c:
                if len(mid_c) == 0:
                    mid_c.append(i)
                else:
                    mid_c.append(i * 2)

            fpn_convs = [nn.Identity()]
            for i in range(1, num_in): fpn_convs.append(CBA(mid_c[i], out_c[i], 3, 1, activate=activate))
            self.fpn_convs = nn.ModuleList(fpn_convs)

        self.lateral = nn.ModuleList(lateral)
        self.num_in = num_in
        self.mode = mode

    def forward(self, x):
        out = []
        assert self.num_in == len(x)
        for i in range(self.num_in):
            if i == 0:
                out.append(x[i])
            else:
                if self.mode == "add":
                    out.append(self.lateral[i](out[-1]) + x[i])
                else:
                    out.append(torch.cat((self.lateral[i](out[-1]), x[i]), 1))

        for i in range(self.num_in):
            out[i] = self.fpn_convs[i](out[i])

        return out


class FPNv2(nn.Module):
    """
    在 RetinaNet 的 FPN 模块中只包括卷积，不包括 BN 和 ReLU。
    """

    def __init__(self, in_c=[], out_c=[], use_bn=False, use_act=False, do_p6=True, do_p7=True):
        super().__init__()
        self.lateral = nn.ModuleList(
            [CBA(in_c[i], out_c[i], 1, use_bn=use_bn, activate=use_act) for i in range(len(in_c))])

        self.fpn_convs = nn.ModuleList(
            [CBA(out_c[i], out_c[i], 3, use_bn=use_bn, activate=use_act) for i in range(len(in_c))])

        self.upsample = nn.Upsample(None, 2)

        self.do_p6 = do_p6
        self.do_p7 = do_p7
        if self.do_p6:
            self.convp6 = CBA(in_c[-1], out_c[-1], 3, 2, use_bn=use_bn, activate=use_act)
        if self.do_p7:
            self.convp7 = CBA(out_c[-1], out_c[-1], 3, 2, use_bn=use_bn, activate=use_act)

    def forward(self, x):
        if len(x) == 3:
            return self.forward_three(x)
        else:
            return self.forward_four(x)

    def forward_three(self, x):
        assert len(x) == 3
        c3, c4, c5 = x
        m3 = self.lateral[0](c3)
        m4 = self.lateral[1](c4)
        m5 = self.lateral[2](c5)

        m4 = self.upsample(m5) + m4
        m3 = self.upsample(m4) + m3

        p3 = self.fpn_convs[0](m3)
        p4 = self.fpn_convs[1](m4)
        p5 = self.fpn_convs[2](m5)

        if self.do_p6 and not self.do_p7:
            p6 = self.convp6(c5)
            return p3, p4, p5, p6

        if self.do_p6 and self.do_p7:
            p6 = self.convp6(c5)
            p7 = self.convp7(p6)

            return p3, p4, p5, p6, p7

        return p3, p4, p5

    def forward_four(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        m2 = self.lateral[0](c2)
        m3 = self.lateral[1](c3)
        m4 = self.lateral[2](c4)
        m5 = self.lateral[3](c5)

        m4 = self.upsample(m5) + m4
        m3 = self.upsample(m4) + m3
        m2 = self.upsample(m3) + m2

        p2 = self.fpn_convs[0](m2)
        p3 = self.fpn_convs[1](m3)
        p4 = self.fpn_convs[2](m4)
        p5 = self.fpn_convs[3](m5)

        if self.do_p6 and not self.do_p7:
            p6 = self.convp6(c5)
            return p2, p3, p4, p5, p6

        if self.do_p6 and self.do_p7:
            p6 = self.convp6(c5)
            p7 = self.convp7(p6)

            return p2, p3, p4, p5, p6, p7

        return p2, p3, p4, p5


class PANv2(nn.Module):
    def __init__(self, in_c=[], out_c=[], use_bn=False, use_act=False):
        super().__init__()
        self.nums = len(in_c)
        self.downsample = nn.ModuleList(
            [CBA(in_c[i], out_c[i], 3, 2, use_bn=use_bn, activate=use_act) for i in range(self.nums - 1)])

        self.pan_convs = nn.ModuleList(
            [CBA(out_c[i], out_c[i], 3, use_bn=use_bn, activate=use_act) for i in range(self.nums - 1)])

    # def forward(self, x):
    #     if len(x) == 3:
    #         return self.forward_three(x)
    #     else:
    #         return self.forward_four(x)
    #
    # def forward_three(self, x):
    #     p3, p4, p5 = x
    #     p4 = self.downsample[0](p3) + p4
    #     p5 = self.downsample[1](p4) + p5
    #
    #     p4 = self.pan_convs[0](p4)
    #     p5 = self.pan_convs[1](p5)
    #
    #     return p3, p4, p5
    #
    # def forward_four(self, x):
    #     p2, p3, p4, p5 = x
    #     p3 = self.downsample[0](p2) + p3
    #     p4 = self.downsample[1](p3) + p4
    #     p5 = self.downsample[2](p4) + p5
    #
    #     p3 = self.pan_convs[0](p3)
    #     p4 = self.pan_convs[1](p4)
    #     p5 = self.pan_convs[2](p5)
    #
    #     return p2, p3, p4, p5

    def forward(self, x):
        out = []
        nums = len(x)
        for i in range(nums):
            if i == 0 or i >= self.nums:
                out.append(x[i])
            else:
                out.append(self.downsample[i - 1](out[-1]) + x[i])

        for i in range(1, self.nums):
            out[i] = self.pan_convs[i - 1](out[i])

        return out


if __name__ == "__main__":
    from toolcv.utils.net.net import Backbone

    x = torch.rand([2, 3, 256, 256])
    m = Backbone(num_out=3)

    # m2 = FPN([64, 128, 256, 512], [256, 256, 256, 256], mode="cat")
    # p2, p3, p4, p5 = PAN([256, 256, 256, 256], [256, 256, 256, 256], mode="cat")(m2(m(x)))
    # print(p2.shape)

    m2 = FPNv2([128, 256, 512], [256, 256, 256])
    p3, p4, p5, p6, p7 = m2(m(x))
    m3 = PANv2([256, 256, 256], [256, 256, 256])
    print(p6.shape)
    print(p7.shape)
    p3, p4, p5, p6, p7 = m3((p3, p4, p5, p6, p7))
    print(p6.shape)
    print(p7.shape)
