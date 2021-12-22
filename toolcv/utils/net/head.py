import torch
from torch import nn
import torch.nn.functional as F

from toolcv.utils.net.net import CBA


class YOLOV3Head(nn.Module):
    def __init__(self, in_c=[128, 256, 512], num_anchor=3, num_classes=80, activate=None, use_bn=True,
                 dim_transform=True):
        super().__init__()
        self.dim_transform = dim_transform
        filters = num_anchor * (num_classes + 5)
        if activate is None: activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.convs_bridge = nn.ModuleList([CBA(in_c[2], in_c[2] * 2, 3, 1, use_bn=use_bn, activate=activate),
                                           CBA(in_c[1], in_c[1] * 2, 3, 1, use_bn=use_bn, activate=activate),
                                           CBA(in_c[0], in_c[0] * 2, 3, 1, use_bn=use_bn, activate=activate)
                                           ])
        self.convs_pred = nn.ModuleList([nn.Conv2d(in_c[2] * 2, filters, 1),
                                         nn.Conv2d(in_c[1] * 2, filters, 1),
                                         nn.Conv2d(in_c[0] * 2, filters, 1)])

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


class YOLOFHeadSelf(nn.Module):
    """动态选择 anchor"""

    def __init__(self, in_c, len_anchor, num_anchor=1, num_classes=80,
                 use_bn=True, activate=nn.ReLU(inplace=True), do_cat=False, do_wh_cls=True):
        super().__init__()
        self.do_wh_cls = do_wh_cls
        self.cls_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn, activate=activate) for _ in range(2)])
        self.bbox_subnet = nn.Sequential(*[CBA(in_c, in_c, 3, use_bn=use_bn, activate=activate) for _ in range(4)])
        self.cls_score = nn.Conv2d(in_c, num_anchor * num_classes, 3, 1, 1)
        self.bbox_offset = nn.Conv2d(in_c, num_anchor * 4, 3, 1, 1)  # 偏移值
        self.object_pred = nn.Conv2d(in_c, num_anchor * 1, 3, 1, 1)
        if self.do_wh_cls:
            self.wh_cls = nn.Conv2d(in_c, num_anchor * 2 * len_anchor, 3, 1, 1)  # 从备选的anchor中选择合适的anchor
        self.do_cat = do_cat

    def forward(self, x):
        cls_score = self.cls_score(self.cls_subnet(x))
        bbox_subnet = self.bbox_subnet(x)
        bbox_offset = self.bbox_offset(bbox_subnet)
        object_pred = self.object_pred(bbox_subnet)
        if self.do_wh_cls:
            wh_cls = self.wh_cls(bbox_subnet)
            if self.do_cat:
                return torch.cat((object_pred, cls_score, bbox_offset, wh_cls), 1)

            return object_pred, cls_score, bbox_offset, wh_cls
        else:
            if self.do_cat:
                return torch.cat((object_pred, cls_score, bbox_offset), 1)

            return object_pred, cls_score, bbox_offset


class YOLOFHeadSelfMS(nn.Module):
    def __init__(self, in_c=[256, 256, 256], len_anchor=5, num_anchor=1, num_classes=80, use_bn=True,
                 activate=nn.ReLU(inplace=True), do_cat=True, share=True, do_wh_cls=True):
        super().__init__()
        if isinstance(in_c, int): in_c = [in_c]
        # if share: use_bn = False
        if share or len(in_c) == 1:
            assert max(in_c) == min(in_c)
            self.m = YOLOFHeadSelf(in_c[0], len_anchor, num_anchor, num_classes, use_bn, activate, do_cat, do_wh_cls)
        else:
            self.m = nn.ModuleList(
                [YOLOFHeadSelf(in_c[i], len_anchor, num_anchor, num_classes, use_bn, activate, do_cat, do_wh_cls) for i
                 in range(len(in_c))])

        self.in_c = in_c
        self.share = share

    def forward(self, x):
        out = []
        if isinstance(x, (list, tuple)):
            assert len(x) == len(self.in_c)
            if self.share:
                for i in range(len(self.in_c)):
                    out.append(self.m(x[i]))
            else:
                for i in range(len(self.in_c)):
                    out.append(self.m[i](x[i]))
        else:
            assert len(self.in_c) == 1
            out.append(self.m(x))

        return out


class RetinaNetHead(nn.Module):
    def __init__(self, in_channels=256, feat_channels=256, num_anchors=9, num_class=81,
                 stacked_convs=4, use_bn=False, use_act=True):
        super().__init__()
        self.stacked_convs = stacked_convs

        cls_convs = []
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            cls_convs.append(CBA(chn, feat_channels, 3, use_bn=use_bn, activate=use_act))
            reg_convs.append(CBA(chn, feat_channels, 3, use_bn=use_bn, activate=use_act))

        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)

        # 构建最终输出层
        self.retina_cls = nn.Conv2d(feat_channels, num_anchors * num_class, 3, 1, 1)
        self.retina_reg = nn.Conv2d(feat_channels, num_anchors * 4, 3, 1, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            out = []
            for _x in x:
                out.append(self.forward_single(_x))

            return out
        else:
            return self.forward_single(x)

    def forward_single(self, x):
        # x是 p3-p7 中的某个特征图
        cls_feat = x
        reg_feat = x
        # 4层不共享参数卷积
        cls_feat = self.cls_convs(cls_feat)
        reg_feat = self.reg_convs(reg_feat)
        # 输出特征图
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred


class RPNHead(nn.Module):
    def __init__(self, in_channels, feat_channels, num_anchors=9, cls_out_channels=1):
        super().__init__()
        # 特征通道变换
        self.rpn_conv = CBA(in_channels, feat_channels, 3, use_bn=False)
        # 分类分支，类别固定是2，表示前后景分类
        # 并且由于 cls loss 是 bce，故实际上 self.cls_out_channels=1
        self.rpn_cls = nn.Conv2d(feat_channels, num_anchors * cls_out_channels, 1)
        # 回归分支，固定输出4个数值，表示基于 anchor 的变换值
        self.rpn_reg = nn.Conv2d(feat_channels, num_anchors * 4, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            out = []
            for _x in x:
                out.append(self.forward_single(_x))

            return out
        else:
            return self.forward_single(x)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        return self.rpn_cls(x), self.rpn_reg(x)


if __name__ == "__main__":
    c3 = torch.rand([2, 256, 52, 52])
    c4 = torch.rand([2, 256, 26, 26])
    c5 = torch.rand([2, 256, 13, 13])

    # YOLOV3Head()((c3, c4, c5))

    # m = YOLOFHeadSelfMS([256, 256, 256], share=False, do_wh_cls=False)
    # output = m((c3, c4, c5))
    # print(output[0].shape)
    # print(output[0].shape, output[1].shape, output[2].shape)

    m = RetinaNetHead(256, 256)
    output = m((c3, c4, c5))
    print(output[0][0].shape, output[0][1].shape)
