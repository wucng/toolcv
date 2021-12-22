import torch
from torch import nn
from toolcv.api.define.utils.model.net import CBA


class CenterNetHead(nn.Module):
    """num_classes不包含背景"""

    def __init__(self, in_c, num_classes=20):
        super().__init__()

        self.heatmap = nn.Sequential(
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

    def forward_single(self, x):
        return self.heatmap(x), self.wh(x), self.offset(x)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            heatmap_list, wh_list, offset_list = [], [], []
            for _x in x:
                heatmap, wh, offset = self.forward_single(_x)
                heatmap_list.append(heatmap)
                wh_list.append(wh)
                offset_list.append(offset)
            return heatmap_list, wh_list, offset_list
        else:
            return self.forward_single(x)


class CenterNetHeadV2(nn.Module):
    """num_classes不包含背景"""

    def __init__(self, in_c, num_classes=20):
        super().__init__()

        self.centerness = nn.Sequential(
            CBA(in_c, in_c, 3, 1, use_bn=False),
            nn.Conv2d(in_c, 1, 1)
        )

        self.cls = nn.Sequential(
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

    def forward_single(self, x):
        return self.centerness(x), self.cls(x), self.wh(x), self.offset(x)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            centerness_list, cls_list, wh_list, offset_list = [], [], [], []
            for _x in x:
                centerness, cls, wh, offset = self.forward_single(_x)
                centerness_list.append(centerness)
                cls_list.append(cls)
                wh_list.append(wh)
                offset_list.append(offset)
            return centerness_list, cls_list, wh_list, offset_list
        else:
            return self.forward_single(x)