# https://github.com/hustvl/YOLOP/

import torch
from torch import nn
import os
import requests

# from common import *
# from toolcv.api.define.utils.model.yolov5.common import *
from toolcv.tools.segment.net.YOLOP.common import *
from toolcv.tools.tools import load_weight


class YOLOP(nn.Module):
    """
    https://github.com/hustvl/YOLOP/tree/main/weights/End-to-end.pth
    """
    url = "https://codechina.csdn.net/wc781708249/model_pre_weights/-/raw/master/YOLOP/End-to-end.pth"

    def __init__(self, num_classes=1, anchors=[[3, 9, 5, 11, 4, 20],  # s=8
                                               [7, 18, 6, 39, 12, 31],  # s=16
                                               [19, 50, 38, 81, 68, 157]],  # s=32
                 ):
        """num_classes 不含背景"""
        super().__init__()

        self.model = nn.Sequential(
            # backbone
            Focus(3, 32, 3, 1, 1),
            Conv(32, 64, 3, 2, 1),
            BottleneckCSP(64, 64, 1),
            Conv(64, 128, 3, 2, 1),
            BottleneckCSP(128, 128, 3),  # x8 4

            Conv(128, 256, 3, 2, 1),
            BottleneckCSP(256, 256, 3),  # x16 6

            Conv(256, 512, 3, 2, 1),
            SPP(512, 512),
            BottleneckCSP(512, 512, 1, False),  # x32

            # FPN
            Conv(512, 256, 1, 1),  # 10
            nn.Upsample(None, 2, 'nearest'),
            Concat(),  # [-1,6]
            BottleneckCSP(512, 256, 1, False),
            Conv(256, 128, 1, 1),  # 14
            nn.Upsample(None, 2, 'nearest'),
            Concat(),  # [-1,4] # 做分割 16

            # PAN
            BottleneckCSP(256, 128, 1, False),
            Conv(128, 128, 3, 2),
            Concat(),  # [-1,14]
            BottleneckCSP(256, 256, 1, False),
            Conv(256, 256, 3, 2),
            Concat(),  # [-1,10]
            BottleneckCSP(512, 512, 1, False),

            # detect
            Detect(num_classes, anchors,
                   [128, 256, 512]),  # [17,20,23]  24

            # Driving area segmentation head
            Conv(256, 128, 3, 1),  # [16]  25
            nn.Upsample(None, 2, 'nearest'),
            BottleneckCSP(128, 64, 1, False),
            Conv(64, 32, 3, 1),
            nn.Upsample(None, 2, 'nearest'),
            Conv(32, 16, 3, 1),
            BottleneckCSP(16, 8, 1, False),
            nn.Upsample(None, 2, 'nearest'),
            Conv(8, num_classes + 1, 3, 1),  # 33 Driving area segmentation head

            # Lane line segmentation head
            Conv(256, 128, 3, 1),  # [16] 34
            nn.Upsample(None, 2, 'nearest'),
            BottleneckCSP(128, 64, 1, False),
            Conv(64, 32, 3, 1),
            nn.Upsample(None, 2, 'nearest'),
            Conv(32, 16, 3, 1),
            BottleneckCSP(16, 8, 1, False),
            nn.Upsample(None, 2, 'nearest'),
            Conv(8, num_classes + 1, 3, 1),  # 42 Lane line segmentation head
        )

    def forward(self, x):
        # backbone
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:10](x16)

        # FPN
        x32 = self.model[10](x32)  # 10
        x16 = self.model[12]((self.model[11](x32), x16))
        x16 = self.model[13:15](x16)
        x8 = self.model[16]((self.model[15](x16), x8))  # 16

        # PAN
        x8_ = self.model[17](x8)  # 17
        x8_2 = self.model[18](x8_)
        x16 = self.model[19]((x8_2, x16))
        x16_ = self.model[20](x16)  # 20
        x16_2 = self.model[21](x16_)
        x32 = self.model[22]((x16_2, x32))
        x32_ = self.model[23](x32)  # 23

        # detect
        detect_out = self.model[24]([x8_, x16_, x32_])

        # Driving area segmentation head
        seg_driving = self.model[25:34](x8)

        # Lane line segmentation head
        seg_lane = self.model[34:](x8)

        return detect_out, seg_driving, seg_lane


class YOLOPV2(nn.Module):
    """
    https://github.com/hustvl/YOLOP/tree/main/weights/End-to-end.pth
    """
    # url = "https://github.com/hustvl/YOLOP/tree/main/weights/End-to-end.pth"
    url = "https://codechina.csdn.net/wc781708249/model_pre_weights/-/raw/master/YOLOP/End-to-end.pth"
    def __init__(self, num_classes=1, anchors=[[3, 9, 5, 11, 4, 20],  # s=8
                                               [7, 18, 6, 39, 12, 31],  # s=16
                                               [19, 50, 38, 81, 68, 157]],  # s=32
                 stride=[8, 16, 32],
                 mode=["backbone", "fpn", "pan", "detect", "seg_driving", "seg_lane"]
                 ):
        """num_classes 不含背景"""
        super().__init__()

        layers = []

        if "backbone" in mode:
            layers.extend([
                Focus(3, 32, 3, 1, 1),
                Conv(32, 64, 3, 2, 1),
                BottleneckCSP(64, 64, 1),
                Conv(64, 128, 3, 2, 1),
                BottleneckCSP(128, 128, 3),  # x8 4

                Conv(128, 256, 3, 2, 1),
                BottleneckCSP(256, 256, 3),  # x16 6

                Conv(256, 512, 3, 2, 1),
                SPP(512, 512),
                BottleneckCSP(512, 512, 1, False),  # x32
            ])
        if "fpn" in mode:
            assert "backbone" in mode
            layers.extend([
                # FPN
                Conv(512, 256, 1, 1),  # 10
                nn.Upsample(None, 2, 'nearest'),
                Concat(),  # [-1,6]
                BottleneckCSP(512, 256, 1, False),
                Conv(256, 128, 1, 1),  # 14
                nn.Upsample(None, 2, 'nearest'),
                Concat(),  # [-1,4] # 做分割 16
            ])
        if "pan" in mode:
            assert "fpn" in mode
            layers.extend([
                # PAN
                BottleneckCSP(256, 128, 1, False),
                Conv(128, 128, 3, 2),
                Concat(),  # [-1,14]
                BottleneckCSP(256, 256, 1, False),
                Conv(256, 256, 3, 2),
                Concat(),  # [-1,10]
                BottleneckCSP(512, 512, 1, False),
            ])
        else:
            if "seg_driving" in mode or "seg_lane" in mode:
                layers.extend([nn.Identity() for _ in range(7)])

        if "detect" in mode:
            assert "pan" in mode
            layers.append(
                # detect
                Detect(num_classes, anchors,
                       [128, 256, 512]),  # [17,20,23]  24
            )
            layers[-1].stride = stride
        else:
            if "seg_driving" in mode or "seg_lane" in mode:
                layers.append(nn.Identity())

        if "seg_driving" in mode:
            assert "fpn" in mode
            layers.extend([
                # Driving area segmentation head
                Conv(256, 128, 3, 1),  # [16]  25
                nn.Upsample(None, 2, 'nearest'),
                BottleneckCSP(128, 64, 1, False),
                Conv(64, 32, 3, 1),
                nn.Upsample(None, 2, 'nearest'),
                Conv(32, 16, 3, 1),
                BottleneckCSP(16, 8, 1, False),
                nn.Upsample(None, 2, 'nearest'),
                Conv(8, num_classes + 1, 3, 1),  # 33 Driving area segmentation head
            ])
        else:
            if "seg_lane" in mode:
                layers.extend([nn.Identity() for _ in range(9)])

        if "seg_lane" in mode:
            assert "fpn" in mode
            layers.extend([
                # Lane line segmentation head
                Conv(256, 128, 3, 1),  # [16] 34
                nn.Upsample(None, 2, 'nearest'),
                BottleneckCSP(128, 64, 1, False),
                Conv(64, 32, 3, 1),
                nn.Upsample(None, 2, 'nearest'),
                Conv(32, 16, 3, 1),
                BottleneckCSP(16, 8, 1, False),
                nn.Upsample(None, 2, 'nearest'),
                Conv(8, num_classes + 1, 3, 1),  # 42 Lane line segmentation head
            ])

        self.model = nn.Sequential(*layers)
        self.mode = mode

    def forward_backbone(self, x):
        # backbone
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:10](x16)

        return x8, x16, x32

    def forward_fpn(self, x):
        x8, x16, x32 = self.forward_backbone(x)
        # FPN
        x32 = self.model[10](x32)  # 10
        x16 = self.model[12]((self.model[11](x32), x16))
        x16 = self.model[13:15](x16)
        x8 = self.model[16]((self.model[15](x16), x8))  # 16

        return x8, x16, x32

    def forward_pan(self, x):
        x8, x16, x32 = self.forward_fpn(x)
        # PAN
        x8_ = self.model[17](x8)  # 17
        x8_2 = self.model[18](x8_)
        x16 = self.model[19]((x8_2, x16))
        x16_ = self.model[20](x16)  # 20
        x16_2 = self.model[21](x16_)
        x32 = self.model[22]((x16_2, x32))
        x32_ = self.model[23](x32)  # 23

        return x8_, x16_, x32_

    def forward_detect(self, x):
        x8_, x16_, x32_ = self.forward_pan(x)
        # detect
        detect_out = self.model[24]([x8_, x16_, x32_])
        return detect_out

    def forward_seg_driving(self, x):
        x8, _, _ = self.forward_fpn(x)
        # Driving area segmentation head
        seg_driving = self.model[25:34](x8)
        return seg_driving

    def forward_seg_lane(self, x):
        x8, _, _ = self.forward_fpn(x)
        # Lane line segmentation head
        seg_lane = self.model[34:](x8)
        return seg_lane

    def forward(self, x):
        # backbone
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:10](x16)

        # FPN
        x32 = self.model[10](x32)  # 10
        x16 = self.model[12]((self.model[11](x32), x16))
        x16 = self.model[13:15](x16)
        x8 = self.model[16]((self.model[15](x16), x8))  # 16

        # PAN
        x8_ = self.model[17](x8)  # 17
        x8_2 = self.model[18](x8_)
        x16 = self.model[19]((x8_2, x16))
        x16_ = self.model[20](x16)  # 20
        x16_2 = self.model[21](x16_)
        x32 = self.model[22]((x16_2, x32))
        x32_ = self.model[23](x32)  # 23

        # detect
        detect_out = self.model[24]([x8_, x16_, x32_])

        # Driving area segmentation head
        seg_driving = self.model[25:34](x8)

        # Lane line segmentation head
        seg_lane = self.model[34:](x8)

        return detect_out, seg_driving, seg_lane


if __name__ == "__main__":
    url = "https://codechina.csdn.net/wc781708249/model_pre_weights/-/raw/master/YOLOP/End-to-end.pth"
    # data = requests.get(url).content
    # with open(url.split("/")[-1], 'wb') as fp:
    #     fp.write(data)

    model = YOLOPV2(mode=["backbone", "fpn", "pan", "detect"])
    model.eval()
    # load_weight(model, "./End-to-end.pth")
    load_weight(model, url=model.url)
    # x = torch.randn([1, 3, 256,256])
    # preds = model.forward_detect(x)
    # print(preds.shape)
    # torch.save(model.state_dict(), "weight.pth")
