import numpy as np

from toolcv.data.augment.bboxAug import ToTensor
from toolcv.data.augment.bboxAugv2 import mosaicFourImg
from toolcv.data.augment import bboxAug, bboxAugv2
from toolcv.data.transform import gaussian_radius, draw_umich_gaussian, gaussianValue
from toolcv.tools.utils import wh_iou_np, box_iou_np, x1y1x2y22xywh_np


def get_transform(train=True, resize=(224, 224), useImgaug=True, advanced=False):
    if train:
        if useImgaug:
            transforms = bboxAug.Compose([
                bboxAug.Augment(advanced=advanced),
                # bboxAug.Pad(), bboxAug.Resize(resize, False),
                bboxAug.Resize(resize, False),
                bboxAug.ToTensor(),  # PIL --> tensor
                bboxAug.Normalize()  # tensor --> tensor
            ])
        else:
            transforms = bboxAug.Compose([
                bboxAugv2.RandomHorizontalFlip(),
                # bboxAugv2.ResizeFixMinAndRandomCrop(int(resize[0]*1.1),resize), # 用于resize到固定大小
                # bboxAugv2.RandomDropAndResizeMaxMin(0.2, 600, 1000),  # 用于 fasterrecnn
                bboxAug.Resize(resize, False),
                bboxAugv2.RandomLight(),
                bboxAugv2.RandomColor(),
                bboxAugv2.RandomChanels(),
                bboxAugv2.RandomNoise(),
                bboxAugv2.RandomBlur(),
                bboxAugv2.RandomRotate(),
                bboxAugv2.RandomAffine(),

                bboxAugv2.RandomDropPixelV2(),
                bboxAugv2.RandomCutMixV2(),
                bboxAugv2.RandomMosaic(),
                bboxAug.ToTensor(),  # PIL --> tensor
                bboxAug.Normalize()  # tensor --> tensor
            ])
            """
            transforms = bboxAug.Compose([
                # bboxAug.RandomChoice(),
                bboxAug.RandomHorizontalFlip(),
                bboxAug.RandomBrightness(),
                bboxAug.RandomBlur(),
                bboxAug.RandomSaturation(),
                bboxAug.RandomHue(),
                # bboxAug.RandomRotate(angle=5),
                # bboxAug.RandomTranslate(),
                bboxAug.Pad(), bboxAug.Resize(resize, False),
                bboxAug.ToTensor(),  # PIL --> tensor
                bboxAug.Normalize()  # tensor --> tensor
            ])
            """
    else:
        transforms = bboxAug.Compose([
            # bboxAug.Pad(), bboxAug.Resize(resize, False),
            bboxAug.Resize(resize, False),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])

    return transforms


def get_transforms(mode, advanced=False, base_size=448, target_size=416):
    if mode == 0:  # 推荐
        return bboxAug.Compose([
            bboxAug.Augment(advanced),
            bboxAugv2.ResizeFixMinAndRandomCrop(base_size, (target_size, target_size)),  # 用于resize到固定大小
            # bboxAugv2.RandomDropAndResizeMaxMin(0.2,600,1000), # 用于 fasterrecnn

            bboxAugv2.RandomRotate(),
            bboxAugv2.RandomAffine(),

            bboxAugv2.RandomDropPixelV2(),
            # # bboxAugv2.RandomCutMixV2(),
            bboxAugv2.RandomMosaic(),
            # random.choice([bboxAugv2.RandomCutMixV2(),bboxAugv2.RandomMosaic()]),

            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])
    elif mode == 1:  # 推荐
        return bboxAug.Compose([
            bboxAugv2.RandomHorizontalFlip(),
            bboxAugv2.ResizeFixMinAndRandomCrop(base_size, (target_size, target_size)),  # 用于resize到固定大小
            # bboxAugv2.RandomDropAndResizeMaxMin(0.2,600,1000), # 用于 fasterrecnn
            bboxAugv2.RandomLight(),
            bboxAugv2.RandomColor(),
            bboxAugv2.RandomChanels(),
            bboxAugv2.RandomNoise(),
            bboxAugv2.RandomBlur(),
            bboxAugv2.RandomRotate(),
            bboxAugv2.RandomAffine(),

            bboxAugv2.RandomDropPixelV2(),
            # bboxAugv2.RandomCutMixV2(),
            bboxAugv2.RandomMosaic(),
            # random.choice([bboxAugv2.RandomCutMixV2(),bboxAugv2.RandomMosaic()]),

            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])
    elif mode == 2:  # 不推荐
        return bboxAug.Compose([
            # bboxAug.RandomChoice(),
            bboxAug.RandomHorizontalFlip(),
            bboxAug.RandomBrightness(),
            bboxAug.RandomBlur(),
            bboxAug.RandomSaturation(),
            bboxAug.RandomHue(),
            bboxAug.RandomRotate(angle=5),
            # bboxAug.RandomTranslate(), # 有问题
            # bboxAug.Augment(False),
            bboxAug.Pad(),
            bboxAug.Resize((target_size, target_size), False),
            # bboxAug.ResizeMinMax(600,1000),
            # bboxAug.ResizeFixAndPad(),
            # bboxAug.RandomHSV(),
            bboxAug.RandomCutout(),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])

class Transforms(object):
    def __init__(self,transforms=None,advanced=False,target_size=416):
        self.transforms = transforms
        self.advanced = advanced
        self.target_size = target_size
        self.base_size = int(target_size*1.2)

    def __call__(self, *args, **kwargs):
        return self.forward()

    def forward(self):
        if self.transforms is not None:
            return self.transforms(self.advanced,self.base_size,self.target_size)
        else:
            return get_transforms(0,self.advanced,self.base_size,self.target_size)

def getitem_fcos(self, target):
    """:param mode: 'sigmoid','exp' (推荐 exp)"""
    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()

    # 按面积从大到小排序
    # boxes = np.stack(sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True), 0)

    fh, fw = int(np.ceil(self.resize / self.strides)), int(np.ceil(self.resize / self.strides))
    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes], np.float32)  # conf,x,y,w,h,class_id....

    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # w = x2 - x1
        # h = y2 - y1

        tfx = cx * fw
        tfy = cy * fh
        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        y1_ = y1 * fh
        y2_ = y2 * fh
        x1_ = x1 * fw
        x2_ = x2 * fw
        l, r = fx - x1_, x2_ - fx
        t, b = fy - y1_, y2_ - fy
        if l > 0 and r > 0 and t > 0 and b > 0:
            cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
        else:
            cenerness = 1
            l, t, r, b = max(0, l), max(0, t), max(0, r), max(0, b)
        featureMap[fy, fx, 0, 0] = cenerness
        # if cenerness > 0.3:  # 0
        if self.mode == "sigmoid":
            featureMap[fy, fx, :, 1:5] = [l / fw, t / fh, r / fw, b / fh]  # 0~1 sigmoid()
        elif self.mode == "exp":
            featureMap[fy, fx, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
        else:
            raise ('error!!')

        featureMap[fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[label]

        y_min, y_max = int(np.ceil(y1_)), int(np.floor(y2_))
        x_min, x_max = int(np.ceil(x1_)), int(np.floor(x2_))
        if y_max > y_min and x_max > x_min:
            for y in range(y_min, y_max):
                t, b = y - y1_, y2_ - y
                for x in range(x_min, x_max):
                    l, r = x - x1_, x2_ - x
                    cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                    # featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                    if cenerness <= featureMap[y, x, 0, 0]: continue
                    featureMap[y, x, 0, 0] = cenerness
                    if self.mode == "sigmoid":
                        featureMap[y, x, :, 1:5] = [l / fw, t / fh, r / fw, b / fh]  # 0~1 sigmoid()
                    elif self.mode == "exp":
                        featureMap[y, x, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
                    else:
                        raise ('error!!')

                    featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[label]

    return featureMap


def getitem_yolov1(self, target):
    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()

    # 按面积从大到小排序 ????
    # boxes = np.stack(sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True), 0)

    fh, fw = int(np.ceil(self.resize / self.strides)), int(np.ceil(self.resize / self.strides))
    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes], np.float32)  # conf,x,y,w,h,class_id....

    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        _w = w
        _h = h
        if self.box_norm == "log":
            w = np.log(_w)
            h = np.log(_h)
        elif self.box_norm == "logv2":
            w = np.log(_w / (self.strides / self.resize))
            h = np.log(_h / (self.strides / self.resize))
        elif self.box_norm == "sqrt":
            w = np.sqrt(_w)
            h = np.sqrt(_h)

        tfx = cx * fw
        tfy = cy * fh
        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        dx = tfx - fx
        dy = tfy - fy
        # featureMap[fy, fx, :, :5] = [1, cx, cy, w, h] # 效果较差
        # yolov1
        featureMap[fy, fx, :, :5] = [1, dx, dy, w, h]
        featureMap[fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[label]

        if self.mode == "centernet":
            # CenterNet
            radius = gaussian_radius((_h * fw, _w * fw))
            radius = max(1, int(radius))
            featureMap[..., 0, 0] = draw_umich_gaussian(featureMap[..., 0, 0], (fx, fy), radius)

        elif self.mode == "centernetv2":
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        # featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        if cenerness <= featureMap[y, x, 0, 0]: continue
                        featureMap[y, x, 0, 0] = cenerness
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y
                        featureMap[y, x, :, 1:5] = [dx, dy, w, h]
                        # featureMap[y, x, :, 1:5] = [cx,cy, w, h] # 效果较差
                        featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        elif self.mode == "fcos":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness ** 2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)

        elif self.mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        if cenerness <= featureMap[y, x, 0, 0]: continue
                        featureMap[y, x, 0, 0] = cenerness
                        dx = tfx - x
                        dy = tfy - y
                        featureMap[y, x, :, 1:5] = [dx, dy, w, h]
                        # featureMap[y, x, :, 1:5] = [cx,cy, w, h] # 效果较差
                        featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

    return featureMap


# 使用先验anchor
def getitem_yolov2(self, target):
    """:param anchor: 5x2 [w,h] 缩放到0~1"""
    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()

    # 按面积从大到小排序
    # boxes = np.stack(sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True), 0)
    fh, fw = int(np.ceil(self.resize / self.strides)), int(np.ceil(self.resize / self.strides))
    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes], np.float32)  # conf,x,y,w,h,class_id....

    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        tfx = cx * fw
        tfy = cy * fh
        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        dx = tfx - fx
        dy = tfy - fy

        # 计算IOU
        iou = wh_iou_np(self.anchors, np.array([[w, h]]))  # [n,1]
        max_idx = iou.argmax(0)[0]
        a_w, a_h = self.anchors[max_idx]
        for i, v in enumerate(iou):
            if v[0] > 0.5 and i != max_idx:
                # 忽略掉
                featureMap[fy, fx, i, 0] = -1

        # yolov2
        # featureMap[fy, fx, max_idx, :5] = [1, dx, dy, w, h]
        featureMap[fy, fx, max_idx, :5] = [1, dx, dy, np.log(w / a_w), np.log(h / a_h)]
        featureMap[fy, fx, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

        if self.mode == "centernet":
            # CenterNet
            radius = gaussian_radius((h * fw, w * fw))
            radius = max(1, int(radius))
            featureMap[..., max_idx, 0] = draw_umich_gaussian(featureMap[..., max_idx, 0], (fx, fy), radius)

        elif self.mode == "centernetv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        # featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                        if cenerness <= featureMap[y, x, max_idx, 0]: continue
                        featureMap[y, x, max_idx, 0] = cenerness
                        dx = tfx - x
                        dy = tfy - y
                        # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                        featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                        featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

        elif self.mode == "fcos":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        cenerness = cenerness ** 2
                        featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)

        elif self.mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        # featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                        if cenerness <= featureMap[y, x, max_idx, 0]: continue
                        featureMap[y, x, max_idx, 0] = cenerness
                        dx = tfx - x
                        dy = tfy - y
                        # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                        featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                        featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

    return featureMap


def getitem_yolov3(self, target):
    """:param anchors: [[],[],[]] w,h 缩放到0~1"""
    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()

    # 按面积从大到小排序
    # boxes = np.stack(sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True), 0)

    featureMap_list = []
    for ii, stride in enumerate(self.strides):
        anchor = self.anchors[ii]
        fh, fw = int(np.ceil(self.resize / stride)), int(np.ceil(self.resize / stride))
        featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes], np.float32)  # conf,x,y,w,h,class_id....

        for label, (x1, y1, x2, y2) in zip(labels, boxes):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            tfx = cx * fw
            tfy = cy * fh
            fx = int(tfx)  # 缩放到 featureMap上
            fy = int(tfy)

            dx = tfx - fx
            dy = tfy - fy

            # 计算IOU
            iou = wh_iou_np(anchor, np.array([[w, h]]))  # [n,1]
            max_idx = iou.argmax(0)[0]
            a_w, a_h = anchor[max_idx]
            for i, v in enumerate(iou):
                if v[0] > 0.5 and i != max_idx:
                    # 忽略掉
                    featureMap[fy, fx, i, 0] = -1

            # yolov2
            # featureMap[fy, fx, max_idx, :5] = [1, dx, dy, w, h]
            featureMap[fy, fx, max_idx, :5] = [1, dx, dy, np.log(w / a_w), np.log(h / a_h)]
            featureMap[fy, fx, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

            if self.mode == "centernet":
                # CenterNet
                radius = gaussian_radius((h * fw, w * fw))
                radius = max(1, int(radius))
                featureMap[..., max_idx, 0] = draw_umich_gaussian(featureMap[..., max_idx, 0], (fx, fy), radius)

            elif self.mode == "centernetv2":
                # Fcos
                y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
                x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
                if y_max > y_min and x_max > x_min:
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            cenerness = gaussianValue(x, y, cx, cy)
                            # featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                            if cenerness <= featureMap[y, x, max_idx, 0]: continue
                            featureMap[y, x, max_idx, 0] = cenerness
                            dx = tfx - x
                            dy = tfy - y
                            # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                            featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                            featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

            elif self.mode == "fcos":
                # Fcos
                y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
                x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
                if y_max > y_min and x_max > x_min:
                    for y in range(y_min, y_max):
                        t, b = y - y_min, y_max - y
                        for x in range(x_min, x_max):
                            l, r = x - x_min, x_max - x
                            cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                            # cenerness = cenerness ** 2
                            featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)

            elif self.mode == "fcosv2":
                # Fcos
                y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
                x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
                if y_max > y_min and x_max > x_min:
                    for y in range(y_min, y_max):
                        t, b = y - y_min, y_max - y
                        for x in range(x_min, x_max):
                            l, r = x - x_min, x_max - x
                            cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                            # featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                            if cenerness <= featureMap[y, x, max_idx, 0]: continue
                            featureMap[y, x, max_idx, 0] = cenerness
                            dx = tfx - x
                            dy = tfy - y
                            # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                            featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                            featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[label]

        featureMap_list.append(featureMap.reshape(-1, 5 + self.num_classes))

    return np.concatenate(featureMap_list, 0)


def getitem_ssd(self, target):
    """:param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1"""
    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    gt_boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()

    fh, fw = int(np.ceil(self.resize / self.strides)), int(np.ceil(self.resize / self.strides))
    featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

    anchor_xywh = x1y1x2y22xywh_np(self.anchors)
    gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

    iou = box_iou_np(self.anchors, gt_boxes)

    iou_v = iou.max(1)
    iou_i = iou.argmax(1)
    for i, v in enumerate(iou_v):
        if v >= 0.5:  # 正
            x, y, w, h = gt_boxes_xywh[iou_i[i]]
            a_x, a_y, a_w, a_h = anchor_xywh[i]
            featureMap[i, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
            featureMap[i, -1] = int(labels[iou_i[i]])
        elif v < 0.5:  # 负
            featureMap[i, -1] = 0

    # 与gt_boxes最大iou的anchor为正
    # iou_v = iou.max(0)
    iou_i = iou.argmax(0)
    for i, idx in enumerate(iou_i):
        x, y, w, h = gt_boxes_xywh[i]
        a_x, a_y, a_w, a_h = anchor_xywh[idx]
        featureMap[idx, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
        featureMap[idx, -1] = int(labels[i])

    return featureMap


def getitem_ssdMS(self, target):
    """:param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1"""

    boxes = target["boxes"].cpu().numpy()  # [x1,y1,x2,y2] 没有作归一化（归一化0~1）
    gt_boxes = boxes / np.array([[self.resize, self.resize, self.resize, self.resize]], np.float32)  # 归一化0~1
    labels = target["labels"].cpu().numpy()
    gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

    _featureMap_list = []
    for i, stride in enumerate(self.strides):
        anchor = self.anchors[i]
        fh, fw = int(np.ceil(self.resize / stride)), int(np.ceil(self.resize / stride))
        featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

        anchor_xywh = x1y1x2y22xywh_np(anchor)

        iou = box_iou_np(anchor, gt_boxes)

        iou_v = iou.max(1)
        iou_i = iou.argmax(1)
        for i, v in enumerate(iou_v):
            if v >= 0.5:  # 正
                x, y, w, h = gt_boxes_xywh[iou_i[i]]
                a_x, a_y, a_w, a_h = anchor_xywh[i]
                featureMap[i, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
                featureMap[i, -1] = int(labels[iou_i[i]])
            elif v < 0.5:  # 负
                featureMap[i, -1] = 0

        # 与gt_boxes最大iou的anchor为正
        # iou_v = iou.max(0)
        iou_i = iou.argmax(0)
        for i, idx in enumerate(iou_i):
            x, y, w, h = gt_boxes_xywh[i]
            a_x, a_y, a_w, a_h = anchor_xywh[idx]
            featureMap[idx, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
            featureMap[idx, -1] = int(labels[i])

        _featureMap_list.append(featureMap)

    return np.concatenate(_featureMap_list, 0)


