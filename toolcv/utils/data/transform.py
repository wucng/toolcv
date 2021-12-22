# https://github.com/albumentations-team/albumentations
# pip install -U albumentations
from torchvision import transforms as T2
import torchvision.transforms.functional as F
from torch.nn import functional as F2
from torch.nn.functional import interpolate, pad
import random
from PIL import Image
import numpy as np
import torch
import cv2
from copy import deepcopy

# from toolcv.tools.cls.augment.transforms import RandomResizedCropAndInterpolation
from timm.data.transforms import RandomResizedCropAndInterpolation, _pil_interp

# from toolcv.tools.tools import getbbox_from_mask, compute_area_from_polygons, mask2polygons, getbbox_from_polygons, \
#     polygons2mask, mask2segmentation, segmentation2mask, segmentation2maskV2
from toolcv.utils.tools.tools import getbbox_from_mask, compute_area_from_polygons, mask2polygons, \
    getbbox_from_polygons, \
    polygons2mask, mask2segmentation, segmentation2mask, segmentation2maskV2

__all__ = ["RandomRoate", "RandomCrop", "RandomCropV2", "RandomCropV3", "ResizeRatio", "Resize",
           "RandomScale", "RandomShift", "RandomCropAndResize", "RandomHorizontalFlip", "RandomChoice",
           "NotChangeLabel",
           "RandomAffine", "RandomPerspective",
           "ToTensor", "Normalize", "RandomErasing"]


def filterByCenter(boxes, x_min, x_max, y_min, y_max, scale=[0.9, 1, 1]):
    # 按中心点过滤
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
    keep = torch.bitwise_and(cxcy[:, 0] > x_min * scale[0], cxcy[:, 0] < x_max * scale[1]) & \
           torch.bitwise_and(cxcy[:, 1] > y_min * scale[0], cxcy[:, 1] < y_max * scale[1])
    return keep


def filterByArea(boxes, x_min, x_max, y_min, y_max, ratio=0.3):
    # 按面积过滤
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = torch.clamp(x1, x_min, x_max)
        x2 = torch.clamp(x2, x_min, x_max)
        y1 = torch.clamp(y1, y_min, y_max)
        y2 = torch.clamp(y2, y_min, y_max)

        if (x2 - x1) / wh[i, 0] > ratio and (y2 - y1) / wh[i, 1] > ratio:
            keep.append(i)

    return torch.tensor(keep)


class RandomScale:
    """等比例缩放"""

    def __init__(self, rangs=[], p=0.5):
        if len(rangs) == 0: rangs = np.linspace(0.5, 1.5, 11)
        self.rangs = rangs
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        w, h = img.size
        if random.random() < self.p:
            scale = random.choice(self.rangs)
            _w = int(scale * w)
            _h = int(scale * h)

            return Resize((_h, _w))(img, target)

        if target is not None:
            return img, target
        return img


class ZoomOut:
    """获取小目标训练样本"""

    def __init__(self, rangs=[], fill=0, p=0.5):
        if len(rangs) == 0: rangs = np.linspace(0.3, 0.8, 11)
        self.rangs = rangs
        self.fill = fill
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        if random.random() > self.p:
            if target is not None:
                return img, target
            else:
                return img

        w, h = img.size
        scale = random.choice(self.rangs)
        _w = int(scale * w)
        _h = int(scale * h)
        if target is not None:
            img, target = Resize((_h, _w))(img, target)
        else:
            img = Resize((_h, _w))(img, target)

        # 随机贴到原图上
        tmp = np.ones([h, w, 3], np.uint8) * self.fill
        x = random.choice(range(0, w - _w))
        y = random.choice(range(0, h - _h))

        tmp[y:y + _h, x:x + _w] = np.array(img)
        img = Image.fromarray(tmp)

        if target is not None:
            if not isinstance(target, dict):
                target = np.array(target)
                tmp = np.zeros([h, w], np.uint8)
                tmp[y:y + _h, x:x + _w] = target
                target = Image.fromarray(tmp)
            else:
                boxes = target["boxes"]
                boxes[..., [0, 2]] += x
                boxes[..., [1, 3]] += y
                target["boxes"] = boxes
                if "masks" in target:
                    masks = target["masks"]
                    tmp = torch.zeros([masks.size(0), h, w], dtype=masks.dtype)
                    tmp[:, y:y + _h, x:x + _w] = masks
                    target["masks"] = tmp
                if "keypoints" in target:
                    keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                    keypoints[..., 0] += x
                    keypoints[..., 1] += y
                    target['keypoints'] = keypoints

            return img, target

        return img


class RandomCrop:
    """
    随机从 图像的 左上角 右上角 右下角 左下角 中心 选择
    """

    def __init__(self, ranges=[], center=False):
        if len(ranges) == 0: ranges = np.linspace(0.5, 1.5, 11).tolist()
        self.ranges = ranges
        self.center = center

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_params(self, w, h):
        cx, cy = w // 2, h // 2

        if random.choice([0, 1]):
            sx, ex, sy, ey = 0, w, 0, h
            return sx, ex, sy, ey

        idx = random.choice([0, 1, 2, 3, 4])
        # self.expand = (random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))
        # self.expand = (random.choice(np.linspace(0.5, 1.5, 11)), random.choice(np.linspace(0.5, 1.5, 11)))
        if self.center: idx = 4

        try:
            while True:
                rx = random.choice(self.ranges)
                ry = random.choice(self.ranges)
                if idx == 0:
                    # 左上角
                    sx = 0
                    ex = int(cx * rx)
                    sy = 0
                    ey = int(cy * ry)
                if idx == 1:
                    # 右上角
                    sx = int(cx * rx)
                    ex = w
                    sy = 0
                    ey = int(cy * ry)
                if idx == 2:
                    # 右下角
                    sx = int(cx * rx)
                    ex = w
                    sy = int(cy * ry)
                    ey = h
                if idx == 3:
                    # 左下角
                    sx = 0
                    ex = int(cx * rx)
                    sy = int(cy * ry)
                    ey = h
                if idx == 4:
                    # 中心
                    cx_ = int(cx / 2 * rx)
                    cy_ = int(cy / 2 * ry)

                    sx = cx_
                    ex = sx + 2 * (cx - cx_)
                    sy = cy_
                    ey = sy + 2 * (cy - cy_)

                sx = max(0, sx)
                sy = max(0, sy)
                ex = min(w, ex)
                ey = min(h, ey)

                if (ex - sx) / w >= 0.5 and (ey - sy) / h >= 0.5: break
                # if (ex - sx) / w * (ey - sy) / h > 0.5: break
        except:
            sx, ex, sy, ey = cx // 3, cx // 3 + (cx - cx // 3) * 2, cy // 3, cy // 3 + (cy - cy // 3) * 2

        return sx, ex, sy, ey

    def forward(self, x, target=None):
        """
        x:PIL.Image
        """
        w, h = x.size
        x_min, x_max, y_min, y_max = self.get_params(w, h)
        if target is not None:
            if not isinstance(target, dict):  # mask
                x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])
                target = Image.fromarray(np.array(target)[y_min:y_max, x_min:x_max])
                return x, target
            else:
                # tensor
                boxes = target["boxes"].clone()

                # keep = filterByCenter(boxes,x_min,x_max,y_min,y_max)
                keep = filterByArea(boxes, x_min, x_max, y_min, y_max)

                if keep.size(0) == 0:
                    return x, target
                else:
                    boxes = boxes[keep]
                    labels = target["labels"]
                    labels = labels[keep]
                    if "masks" in target:
                        masks = target["masks"]
                        masks = masks[keep]
                        masks = masks[:, y_min:y_max, x_min:x_max]
                        target["masks"] = masks
                    if "keypoints" in target:
                        keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                        keypoints = keypoints[keep]
                        keep = torch.bitwise_and(keypoints[..., 0] > x_min, keypoints[..., 0] < x_max) & \
                               torch.bitwise_and(keypoints[..., 1] > y_min, keypoints[..., 1] < y_max)
                        keypoints[..., 0] = keypoints[..., 0].clamp(x_min, x_max) - x_min
                        keypoints[..., 1] = keypoints[..., 1].clamp(y_min, y_max) - y_min
                        keypoints *= keep.float()[..., None]
                        target['keypoints'] = keypoints

                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min

                    target["boxes"] = boxes
                    target["labels"] = labels

                    x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])

                    return x, target

        x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])

        return x


class Resize:
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, target=None):
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        interpolation = _pil_interp(interpolation)

        w, h = x.size

        x = F.resize(x, self.size, interpolation)

        if target is not None:
            if not isinstance(target, dict):  # mask
                target = F.resize(target, self.size, Image.NEAREST)
            else:
                if "masks" in target:
                    target["masks"] = interpolate(target["masks"][None], self.size)[0]
                if "keypoints" in target:
                    keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                    keypoints[..., 0] *= self.size[1] / w
                    keypoints[..., 1] *= self.size[0] / h
                    target['keypoints'] = keypoints

                boxes = target["boxes"]
                # resize
                boxes[:, [0, 2]] *= self.size[1] / w
                boxes[:, [1, 3]] *= self.size[0] / h
                target["boxes"] = boxes

            return x, target

        return x


class ResizeRatio:
    """按原图比例缩放
        最大边 缩放到指定大小，最小边等比例缩放 不足的填充
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, target=None):
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        interpolation = _pil_interp(interpolation)

        w, h = x.size
        ratio = self.size[0] / max(w, h)
        resize_w = self.size[1] if w >= h else int(w * ratio)
        resize_h = int(h * ratio) if w >= h else self.size[0]

        pad_x = self.size[1] - resize_w
        pad_y = self.size[0] - resize_h

        x = F.resize(x, [resize_h, resize_w], interpolation)
        x = F.pad(x, [pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2])

        if target is not None:
            if not isinstance(target, dict):  # mask
                target = F.resize(target, [resize_h, resize_w], Image.NEAREST)
                target = F.pad(target, [pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2])
            else:
                if "masks" in target:
                    masks = interpolate(target["masks"][None], [resize_h, resize_w])
                    target["masks"] = pad(masks, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2])[0]

                if "keypoints" in target:
                    keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                    keypoints[..., 0] *= resize_w / w
                    keypoints[..., 1] *= resize_h / h
                    keypoints[..., 0] += pad_x // 2
                    keypoints[..., 1] += pad_y // 2
                    target['keypoints'] = keypoints

                boxes = target["boxes"]
                # resize
                boxes[:, [0, 2]] *= resize_w / w
                boxes[:, [1, 3]] *= resize_h / h
                boxes[:, [0, 2]] += pad_x // 2
                boxes[:, [1, 3]] += pad_y // 2

                target["boxes"] = boxes

            return x, target

        return x


class ResizeRatiov2:
    """按原图比例缩放
        最小边 缩放到指定大小，最大边等比例缩放 超过指定大小 则 裁剪掉
    """

    def __init__(self, size, interpolation='bilinear', mode="none"):
        self.size = size
        self.interpolation = interpolation
        self.mode = mode

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, target=None):
        # x_bk = x.copy()
        x_bk = deepcopy(x)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        interpolation = _pil_interp(interpolation)

        w, h = x.size
        ratio = self.size[0] / min(w, h)
        resize_w = int(w * ratio) if w >= h else self.size[1]
        resize_h = self.size[0] if w >= h else int(h * ratio)

        # pad_x = self.size[1] - resize_w
        # pad_y = self.size[0] - resize_h
        crop_x = resize_w - self.size[1]
        crop_y = resize_h - self.size[0]

        mode = self.mode
        if mode == "none":
            mode = random.choice(["top", "center", "down"])

        if mode == "top":
            start_x = 0
            start_y = 0
        elif mode == "down":
            start_x = crop_x
            start_y = crop_y
        elif mode == "center":
            start_x = crop_x // 2
            start_y = crop_y // 2

        x = F.resize(x, [resize_h, resize_w], interpolation)
        # x = F.pad(x, [pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2])
        x = F.crop(x, start_y, start_x, self.size[0], self.size[1])

        if target is not None:
            if not isinstance(target, dict):  # mask
                target = F.resize(target, [resize_h, resize_w], Image.NEAREST)
                # target = F.pad(target, [pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2])
                target = F.crop(target, start_y, start_x, self.size[0], self.size[1])
            else:
                boxes = target["boxes"].clone()
                # resize
                boxes[:, [0, 2]] *= resize_w / w
                boxes[:, [1, 3]] *= resize_h / h
                # boxes[:, [0, 2]] += pad_x // 2
                # boxes[:, [1, 3]] += pad_y // 2

                keep = filterByArea(boxes, start_x, start_x + self.size[1], start_y, start_y + self.size[0])
                if keep.size(0) == 0:
                    # return x_bk, target
                    return ResizeRatio(self.size, self.interpolation)(x_bk, target)

                if "masks" in target:
                    masks = interpolate(target["masks"][keep][None], [resize_h, resize_w])
                    # target["masks"] = pad(masks, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2])[0]
                    target["masks"] = masks[0][:, start_y:start_y + self.size[0], start_x:start_x + self.size[1]]

                if "keypoints" in target:
                    keypoints = target['keypoints'][keep]  # [-1,17,3] 3:(x,y,v)
                    keypoints[..., 0] *= resize_w / w
                    keypoints[..., 1] *= resize_h / h
                    # keypoints[..., 0] += pad_x // 2
                    # keypoints[..., 1] += pad_y // 2
                    keypoints[..., 0] -= start_x
                    keypoints[..., 1] -= start_y
                    keep = ((keypoints[..., 0] * keypoints[..., 1]) > 0).float()
                    keypoints *= keep[..., None]

                    target['keypoints'] = keypoints

                boxes = boxes[keep]
                boxes[:, [0, 2]] -= start_x
                boxes[:, [1, 3]] -= start_y
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.size[1] - 1)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.size[0] - 1)

                target["boxes"] = boxes
                target["labels"] = target["labels"][keep]

            return x, target

        return x


class RandomCropAndResize(RandomResizedCropAndInterpolation):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        super().__init__(size, scale, ratio, interpolation)

    def __call__(self, img, target=None):
        if target is None:
            return super().__call__(img)
        else:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            if isinstance(self.interpolation, (tuple, list)):
                interpolation = random.choice(self.interpolation)
            else:
                interpolation = self.interpolation
            interpolation = _pil_interp(interpolation)

            x_min = j
            x_max = j + w
            y_min = i
            y_max = i + h
            if not isinstance(target, dict):  # mask
                img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
                target = F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)
                return img, target
            else:
                boxes = target["boxes"].clone()
                labels = target["labels"].clone()

                # 按中心点过滤
                # keep = filterByCenter(boxes,x_min,x_max,y_min,y_max)
                keep = filterByArea(boxes, x_min, x_max, y_min, y_max)

                if keep.size(0) == 0:
                    w, h = img.size
                    img = F.resize(img, self.size, interpolation)
                    # resize
                    boxes[:, [0, 2]] *= self.size[1] / w
                    boxes[:, [1, 3]] *= self.size[0] / h

                    target["boxes"] = boxes
                    # target["labels"] = labels
                    if "masks" in target:
                        target["masks"] = interpolate(target["masks"][None], self.size)[0]
                    if "keypoints" in target:
                        keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                        keypoints[..., 0] *= self.size[1] / w
                        keypoints[..., 1] *= self.size[0] / h
                        target['keypoints'] = keypoints

                    return img, target

                boxes = boxes[keep]
                labels = labels[keep]

                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min

                # resize
                boxes[:, [0, 2]] *= self.size[1] / (x_max - x_min)
                boxes[:, [1, 3]] *= self.size[0] / (y_max - y_min)

                target["boxes"] = boxes
                target["labels"] = labels

                img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
                if "masks" in target:
                    masks = target["masks"][keep]
                    masks = masks[:, y_min:y_max, x_min:x_max]
                    target["masks"] = interpolate(masks[None], self.size)[0]

                if "keypoints" in target:
                    # crop
                    keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                    keypoints = keypoints[keep]
                    keep = torch.bitwise_and(keypoints[..., 0] > x_min, keypoints[..., 0] < x_max) & \
                           torch.bitwise_and(keypoints[..., 1] > y_min, keypoints[..., 1] < y_max)
                    keypoints[..., 0] = keypoints[..., 0].clamp(x_min, x_max) - x_min
                    keypoints[..., 1] = keypoints[..., 1].clamp(y_min, y_max) - y_min
                    keypoints *= keep.float()[..., None]
                    # resize
                    keypoints[..., 0] *= self.size[1] / (x_max - x_min)
                    keypoints[..., 1] *= self.size[0] / (y_max - y_min)
                    target['keypoints'] = keypoints

            return img, target


class RandomCropV2(RandomResizedCropAndInterpolation):
    """只做crop 没有做resize"""

    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        size = (0, 0)
        super().__init__(size, scale, ratio, interpolation)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, target=None):
        """
        x:PIL.Image
        """
        # w, h = x.size
        i, j, h, w = self.get_params(x, self.scale, self.ratio)
        x_min = j
        x_max = j + w
        y_min = i
        y_max = i + h
        if target is not None:
            if not isinstance(target, dict):  # mask
                x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])
                target = Image.fromarray(np.array(target)[y_min:y_max, x_min:x_max])
                return x, target
            else:
                # tensor
                boxes = target["boxes"].clone()

                # keep = filterByCenter(boxes,x_min,x_max,y_min,y_max)
                keep = filterByArea(boxes, x_min, x_max, y_min, y_max)

                if keep.size(0) == 0:
                    return x, target
                else:
                    boxes = boxes[keep]
                    labels = target["labels"]
                    labels = labels[keep]
                    if "masks" in target:
                        masks = target["masks"]
                        masks = masks[keep]
                        masks = masks[:, y_min:y_max, x_min:x_max]
                        target["masks"] = masks
                    if "keypoints" in target:
                        # crop
                        keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                        keypoints = keypoints[keep]
                        keep = torch.bitwise_and(keypoints[..., 0] > x_min, keypoints[..., 0] < x_max) & \
                               torch.bitwise_and(keypoints[..., 1] > y_min, keypoints[..., 1] < y_max)
                        keypoints[..., 0] = keypoints[..., 0].clamp(x_min, x_max) - x_min
                        keypoints[..., 1] = keypoints[..., 1].clamp(y_min, y_max) - y_min
                        keypoints *= keep.float()[..., None]
                        target['keypoints'] = keypoints

                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min

                    target["boxes"] = boxes
                    target["labels"] = labels

                    x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])

                    return x, target

        x = Image.fromarray(np.array(x)[y_min:y_max, x_min:x_max])

        return x


class RandomCropV3(RandomCrop):
    def __init__(self, ranges=[], center=False, ratio=0.5, p=0.5):
        if len(ranges) == 0: ranges = np.linspace(0.5, 1, 6).tolist()
        super().__init__(ranges, center)
        self.ratio = ratio
        self.p = p


    def get_params(self, w, h):
        cx, cy = w // 2, h // 2
        if self.center:
            sx, ex, sy, ey = cx // 3, cx // 3 + (cx - cx // 3) * 2, cy // 3, cy // 3 + (cy - cy // 3) * 2
            return sx, ex, sy, ey

        if random.random() < self.p:
            try:
                while True:
                    _w = int(random.choice(self.ranges) * w)
                    _h = int(random.choice(self.ranges) * h)
                    sx = random.choice(range(0, w - _w))
                    sy = random.choice(range(0, h - _h))
                    ex = sx + _w
                    ey = sy + _h

                    if (ex - sx) / w >= self.ratio and (ey - sy) / h >= self.ratio: break
            except:
                sx, ex, sy, ey = cx // 3, cx // 3 + (cx - cx // 3) * 2, cy // 3, cy // 3 + (cy - cy // 3) * 2
        else:
            sx, ex, sy, ey = 0, w, 0, h

        return sx, ex, sy, ey


def roatexy(x0, y0, cx, cy, radian):
    x0 -= cx
    y0 -= cy
    x1 = x0 * np.cos(radian) + y0 * np.sin(radian) + cx
    y1 = -x0 * np.sin(radian) + y0 * np.cos(radian) + cy

    return x1, y1


def roate_boxes(boxes, center, angle=90, fourpoints=False, ienet=False):
    radian = angle / 180.0 * np.pi

    x1, y1, x2, y2, x3, y3, x4, y4 = boxes
    cx, cy = center

    tmp = np.array([x1, y1, x2, y2, x3, y3, x4, y4])

    if radian != 0:
        x1, y1 = roatexy(x1, y1, cx, cy, radian)
        x2, y2 = roatexy(x2, y2, cx, cy, radian)
        x3, y3 = roatexy(x3, y3, cx, cy, radian)
        x4, y4 = roatexy(x4, y4, cx, cy, radian)

        # 重新排序 x1,y1,x2,y2,x3,y3,x4,y4
        tmp = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
        tmp = tmp.reshape(4, 2)

        _x1, _y1 = tmp[tmp[:, 0].argmin()]
        _x2, _y2 = tmp[tmp[:, 1].argmin()]
        _x3, _y3 = tmp[tmp[:, 0].argmax()]
        _x4, _y4 = tmp[tmp[:, 1].argmax()]

        tmp = np.array([_x1, _y1, _x2, _y2, _x3, _y3, _x4, _y4])

    if fourpoints:
        return tmp

    tmp = tmp.reshape(4, 2)
    x1, y1 = np.min(tmp, 0)
    x2, y2 = np.max(tmp, 0)

    if ienet:
        w, h = x2 - x1, y2 - y1
        w1 = x2 - tmp[1, 0]
        h1 = y2 - tmp[0, 1]
        return [x1, y1, x2, y2, w1, h1]

    return [x1, y1, x2, y2]  # 矩形的左上角坐标和右下角坐标


def roate(center, boxes, angle=90, fourpoints=False, ienet=False):
    """angle 是角度 不是弧度"""
    # radian = angle /180.0 * np.pi
    # h, w = img.shape[:2]
    # center = (w / 2, h / 2)
    # affine_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 求得旋转矩阵
    # img = cv2.warpAffine(img, affine_matrix, (w, h))

    w, h = center[0] * 2, center[1] * 2
    # 转成4点坐标 x1, y1, x2, y2, x3, y3, x4, y4
    boxes_ = []
    for x1, y1, x2, y2 in boxes:
        x3, y3 = x2, y2
        x2, y2 = x3, y1
        x4, y4 = x1, y3

        boxes_.append([x1, y1, x2, y2, x3, y3, x4, y4])

    tmp = [roate_boxes(box, center, angle) for box in boxes_]

    if fourpoints:
        boxes = np.stack(tmp, 0)
        boxes[..., [0, 2, 4, 6]] = boxes[..., [0, 2, 4, 6]].clip(0, w - 1)
        boxes[..., [1, 3, 5, 7]] = boxes[..., [1, 3, 5, 7]].clip(0, h - 1)

        return boxes

    if ienet:
        boxes = np.array(tmp)
        boxes[..., [0, 2, 4]] = boxes[..., [0, 2, 4]].clip(0, w - 1)
        boxes[..., [1, 3, 5]] = boxes[..., [1, 3, 5]].clip(0, h - 1)
        return boxes

    boxes = np.array(tmp)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)

    return boxes


class RandomRoate:
    """
    1、有 masks时 做旋转 可以通过masks 反算出 bbox 保证bbox的准确行  （旋转角度任意）
    2、没有 masks时 做旋转 bbox 准确性会降低 且 旋转的角度越大 误差越大 （旋转角度推荐 -15~15 or 10~10）
    """

    def __init__(self, angle=15, scale=1.0):
        """angle 是角度 不是弧度"""
        # radian = angle /180.0 * np.pi
        self.angle = angle

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        if random.choice([0, 1]):
            scale = 1.0  # random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
            angle = random.choice(range(-self.angle, self.angle, 5))
            img = np.array(img)
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            affine_matrix = cv2.getRotationMatrix2D(center, angle, scale)  # 求得旋转矩阵
            img = cv2.warpAffine(img, affine_matrix, (w, h))

            img = Image.fromarray(img.astype(np.uint8))

            if target is not None:
                if not isinstance(target, dict):
                    target = np.array(target)
                    target = cv2.warpAffine(target, affine_matrix, (w, h))
                    target = Image.fromarray(target.astype(np.uint8))
                else:
                    # tensor
                    boxes = target["boxes"].cpu().numpy()
                    boxes = roate(center, boxes, angle)
                    target["boxes"] = torch.from_numpy(boxes)
                    if "masks" in target:
                        masks = target["masks"].permute(1, 2, 0).cpu().numpy()
                        masks = cv2.warpAffine(masks, affine_matrix, (w, h))
                        masks = torch.from_numpy(masks).permute(2, 0, 1)
                        target["masks"] = masks

                        # 从masks反算bbox（因为旋转以后原来的box并不准确）
                        bs = masks.size(0)
                        masks = masks.cpu().numpy()
                        boxes_ = []
                        for i in range(bs):
                            boxes_.append(getbbox_from_mask(masks[i]))
                        target["boxes"] = torch.tensor(boxes_)

                    if "keypoints" in target:
                        keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                        bs, n, c = keypoints.shape
                        radian = angle / 180.0 * np.pi
                        tmp = keypoints.cpu().numpy()
                        _keys = []
                        for i in range(bs):
                            for j in range(n):
                                x, y, v = tmp[i, j]
                                _x, _y = roatexy(x, y, *center, radian=radian)
                                _keys.append([_x, _y, v])
                        target['keypoints'] = torch.from_numpy(np.array(_keys).reshape(bs, n, c))

        if target is not None:
            return img, target

        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        w, h = img.size
        if random.choice([0, 1]):
            img = F.hflip(img)
            if target is not None:
                if not isinstance(target, dict):
                    target = F.hflip(target)  # mask
                else:
                    boxes = target['boxes']
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target['boxes'] = boxes
                    if 'masks' in target:
                        masks = target['masks']
                        assert masks.ndim == 3
                        masks = torch.flip(masks, [2])
                        target['masks'] = masks
                    if 'keypoints' in target:
                        keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                        keypoints[..., 0] = w - keypoints[..., 0]
                        target['keypoints'] = keypoints

        if target is not None:
            return img, target
        return img


class ToTensor(object):
    def __call__(self, image, target=None):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image:Tensor
                target: Tensor
        """
        image = F.to_tensor(image)  # 0~1
        if target is not None:
            if not isinstance(target, dict): target = torch.from_numpy(np.array(target))
            return image, target
        return image


class Normalize(object):
    def __init__(self, image_mean=None, image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # RGB格式
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]  # ImageNet std
            # image_std = [1.0, 1.0, 1.0]
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, image, target=None):
        """
        :param image: Tensor
        :param target: Tensor
        :return:
                image: Tensor
                target: Tensor
        """
        # dtype, device = image.dtype, image.device
        # mean = torch.as_tensor(self.image_mean, dtype=torch.float32, device=device)
        # std = torch.as_tensor(self.image_std, dtype=torch.float32, device=device)
        # image = (image - mean[:, None, None]) / std[:, None, None]
        # image = T.Normalize(self.image_mean, self.image_std)(image)
        image = F.normalize(image, self.image_mean, self.image_std)
        if target is not None:
            return image, target
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image, target)
            return image


class RandomChoice(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        t = random.choice(self.transforms)
        if isinstance(t, (tuple, list)): t = Compose(t)
        if target is not None:
            image, target = t(image, target)
            return image, target
        else:
            image = t(image, target)
            return image


class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, target=None):

        if isinstance(self.transforms, (tuple, list)):
            if target is not None:
                for t in self.transforms:
                    if random.random() < self.p:
                        image, target = t(image, target)
                return image, target
            else:
                for t in self.transforms:
                    if random.random() < self.p:
                        image = t(image)
                return image

        if random.random() < self.p:
            return self.transforms(image, target)

        if target is not None:
            return image, target
        else:
            return image


class NotChangeLabel:
    def __init__(self, transform_list=[], mode="compose", p=0.5):
        if len(transform_list) == 0:
            transform_list = [T2.ColorJitter(0.5, 0.5, 0.5, 0.5), T2.GaussianBlur((5, 5)),
                              T2.RandomGrayscale(), T2.RandomAdjustSharpness(5),
                              T2.RandomAutocontrast(), T2.RandomEqualize(),
                              T2.RandomInvert(), T2.RandomSolarize(10)]

        self.transform_list = transform_list
        self.mode = mode
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        if self.mode == "RandomChoice":
            img = T2.RandomChoice(self.transform_list)(img)
        elif self.mode == "RandomOrder":  # 随机顺序执行
            img = T2.RandomOrder(self.transform_list)(img)
        elif self.mode == "RandomApply":  # 会报错
            img = T2.RandomApply(self.transform_list, self.p)
        else:
            img = T2.Compose(self.transform_list)(img)

        if target is not None:
            return img, target

        return img


class RandomShift:
    """随机平移"""

    def __init__(self, x=20, y=20, fill=0, interval=10):
        self.x = x
        self.y = y
        self.fill = fill
        self.interval = interval

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_params(self, w, h):
        if isinstance(self.x, float):
            x = int(self.x * w)
        else:
            x = self.x
        if isinstance(self.y, float):
            y = int(self.y * h)
        else:
            y = self.y
        x = random.choice(range(-x, x, self.interval))
        y = random.choice(range(-y, y, self.interval))

        shift_x = 0
        shift_y = 0
        if x >= 0:
            shift_x = x
            sx_1, ex_1 = x, w  # targ
            sx_2, ex_2 = 0, w - x  # src
        if x < 0:
            shift_x = x
            x = abs(x)
            sx_1, ex_1 = 0, w - x  # targ
            sx_2, ex_2 = x, w  # src
        if y >= 0:
            shift_y = y
            sy_1, ey_1 = y, h  # targ
            sy_2, ey_2 = 0, h - y  # src
        if y < 0:
            shift_y = y
            y = abs(y)
            sy_1, ey_1 = 0, h - y  # targ
            sy_2, ey_2 = y, h  # src

        return (sx_1, ex_1, sy_1, ey_1), (sx_2, ex_2, sy_2, ey_2), shift_x, shift_y

    def forward(self, img, target=None):
        w, h = img.size
        (sx_1, ex_1, sy_1, ey_1), (sx_2, ex_2, sy_2, ey_2), shift_x, shift_y = self.get_params(w, h)
        img = np.array(img)
        tmp = np.ones_like(img) * self.fill
        tmp[sy_1:ey_1, sx_1:ex_1] = img[sy_2:ey_2, sx_2:ex_2]
        img = Image.fromarray(tmp)
        if target is not None:
            if not isinstance(target, dict):
                target = np.array(target)
                tmp = np.ones_like(target) * 0
                tmp[sy_1:ey_1, sx_1:ex_1] = target[sy_2:ey_2, sx_2:ex_2]
                target = Image.fromarray(tmp)
            else:
                boxes = target['boxes']
                wh = boxes[:, 2:] - boxes[:, :2]
                # labels = target['labels']

                tmp = boxes.clone()
                tmp[:, [0, 2]] += shift_x
                tmp[:, [1, 3]] += shift_y
                tmp[:, [0, 2]] = tmp[:, [0, 2]].clamp(0, w)
                tmp[:, [1, 3]] = tmp[:, [1, 3]].clamp(0, h)
                keep = []
                for i, (x1, y1, x2, y2) in enumerate(tmp):
                    if (x2 - x1) / wh[i, 0] > 0.3 and (y2 - y1) / wh[i, 1] > 0.3:
                        keep.append(i)
                keep = torch.tensor(keep)

                target['labels'] = target['labels'][keep]
                target['boxes'] = tmp[keep]

                if "masks" in target:
                    masks = target['masks']
                    masks = masks[keep]
                    tmp = torch.zeros_like(masks)
                    tmp[:, sy_1:ey_1, sx_1:ex_1] = masks[:, sy_2:ey_2, sx_2:ex_2]
                    target['masks'] = tmp
                if "keypoints" in target:
                    keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                    keypoints = keypoints[keep]
                    keypoints[..., 0] += shift_x
                    keypoints[..., 1] += shift_y

                    keep = torch.bitwise_and(keypoints[..., 0] > 0, keypoints[..., 0] < w) & \
                           torch.bitwise_and(keypoints[..., 1] > 0, keypoints[..., 1] < h)

                    keypoints[..., 0] = keypoints[..., 0].clamp(0, w)
                    keypoints[..., 1] = keypoints[..., 1].clamp(0, h)

                    keypoints *= keep.float()[..., None]

                    target['keypoints'] = keypoints

        if target is not None:
            return img, target
        return img


class RandomErasing(T2.RandomErasing):
    """
    # 以下可以使用
    1、分类
    2、分割
    3、目标检测（带有 masks）

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, fill=0):
        super().__init__(p, scale, ratio, value, inplace)
        self.fill = fill

    def __call__(self, *args, **kwargs):
        """img:tensor"""
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )
            y, x, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            # img = F.erase(img, y, x, h, w, self.fill, self.inplace)
            # img[...,y:y + h, x:x + w] = self.fill

            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            fill = (self.fill - torch.tensor(image_mean)) / torch.tensor(image_std)
            img[..., y:y + h, x:x + w] = fill[:, None, None]

            if target is not None:
                if not isinstance(target, dict):
                    # target = np.array(target)
                    target[y:y + h, x:x + w] = 0
                    # target = Image.fromarray(target)
                else:
                    assert "masks" in target
                    masks = target["masks"]
                    masks[..., y:y + h, x:x + w] = 0
                    target["masks"] = masks

                    # 从masks反算bbox（因为旋转以后原来的box并不准确）
                    labels = target['labels'].cpu().numpy()
                    boxes = target['boxes'].cpu().numpy()
                    bs, h, w = masks.shape
                    masks = masks.cpu().numpy()
                    boxes_ = []
                    labels_ = []
                    masks_ = []
                    for i in range(bs):
                        if sum(np.unique(masks[i])) > 0:
                            x1, y1, x2, y2 = boxes[i]
                            # 查找轮廓
                            contours = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
                            if len(contours) == 3:
                                contours = contours[1]
                            else:
                                contours = contours[0]
                            for contour in contours:  # 可能存在多个轮廓
                                polygons = contour.reshape(-1).tolist()
                                # area = compute_area_from_polygons(polygons)
                                x_min, y_min, x_max, y_max = getbbox_from_polygons(polygons)
                                if (x_max - x_min) / (x2 - x1) >= 0.3 and (y_max - y_min) / (y2 - y1) >= 0.3:
                                    boxes_.append([x_min, y_min, x_max, y_max])
                                    labels_.append(labels[i])
                                    masks_.append(polygons2mask((h, w), polygons))

                            # boxes_.append(getbbox_from_mask(masks[i]))
                            # labels_.append(labels[i])
                    target["boxes"] = torch.tensor(boxes_)
                    target["labels"] = torch.tensor(labels_)
                    target["masks"] = torch.tensor(masks_)

        if target is not None:
            return img, target
        return img


class RandomPerspective(T2.RandomPerspective):
    """
    # 以下可以使用
    1、分类
    2、分割
    3、目标检测（带有 masks）
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=F.InterpolationMode.BILINEAR, fill=0):
        super().__init__(distortion_scale, p, interpolation, fill)

    def forward(self, img, target=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            width, height = F._get_image_size(img)
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img = F.perspective(img, startpoints, endpoints, self.interpolation, fill)

            if target is not None:
                if not isinstance(target, dict):
                    target = F.perspective(target, startpoints, endpoints, F.InterpolationMode.NEAREST, fill)
                else:
                    assert "masks" in target
                    masks = target["masks"]
                    masks = F.perspective(masks, startpoints, endpoints, F.InterpolationMode.NEAREST, fill)
                    target["masks"] = masks

                    # 从masks反算bbox（因为旋转以后原来的box并不准确）
                    bs = masks.size(0)
                    masks = masks.cpu().numpy()
                    boxes_ = []
                    for i in range(bs):
                        boxes_.append(getbbox_from_mask(masks[i]))
                    target["boxes"] = torch.tensor(boxes_)

        if target is not None:
            return img, target
        return img

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RandomAffine(T2.RandomAffine):
    """
    # 以下可以使用
    1、分类
    2、分割
    3、目标检测（带有 masks）
    """

    def __init__(
            self, degrees, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.NEAREST, fill=0,
            fillcolor=None, resample=None
    ):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, fillcolor, resample)

    def forward(self, img, target=None):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill)

        if target is not None:
            if not isinstance(target, dict):  # segment
                target = F.affine(target, *ret, interpolation=F.InterpolationMode.NEAREST, fill=fill)
            else:
                assert "masks" in target
                masks = target["masks"]
                masks = F.affine(masks, *ret, interpolation=F.InterpolationMode.NEAREST, fill=fill)
                target["masks"] = masks

                # 从masks反算bbox（因为旋转以后原来的box并不准确）
                bs = masks.size(0)
                masks = masks.cpu().numpy()
                boxes_ = []
                for i in range(bs):
                    boxes_.append(getbbox_from_mask(masks[i]))
                target["boxes"] = torch.tensor(boxes_)

        if target is not None:
            return img, target
        return img

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CutOut:
    """与  RandomErasing 类似
    # 以下可以使用
    1、分类
    2、分割
    3、目标检测（带有 masks）
    """

    def __init__(self, fill=0):
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.fill = (fill - torch.tensor(image_mean)) / torch.tensor(image_std)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        img_ = img.clone()
        alpha = random.choice(np.linspace(0.1, 0.5, 5))
        _, h, w = img.shape

        new_h = int(alpha * h)
        new_w = int(alpha * w)
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)

        x_min, x_max = x, x + new_w
        y_min, y_max = y, y + new_h

        img[:, y_min:y_max, x_min:x_max] = self.fill[:, None, None]

        if target is not None:
            if not isinstance(target, dict):
                target[y_min:y_max, x_min:x_max] = 0
            else:
                boxes = target["boxes"].clone()
                labels = target["labels"].clone()
                keep = filterByArea(boxes, x_min, x_max, y_min, y_max, 0.7)  # 排除在这个范围内
                keep = torch.tensor(list(set(range(len(boxes))) - set(keep.numpy().tolist())))
                if keep.size(0) == 0:
                    return img_, target

                boxes = boxes[keep]
                labels = labels[keep]
                # boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, x_min)
                # boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_max, w)
                # boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, y_min)
                # boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_max, h)

                target["boxes"] = boxes
                target["labels"] = labels

                if "masks" in target:
                    masks = target["masks"]
                    masks = masks[keep]
                    masks[:, y_min:y_max, x_min:x_max] = 0
                    target["masks"] = masks

                    # 从masks反算bbox（因为旋转以后原来的box并不准确）
                    labels = target['labels'].cpu().numpy()
                    boxes = target['boxes'].cpu().numpy()
                    bs, h, w = masks.shape
                    masks = masks.cpu().numpy()
                    boxes_ = []
                    labels_ = []
                    masks_ = []
                    for i in range(bs):
                        if sum(np.unique(masks[i])) > 0:
                            x1, y1, x2, y2 = boxes[i]
                            # 查找轮廓
                            contours = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
                            if len(contours) == 3:
                                contours = contours[1]
                            else:
                                contours = contours[0]
                            for contour in contours:  # 可能存在多个轮廓
                                polygons = contour.reshape(-1).tolist()
                                # area = compute_area_from_polygons(polygons)
                                x_min, y_min, x_max, y_max = getbbox_from_polygons(polygons)
                                if (x_max - x_min) / (x2 - x1) >= 0.3 and (y_max - y_min) / (y2 - y1) >= 0.3:
                                    boxes_.append([x_min, y_min, x_max, y_max])
                                    labels_.append(labels[i])
                                    masks_.append(polygons2mask((h, w), polygons))

                            # boxes_.append(getbbox_from_mask(masks[i]))
                            # labels_.append(labels[i])
                    target["boxes"] = torch.tensor(boxes_)
                    target["labels"] = torch.tensor(labels_)
                    target["masks"] = torch.tensor(masks_)

            return img, target

        return img


def mixup(self, idx, mode="cls", num_classes=None):
    img, target = self._load(idx)
    len_d = self.__len__()
    while True:
        idx_ = np.random.randint(0, len_d)
        img_, target_ = self._load(idx_)
        if mode == "cls":
            if idx_ != idx and target != target_: break
        if idx_ != idx: break

    if mode == "cls" or mode == "seg":
        alpha = random.choice(np.linspace(0.55, 0.9, 11))
    else:
        alpha = 0.5  # random.choice(np.linspace(0.4, 0.6, 3))
    img = img * alpha + img_ * (1 - alpha)

    if mode == "cls":
        assert num_classes is not None
        target = F2.one_hot(target.long(), num_classes) * alpha + F2.one_hot(target_.long(), num_classes) * (1 - alpha)
    if mode == "seg":
        assert num_classes is not None
        target = F2.one_hot(target.long(), num_classes).permute(2, 0, 1) * alpha + \
                 F2.one_hot(target_.long(), num_classes).permute(2, 0, 1) * (1 - alpha)
    if mode == "det":
        labels = target["labels"]
        boxes = target["boxes"]
        labels_ = target_["labels"]
        boxes_ = target_["boxes"]

        target["labels"] = torch.cat((labels, labels_), 0)
        target["boxes"] = torch.cat((boxes, boxes_), 0)

        if "masks" in target:
            target["masks"] = torch.cat((target["masks"], target_["masks"]), 0)

    return img, target


def cutmix(self, idx, mode="cls", num_classes=None):
    """
    # 以下可以使用
    1、分类
    2、分割
    3、目标检测（带有 masks）
    """

    # fill=0
    # image_mean = [0.485, 0.456, 0.406]
    # image_std = [0.229, 0.224, 0.225]
    # fill = (fill - torch.tensor(image_mean)) / torch.tensor(image_std)

    img, target = self._load(idx)
    len_d = self.__len__()
    _, h, w = img.shape
    while True:
        idx_ = np.random.randint(0, len_d)
        img_, target_ = self._load(idx_)
        if mode == "cls":
            if idx_ != idx and target != target_: break
        if idx_ != idx: break

    alpha = random.choice(np.linspace(0.1, 0.5, 5))
    new_h = int(alpha * h)
    new_w = int(alpha * w)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    x_min, x_max = x, x + new_w
    y_min, y_max = y, y + new_h

    img[:, y_min:y_max, x_min:x_max] = img_[:, y_min:y_max, x_min:x_max]

    if mode == "cls":
        assert num_classes is not None
        target = F2.one_hot(target.long(), num_classes) * (1 - alpha) + F2.one_hot(target_.long(), num_classes) * alpha
    if mode == "seg":
        target[y_min:y_max, x_min:x_max] = target_[y_min:y_max, x_min:x_max]
    if mode == "det":
        assert "masks" in target
        masks = target['masks'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        boxes = target['boxes'].cpu().numpy()
        masks_ = target_['masks'].cpu().numpy()
        labels_ = target_['labels'].cpu().numpy()
        boxes_ = target_['boxes'].cpu().numpy()

        masks[:, y_min:y_max, x_min:x_max] = 0
        tmp = np.zeros_like(masks_)
        tmp[:, y_min:y_max, x_min:x_max] = 1
        masks_ *= tmp

        masks_ = np.concatenate((masks, masks_), 0)
        labels_ = np.concatenate((labels, labels_), 0)
        boxes_ = np.concatenate((boxes, boxes_), 0)
        masks, boxes, labels = [], [], []
        for mask, label, (x1, y1, x2, y2) in zip(masks_, labels_, boxes_):
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
            if len(contours) == 3:
                contours = contours[1]
            else:
                contours = contours[0]
            for contour in contours:  # 可能存在多个轮廓
                polygons = contour.reshape(-1).tolist()
                if compute_area_from_polygons([polygons]) > 0:
                    x_min, y_min, x_max, y_max = getbbox_from_polygons(polygons)
                    # if (y_max - y_min) / (x_max - x_min) > 1 / 5 and (y_max - y_min) / (x_max - x_min) < 5:
                    if (y_max - y_min) / (y2 - y1) >= 0.3 and (x_max - x_min) / (x2 - x1) >= 0.3:
                        masks.append(polygons2mask((h, w), polygons))
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(label)

        target["boxes"] = torch.tensor(boxes)
        target["labels"] = torch.tensor(labels)
        target["masks"] = torch.tensor(masks)

    return img, target


def mosaic(self, idx, mode="cls", num_classes=None, fill=0):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    fill = (fill - torch.tensor(image_mean)) / torch.tensor(image_std)

    img, target = self._load(idx)
    len_d = self.__len__()
    _, h, w = img.shape
    cx = w / 2
    cy = h / 2

    alpha = np.random.choice(np.linspace(0.5, 1.5, 11), 2)

    x = int(cx * alpha[0])
    y = int(cy * alpha[1])

    x_list = [[0, x], [x, w], [x, w], [0, x]]
    y_list = [[0, y], [0, y], [y, h], [y, h]]
    area_ratio = np.array([x * y, (w - x) * y, (w - x) * (h - y), x * (h - y)]) / (w * h)

    while True:
        target_list = []
        img_list = []
        target_list.append(target)
        img_list.append(img)
        idxs = np.random.randint(0, len_d, 3)
        if np.unique(idxs).size == 3 and idx not in idxs:
            for idx in idxs:
                img, target = self._load(idx)
                target_list.append(target)
                img_list.append(img)
            if mode == "cls":
                if np.unique(target_list).size == 4: break  # 取到的4张图 类别不一样
            else:
                break

    new_img = torch.ones_like(img) * fill[:, None, None]
    if mode == "cls":
        new_target = 0
    if mode == "seg":
        new_target = torch.zeros_like(target)
    if mode == "det":
        new_target = {"labels": [], "boxes": []}
        if "masks" in target:
            new_target["masks"] = []

    for i, (img, target) in enumerate(zip(img_list, target_list)):
        x1, x2 = x_list[i]
        y1, y2 = y_list[i]
        new_h = y2 - y1
        new_w = x2 - x1
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        new_img[:, y1:y2, x1:x2] = img[:, y:y + new_h, x:x + new_w]

        if mode == "cls":
            assert num_classes is not None
            new_target += F2.one_hot(target.long(), num_classes) * area_ratio[i]

        if mode == "seg":
            new_target[y1:y2, x1:x2] = target[y:y + new_h, x:x + new_w]

        if mode == "det":
            x_min, x_max = x, x + new_w
            y_min, y_max = y, y + new_h
            labels = target["labels"].clone()
            boxes = target["boxes"].clone()

            keep = filterByArea(boxes, x_min, x_max, y_min, y_max)
            if keep.size(0) > 0:
                labels = labels[keep]
                boxes = boxes[keep]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min + x1
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min + y1

                new_target["labels"].extend(labels)
                new_target["boxes"].extend(boxes)

            if "masks" in target:
                if keep.size(0) > 0:
                    masks = target["masks"][keep]
                    new_masks = torch.zeros_like(masks)
                    new_masks[:, y1:y2, x1:x2] = masks[:, y:y + new_h, x:x + new_w]
                    new_target["masks"].extend(new_masks)

    if mode == "det":
        if len(new_target["labels"]) == 0: return self._load(idx)
        new_target["labels"] = torch.tensor(new_target["labels"])
        new_target["boxes"] = torch.stack(new_target["boxes"])
        if "masks" in target:
            new_target["masks"] = torch.stack(new_target["masks"])

    return new_img, new_target


def mosaicTwo(self, idx, mode="cls", num_classes=None, fill=0):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    fill = (fill - torch.tensor(image_mean)) / torch.tensor(image_std)

    img, target = self._load(idx)
    len_d = self.__len__()
    _, h, w = img.shape
    cx = w / 2
    cy = h / 2

    alpha = np.random.choice(np.linspace(0.5, 1.5, 11), 2)

    x = int(cx * alpha[0])
    y = int(cy * alpha[1])

    x_list = [[0, x], [x, w], [x, w], [0, x]]
    y_list = [[0, y], [0, y], [y, h], [y, h]]
    area_ratio = np.array([x * y, (w - x) * y, (w - x) * (h - y), x * (h - y)]) / (w * h)

    while True:
        target_list = []
        img_list = []
        target_list.append(target)
        img_list.append(img)
        idx_ = np.random.randint(0, len_d)
        if idx != idx_:
            img_, target_ = self._load(idx_)
            target_list.append(target_)
            img_list.append(img_)
            if mode == "cls":
                if np.unique(target_list).size == 2: break  # 取到的4张图 类别不一样
            else:
                break
    target_list.extend([target, target_])
    img_list.extend([img, img_])

    new_img = torch.ones_like(img) * fill[:, None, None]
    if mode == "cls":
        new_target = 0
    if mode == "seg":
        new_target = torch.zeros_like(target)
    if mode == "det":
        new_target = {"labels": [], "boxes": []}
        if "masks" in target:
            new_target["masks"] = []

    for i, (img, target) in enumerate(zip(img_list, target_list)):
        x1, x2 = x_list[i]
        y1, y2 = y_list[i]
        new_h = y2 - y1
        new_w = x2 - x1
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        new_img[:, y1:y2, x1:x2] = img[:, y:y + new_h, x:x + new_w]

        if mode == "cls":
            assert num_classes is not None
            new_target += F2.one_hot(target.long(), num_classes) * area_ratio[i]

        if mode == "seg":
            new_target[y1:y2, x1:x2] = target[y:y + new_h, x:x + new_w]

        if mode == "det":
            x_min, x_max = x, x + new_w
            y_min, y_max = y, y + new_h
            labels = target["labels"]
            boxes = target["boxes"]

            keep = filterByArea(boxes, x_min, x_max, y_min, y_max)
            if keep.size(0) > 0:
                labels = labels[keep]
                boxes = boxes[keep]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min + x1
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min + y1

                new_target["labels"].extend(labels)
                new_target["boxes"].extend(boxes)

            if "masks" in target:
                if keep.size(0) > 0:
                    masks = target["masks"][keep]
                    new_masks = torch.zeros_like(masks)
                    new_masks[:, y1:y2, x1:x2] = masks[:, y:y + new_h, x:x + new_w]
                    new_target["masks"].extend(new_masks)

    if mode == "det":
        if len(new_target["labels"]) == 0: return self._load(idx)
        new_target["labels"] = torch.tensor(new_target["labels"])
        new_target["boxes"] = torch.stack(new_target["boxes"])
        if "masks" in target:
            new_target["masks"] = torch.stack(new_target["masks"])

    return new_img, new_target


def mosaicOne(self, idx, mode="cls", num_classes=None, fill=0):
    img, target = self._load(idx)

    return MosaicOne(mode, num_classes, fill)(img, target)


class MosaicOne:
    def __init__(self, mode="cls", num_classes=None, fill=0):
        self.mode = mode
        self.num_classes = num_classes
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.fill = (fill - torch.tensor(image_mean)) / torch.tensor(image_std)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img, target=None):
        _, h, w = img.shape
        cx = w / 2
        cy = h / 2

        alpha = np.random.choice(np.linspace(0.5, 1.5, 11), 2)

        x = int(cx * alpha[0])
        y = int(cy * alpha[1])

        x_list = [[0, x], [x, w], [x, w], [0, x]]
        y_list = [[0, y], [0, y], [y, h], [y, h]]
        area_ratio = np.array([x * y, (w - x) * y, (w - x) * (h - y), x * (h - y)]) / (w * h)

        target_list = [target, target, target, target]
        img_list = [img, img, img, img]

        new_img = torch.ones_like(img) * self.fill[:, None, None]
        if self.mode == "cls":
            new_target = 0
        if self.mode == "seg":
            new_target = torch.zeros_like(target)
        if self.mode == "det":
            new_target = {"labels": [], "boxes": []}
            if "masks" in target:
                new_target["masks"] = []

        for i, (img, target) in enumerate(zip(img_list, target_list)):
            x1, x2 = x_list[i]
            y1, y2 = y_list[i]
            new_h = y2 - y1
            new_w = x2 - x1
            y = np.random.randint(0, h - new_h)
            x = np.random.randint(0, w - new_w)
            new_img[:, y1:y2, x1:x2] = img[:, y:y + new_h, x:x + new_w]

            if target is not None:
                if self.mode == "cls":
                    assert self.num_classes is not None
                    new_target += F2.one_hot(target.long(), self.num_classes) * area_ratio[i]

                if self.mode == "seg":
                    new_target[y1:y2, x1:x2] = target[y:y + new_h, x:x + new_w]

                if self.mode == "det":
                    x_min, x_max = x, x + new_w
                    y_min, y_max = y, y + new_h
                    labels = target["labels"]
                    boxes = target["boxes"]

                    keep = filterByArea(boxes, x_min, x_max, y_min, y_max)
                    if keep.size(0) > 0:
                        labels = labels[keep]
                        boxes = boxes[keep]
                        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x_min, x_max) - x_min + x1
                        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y_min, y_max) - y_min + y1

                        new_target["labels"].extend(labels)
                        new_target["boxes"].extend(boxes)

                    if "masks" in target:
                        if keep.size(0) > 0:
                            masks = target["masks"][keep]
                            new_masks = torch.zeros_like(masks)
                            new_masks[:, y1:y2, x1:x2] = masks[:, y:y + new_h, x:x + new_w]
                            new_target["masks"].extend(new_masks)

        if target is not None:
            if self.mode == "det":
                if len(new_target["labels"]) == 0: return self._load(idx)
                new_target["labels"] = torch.tensor(new_target["labels"])
                new_target["boxes"] = torch.stack(new_target["boxes"])
                if "masks" in target:
                    new_target["masks"] = torch.stack(new_target["masks"])

            return new_img, new_target

        return new_img


def multiscale(size, imgs, target, mode="cls"):
    """多尺寸训练"""
    if size is None:
        size = np.random.choice([224, 288, 352, 416, 480, 544, 608],
                                p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])

        # p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # p = [0.02, 0.05, 0.1, 0.13, 0.2, 0.2, 0.13, 0.1, 0.05, 0.02]
        # size = np.random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608], p=p)

    if isinstance(size, int): size = [size, size]
    _, _, h, w = imgs.shape
    if size != [h, w]:
        imgs = F2.interpolate(imgs, size, mode='bilinear')
        if mode == "cls":
            pass
        elif mode == "seg":
            target = F2.interpolate(target, size)
        elif mode == "det":
            boxes = target["boxes"]
            boxes[..., [0, 2]] *= size[1] / w
            boxes[..., [1, 3]] *= size[0] / h
            target["boxes"] = boxes
            if "masks" in target:
                masks = target['masks']
                masks = F2.interpolate(masks, size)
                target['masks'] = masks
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints[..., 0] *= size[1] / w
                keypoints[..., 1] *= size[0] / h
                target["keypoints"] = keypoints

    return imgs, target


def batch(imgs: list, stride=32, fill=0):
    nums = len(imgs)
    new_imgs = imgs
    # for i in range(nums):
    #     new_imgs.append(resizeMinMax(imgs[i],min_size,max_size))

    # 获取最大的 高和宽
    max_h = max([img.size(1) for img in new_imgs])
    max_w = max([img.size(2) for img in new_imgs])

    # 扩展到 stride的倍数(32倍数 GPU可以加速，且网络的最终stride=32)
    max_h = int(np.ceil(1.0 * max_h / stride) * stride)
    max_w = int(np.ceil(1.0 * max_w / stride) * stride)

    # 初始一个tensor 用于填充
    batch_img = torch.ones([nums, 3, max_h, max_w], device=imgs[0].device) * fill
    for i, img in enumerate(new_imgs):
        c, h, w = img.size()
        batch_img[i, :, :h, :w] = img  # 从左上角往下填充

    return batch_img


if __name__ == "__main__":
    """
    __all__ = ["RandomRoate", "RandomCrop", "RandomCropV2", "RandomCropV3", "ResizeRatio", "Resize",
           "RandomScale", "RandomShift", "RandomCropAndResize", "RandomHorizontalFlip", "RandomChoice",
           "NotChangeLabel",
           "RandomAffine", "RandomPerspective",
           "ToTensor", "Normalize", "RandomErasing"]
    """
    img = Image.open("000000041990.jpg")
    mask = Image.fromarray(np.array(img)[..., 0])

    transform = Compose([
        # RandomRoate(30),
        # RandomCrop(),
        # RandomCropV2(),
        # RandomCropV3(),
        # ResizeRatio((416, 416)),
        # Resize((416, 416)),
        # RandomScale(),
        # RandomShift(),
        # RandomCropAndResize((416, 416)),
        # RandomHorizontalFlip(),
        # RandomChoice([Resize((416, 416)), ResizeRatio((416, 416))]),
        # NotChangeLabel(),
        # RandomAffine(30),
        # RandomPerspective(),
        ZoomOut(),
        # ToTensor(),
        # Normalize(),
        # RandomErasing()
        # CutOut()
        # MosaicOne("seg")
    ])

    _, mask = transform(img, mask)
    # img.show()
    mask.show()
    # Image.fromarray(mask.numpy()).show()

    img = transform(img)
    # img.show()
