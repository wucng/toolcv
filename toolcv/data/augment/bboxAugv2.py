# from https://github.com/amdegroot/ssd.pytorch


import torch
from torch.nn import functional as F
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import PIL
from PIL import Image
import math
from math import cos, sin

# import random

__all__ = ["RandomHorizontalFlip", "RandomVerticallyFlip", "Resize", "RandomLight",
           "RandomColor", "RandomChanels", "RandomNoise", "RandomBlur",
           "RandomRotate", "RandomAffine", "RandomDrop", "RandomCrop", "CenterCrop", "ResizeFixMax",
           "ResizeFixMin", "ResizeMaxMin", "Pad", "RandomDropAndResizeMaxMin",
           "ResizeFixMinAndRandomCrop", "ResizeFixMaxAndPad", "RandomDropPixel",
           "RandomDropPixelV2", "RandomCutMix", "RandomCutMixV2", "RandomMosaic"
           ]


def filterByCenter(target, x1, y1, x2, y2):
    # 按中心过滤
    boxes = target["boxes"]
    center = (boxes[..., 2:] + boxes[..., :2]) / 2
    k1 = torch.bitwise_and(center[..., 0] >= x1, center[..., 0] <= x2 - 1)
    k2 = torch.bitwise_and(center[..., 1] >= y1, center[..., 1] <= y2 - 1)
    k = torch.bitwise_and(k1, k2)
    if k.sum() > 0:
        target["boxes"] = target["boxes"][k]
        target["labels"] = target["labels"][k]

        return target
    else:
        return None


def filter(target, imgSize=(), minhw=5):
    """过滤掉很小的框"""
    h, w = imgSize
    boxes = target["boxes"]
    # 裁剪到指定范围
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    wh = boxes[..., 2:] - boxes[..., :2]
    keep = (wh > minhw).sum(-1) == 2
    boxes = boxes[keep]
    target["boxes"] = boxes
    target["labels"] = target["labels"][keep]
    if len(boxes) == 0:
        return None

    return target


class RandomHorizontalFlip:  # 左右镜像
    def __init__(self):
        pass

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image).copy()
            width = image.shape[1]
            image = image[:, ::-1, :]
            target["boxes"][..., [0, 2]] = width - target["boxes"][..., [2, 0]]
            image = PIL.Image.fromarray(image.astype(np.uint8))

        return image, target


class RandomVerticallyFlip:  # 上下镜像
    def __init__(self):
        pass

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image).copy()
            height = image.shape[0]
            image = image[::-1, :, :]
            target["boxes"][..., [1, 3]] = height - target["boxes"][..., [3, 1]]
            image = PIL.Image.fromarray(image.astype(np.uint8))

        return image, target


class Resize(object):
    def __init__(self, size=(300, 300)):
        if not isinstance(size, (tuple, list)):
            size = (size, size)
        self.size = size

    def __call__(self, image, target):
        """
        Args:
            image: PIL
            target:

        Returns:

        """
        image = np.asarray(image)
        h, w = image.shape[:2]
        scale_h, scale_w = self.size[0] / h, self.size[1] / w
        image = cv2.resize(image, self.size[::-1])  # (w,h)

        if "boxes" in target:
            target["boxes"][..., [0, 2]] *= scale_w
            target["boxes"][..., [1, 3]] *= scale_h

            target["resize"] = torch.as_tensor(self.size, dtype=torch.float32)
            target["original_size"] = torch.as_tensor((h, w), dtype=torch.float32)

        image = PIL.Image.fromarray(image)

        return image, target


class RandomLight(object):
    def __init__(self, delta=[0.3, 2.0]):  # [0.5,1.5]
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image).astype(np.float32)
            image *= random.uniform(self.delta[0], self.delta[1])
            image = image.clip(0, 255)
            image = PIL.Image.fromarray(image.astype(np.uint8))

        return image, target


class RandomColor(object):
    def __init__(self, delta=[0.3, 2.0]):  # [0.5,1.5]
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image).astype(np.float32)
            image[..., random.randint(0, 3)] *= random.uniform(self.delta[0], self.delta[1])
            image = image.clip(0, 255)
            image = PIL.Image.fromarray(image.astype(np.uint8))

        return image, target


class RandomChanels(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image)
            index = np.arange(0, 3)
            np.random.shuffle(index)
            image = image[..., index]
            image = PIL.Image.fromarray(image)

        return image, target


def noise(image, rand=0.1):
    i = random.randint(0, 3)
    row, column, channel = image.shape
    image.astype("float")
    if i == 0:  #
        noise_salt = np.random.randint(0, 256, (row, column, channel))
        noise_salt = np.where(noise_salt < rand * 256, 255, 0)
        noise_salt.astype("float")
        image = image + noise_salt
    elif i == 1:
        noise_pepper = np.random.randint(0, 256, (row, column, channel))
        noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
        noise_pepper.astype("float")
        image = image + noise_pepper
    else:
        Gauss_noise = np.random.normal(0, 50, (row, column, channel))
        image = image + Gauss_noise

    image = image.clip(0, 255).astype(np.uint8)

    return image


class RandomNoise(object):
    def __init__(self, rand=0.1):
        self.rand = rand

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image)
            image = noise(image, self.rand)
            image = PIL.Image.fromarray(image)

        return image, target


def blur(image, ksize=5):
    if not isinstance(ksize, tuple):
        ksize = (ksize, ksize)
    i = random.randint(0, 3)
    if i == 0:  # 均值滤波
        image = cv2.blur(image, ksize)
    elif i == 1:
        image = cv2.medianBlur(image, ksize[0])
    elif i == 2:
        image = cv2.GaussianBlur(image, ksize, 0)
    return image


class RandomBlur(object):
    def __init__(self, ksize=5):
        self.ksize = ksize

    def __call__(self, image, target):
        if random.randint(0, 2):
            image = np.asarray(image)
            image = blur(image, self.ksize)
            image = PIL.Image.fromarray(image)

        return image, target


def rotate(image, target, angle=5, scale=1.0):
    height, width, channels = image.shape
    # 求得图片中心点， 作为旋转的轴心
    cx = int(width / 2)
    cy = int(height / 2)
    # 旋转的中心
    center = (cx, cy)
    new_dim = (width, height)
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    image = cv2.warpAffine(image, rot_mat, new_dim)

    bboxes = target["boxes"].numpy()

    for i, bbox in enumerate(bboxes):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))

        concat = np.vstack((point1, point2, point3, point4))
        concat = concat.astype(np.int32)

        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        bboxes[i][0] = rx
        bboxes[i][1] = ry
        bboxes[i][2] = rx + rw
        bboxes[i][3] = ry + rh

    target["boxes"] = torch.from_numpy(bboxes)

    return image, target


class RandomRotate(object):
    def __init__(self, angle=15, scale=(0.8, 1.2)):
        self.angle = angle
        self.scale = scale

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image)
                image, target = rotate(image, target, random.randint(-self.angle, self.angle),
                                       random.uniform(self.scale[0], self.scale[1]))
                target = filter(target, image.shape[:2])
                h, w = image.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                image = PIL.Image.fromarray(image)
                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


def affine(image, target, alpha):
    height, width, channels = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

    pts2 = np.float32([
        [random.uniform(pts1[0, 0] * (1 - alpha[0]), pts1[0, 0] * (1 + alpha[0])),
         random.uniform(pts1[0, 1] * (1 - alpha[0]), pts1[0, 1] * (1 + alpha[0]))],
        [random.uniform(pts1[1, 0] * (1 - alpha[1]), pts1[1, 0] * (1 + alpha[1])),
         random.uniform(pts1[1, 1] * (1 - alpha[1]), pts1[1, 1] * (1 + alpha[1]))],
        [random.uniform(pts1[2, 0] * (1 - alpha[2]), pts1[2, 0] * (1 + alpha[2])),
         random.uniform(pts1[2, 1] * (1 - alpha[2]), pts1[2, 1] * (1 + alpha[2]))]])

    rot_mat = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, rot_mat, (width, height))

    bboxes = target["boxes"].numpy()

    for i, bbox in enumerate(bboxes):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))

        concat = np.vstack((point1, point2, point3, point4))
        concat = concat.astype(np.int32)

        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        bboxes[i][0] = rx
        bboxes[i][1] = ry
        bboxes[i][2] = rx + rw
        bboxes[i][3] = ry + rh

    target["boxes"] = torch.from_numpy(bboxes)

    return image, target


class RandomAffine(object):
    def __init__(self, alpha=0.5):
        self.alpha = [random.uniform(0.1, alpha) for _ in range(3)]

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image)
                image, target = affine(image, target, self.alpha)
                target = filter(target, image.shape[:2])
                h, w = image.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                image = PIL.Image.fromarray(image)
                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


class RandomDrop:
    def __init__(self, alpha=0.2):  # 随机裁剪掉0～20%（最大裁剪掉 20%）
        self.alpha = alpha

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image)
                height, width, channels = image.shape
                newH = int(height * (1 - self.alpha))
                newW = int(width * (1 - self.alpha))

                x = random.randint(0, width - newW)
                y = random.randint(0, height - newH)
                image = image[y:y + newH, x:x + newW]

                target["boxes"] -= torch.tensor((x, y, x, y), dtype=torch.float32)
                # target["boxes"][...,[0,2]] = target["boxes"][...,[0,2]].clamp(0,newW-1)
                # target["boxes"][...,[1,3]] = target["boxes"][...,[1,3]].clamp(0,newH-1)

                target = filter(target, (newH, newW))
                h, w = image.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                image = PIL.Image.fromarray(image)

                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


class RandomCrop:
    def __init__(self, size=(300, 300)):  # crop指定大小
        self.size = size

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image)
                height, width, channels = image.shape
                newH, newW = self.size
                assert height >= newH and width >= newW

                x = random.randint(0, width - newW)
                y = random.randint(0, height - newH)
                image = image[y:y + newH, x:x + newW]

                # 按中心过滤
                target = filterByCenter(target, x, y, x + newW, y + newH)

                target["boxes"] -= torch.tensor((x, y, x, y), dtype=torch.float32)
                # target["boxes"][...,[0,2]] = target["boxes"][...,[0,2]].clamp(0,newW-1)
                # target["boxes"][...,[1,3]] = target["boxes"][...,[1,3]].clamp(0,newH-1)

                target = filter(target, (newH, newW))
                h, w = image.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                target["resize"] = torch.as_tensor(self.size, dtype=torch.float32)
                image = PIL.Image.fromarray(image)

                if target is None:
                    image, target = Resize(self.size)(_image, _target)
            except:
                image, target = Resize(self.size)(_image, _target)

            return image, target
        else:
            return Resize(self.size)(image, target)


class RandomCropV2:
    def __init__(self, size=(300, 300)):  # crop指定大小
        self.size = size

    def __call__(self, image2, target2):
        while True:
            image = image2.copy()
            target = target2.copy()

            image = np.asarray(image)
            height, width, channels = image.shape
            newH, newW = self.size
            assert height >= newH and width >= newW

            x = random.randint(0, width - newW)
            y = random.randint(0, height - newH)
            image = image[y:y + newH, x:x + newW]

            # 按中心过滤
            target = filterByCenter(target, x, y, x + newW, y + newH)

            if target is not None:
                h, w = image.shape[:2]
                target["boxes"] -= torch.tensor((x, y, x, y), dtype=torch.float32)
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                target["resize"] = torch.as_tensor(self.size, dtype=torch.float32)
                image = PIL.Image.fromarray(image)

                return image, target


class CenterCrop:
    def __init__(self, size=(300, 300)):  # crop指定大小
        self.size = size

    def __call__(self, image, target):
        # if random.randint(0, 2):
        _image = image.copy()
        _target = target.copy()
        try:
            image = np.asarray(image)
            height, width, channels = image.shape
            newH, newW = self.size
            assert height >= newH and width >= newW

            cy, cx = height // 2, width // 2

            x = cx - newW // 2
            y = cy - newH // 2
            image = image[y:y + newH, x:x + newW]

            # 按中心过滤
            target = filterByCenter(target, x, y, x + newW, y + newH)

            target["boxes"] -= torch.tensor((x, y, x, y), dtype=torch.float32)
            # target["boxes"][...,[0,2]] = target["boxes"][...,[0,2]].clamp(0,newW-1)
            # target["boxes"][...,[1,3]] = target["boxes"][...,[1,3]].clamp(0,newH-1)

            target = filter(target, (newH, newW))
            h, w = image.shape[:2]
            target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
            target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
            target["resize"] = torch.as_tensor(self.size, dtype=torch.float32)
            image = PIL.Image.fromarray(image)

            if target is None:
                image, target = _image, _target
        except:
            image, target = _image, _target

        return image, target


class ResizeFixMax:
    """
    将最大边缩放到指定大小，最小边等比例缩放
    """

    def __init__(self, size=512):
        self.size = size

    def __call__(self, image, target):
        image = np.asarray(image)
        height, width, channels = image.shape
        scale = 1.0 * max(height, width) / self.size
        newH, newW = int(height / scale), int(width / scale)

        return Resize((newH, newW))(image, target)


class ResizeFixMin:
    """
    将最小边缩放到指定大小，最大边等比例缩放
    """

    def __init__(self, size=512):
        self.size = size

    def __call__(self, image, target):
        image = np.asarray(image)
        height, width, channels = image.shape
        scale = 1.0 * min(height, width) / self.size
        newH, newW = int(height / scale), int(width / scale)

        return Resize((newH, newW))(image, target)


class ResizeMaxMin:
    """
    将最小边缩放到指定大小，最大边等比例缩放,最大边不能超过指定大小
    将最大边缩放到指定大小，最小边等比例缩放,最小边不能超过指定大小
    """

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        image = np.asarray(image)
        height, width, channels = image.shape
        scale = 1.0 * min(height, width) / self.min_size
        newH, newW = int(height / scale), int(width / scale)
        if max(newH, newW) > self.max_size:
            scale = 1.0 * max(height, width) / self.max_size
            newH, newW = int(height / scale), int(width / scale)

        return Resize((newH, newW))(image, target)


class Pad:
    def __init__(self, value=114):
        self.value = value

    def __call__(self, image, target):
        image = np.asarray(image)

        height, width, channels = image.shape
        if height > width:  # 宽填充
            newImage = np.ones((height, height, channels), dtype=np.uint8) * self.value
            diff = (height - width) // 2
            newImage[:, diff:diff + width, :] = image
            image = newImage

            target["boxes"][..., [0, 2]] += diff
            target["resize"] = torch.as_tensor((height, height), dtype=torch.float32)

        elif height < width:  # 高填充
            newImage = np.ones((width, width, channels), dtype=np.uint8) * self.value
            diff = (width - height) // 2
            newImage[diff:diff + height, ...] = image
            image = newImage

            target["boxes"][..., [1, 3]] += diff
            target["resize"] = torch.as_tensor((width, width), dtype=torch.float32)
        else:
            pass

        image = PIL.Image.fromarray(image)

        return image, target


# 组合使用
class RandomDropAndResizeMaxMin:
    def __init__(self, alpha=0.2, min_size=600, max_size=1000):
        self.alpha = alpha
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        image, target = RandomDrop(self.alpha)(image, target)
        return ResizeMaxMin(self.min_size, self.max_size)(image, target)


class ResizeFixMinAndRandomCrop:
    def __init__(self, min_size=320, size=(300, 300)):
        self.min_size = min_size
        self.size = size

    def __call__(self, image, target):
        image, target = ResizeFixMin(self.min_size)(image, target)
        return RandomCrop(self.size)(image, target)


class ResizeFixMinAndRandomCropV2:
    def __init__(self, size_range=[224, 256, 300, 416, 512, 608],
                 ratio_range=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.size_range = size_range
        self.ratio_range = ratio_range

    def __call__(self, image, target):
        min_size = np.random.choice(self.size_range)
        ratio = np.random.choice(self.ratio_range)
        size = (int(min_size * ratio), int(min_size * ratio))

        image, target = ResizeFixMin(min_size)(image, target)
        return RandomCropV2(size)(image, target)


class Patch:
    def __init__(self, size=(416, 416)):
        self.size = size

    def __call__(self, image, target):
        image = np.array(image)
        h, w, c = image.shape
        tmp = np.zeros([*self.size, c], image.dtype)

        if max(w, h) <= min(self.size):
            pass
        else:
            scale = min(self.size) / max(w, h)
            size = (math.ceil(h * scale), math.ceil(w * scale))

            image, target = Resize(size)(PIL.Image.fromarray(image), target)
            image = np.array(image)
            h, w, c = image.shape

        x = np.random.choice(np.arange(0, self.size[0] - w))
        y = np.random.choice(np.arange(0, self.size[1] - h))

        tmp[y:y + h, x:x + w] = image
        boxes = target['boxes']
        boxes[..., [0, 2]] += x
        boxes[..., [1, 3]] += y
        target['boxes'] = boxes

        return PIL.Image.fromarray(tmp), target


class ResizeFixMinAndRandomCropV2AndPatch:
    def __init__(self, size_range=[224, 256, 300, 416, 512, 608],
                 ratio_range=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.size_range = size_range
        self.ratio_range = ratio_range

    def __call__(self, image, target):
        min_size = np.random.choice(self.size_range)
        ratio = np.random.choice(self.ratio_range)
        size = (int(min_size * ratio), int(min_size * ratio))

        image, target = ResizeFixMin(min_size)(image, target)
        image, target = RandomCropV2(size)(image, target)

        return Patch((min_size, min_size))(image, target)


class ResizeFixMaxAndPad:
    def __init__(self, max_size=300, value=114):
        self.max_size = max_size
        self.value = value

    def __call__(self, image, target):
        image, target = ResizeFixMax(self.max_size)(image, target)
        return Pad(self.value)(image, target)


# ----------------------------------------

class RandomDropPixel:
    """整张图片应用"""

    def __init__(self, size=10, nums=20, value=114):
        self.size = size
        self.nums = nums
        self.value = value

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image).copy()
                h, w = image.shape[:2]
                top = min(h, w) // 7
                nums = random.randint(self.nums, 100)
                for i in range(nums):
                    size = random.randint(self.size, top)
                    x = random.randint(0, w - size)
                    y = random.randint(0, h - size)
                    image[y:y + size, x:x + size] = self.value

                image = PIL.Image.fromarray(image)
            except:
                image, target = _image, _target

        return image, target


class RandomDropPixelV2:
    """按box做"""

    def __init__(self, value=114):
        self.value = value

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image).copy()
                boxes = target["boxes"].numpy()
                for box in boxes:
                    # if random.randint(0, 2):
                    x1, y1, x2, y2 = list(map(int, box))
                    h, w = y2 - y1, x2 - x1
                    sizew = random.randint(w // 3, w // 2)
                    sizeh = random.randint(h // 3, h // 2)

                    x = random.randint(x1, x2 - sizew)
                    y = random.randint(y1, y2 - sizeh)
                    image[y:y + sizeh, x:x + sizew] = self.value

                image = PIL.Image.fromarray(image)
            except:
                image, target = _image, _target

        return image, target


class RandomCutMix:
    """单张图片做CutMix"""

    def __init__(self):
        pass

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image).copy()
                height, width = image.shape[:2]
                boxes = target["boxes"]
                labels = target["labels"]
                length = boxes.size(0)
                nums = random.randint(1, length // 2 if length // 2 > 2 else 2)

                _boxes = []
                _labels = []
                for i in range(nums):
                    index = random.randint(0, length)
                    box = boxes[index]
                    label = labels[index]

                    x1, y1, x2, y2 = list(map(int, box))
                    h, w = y2 - y1, x2 - x1

                    if width - w > 1 and height - h > 1:
                        x = random.randint(0, width - w)
                        y = random.randint(0, height - h)
                        image[y:y + h, x:x + w] = image[y1:y2, x1:x2]

                        _boxes.append([x, y, x + w, y + h])
                        _labels.append(label)

                if len(_labels) > 0:
                    boxes = torch.cat((boxes, torch.tensor(_boxes, dtype=boxes.dtype)), 0)
                    labels = torch.cat((labels, torch.tensor(_labels, dtype=labels.dtype)), 0)

                target["boxes"] = boxes
                target["labels"] = labels

                target = filter(target, (height, width))

                image = PIL.Image.fromarray(image)

                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


class RandomCutMixV2:
    """单张图片做CutMix"""

    def __init__(self, expand=5):
        self.expand = expand  # 向外扩充像素值

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                image = np.asarray(image).copy()
                height, width = image.shape[:2]
                boxes = target["boxes"]
                labels = target["labels"]
                length = boxes.size(0)
                nums = random.randint(1, length // 2 if length // 2 > 2 else 2)

                _boxes = []
                _labels = []
                for i in range(nums):
                    index = random.randint(0, length)
                    box = boxes[index]
                    label = labels[index]

                    x1, y1, x2, y2 = list(map(int, box))
                    h, w = y2 - y1, x2 - x1

                    if width - w > 1 and height - h > 1:
                        x = random.randint(0, width - w)
                        y = random.randint(0, height - h)

                        # image[y:y + h, x:x + w] = image[y1:y2,x1:x2]

                        image[y - self.expand:y + h + self.expand, x - self.expand:x + w + self.expand] = \
                            image[y1 - self.expand:y2 + self.expand, x1 - self.expand:x2 + self.expand]

                        _boxes.append([x, y, x + w, y + h])

                        _labels.append(label)

                if len(_labels) > 0:
                    boxes = torch.cat((boxes, torch.tensor(_boxes, dtype=boxes.dtype)), 0)
                    labels = torch.cat((labels, torch.tensor(_labels, dtype=labels.dtype)), 0)

                target["boxes"] = boxes
                target["labels"] = labels

                target = filter(target, (height, width))

                image = PIL.Image.fromarray(image)
                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


class RandomMosaic:
    """单张图片做Mosaic"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, image, target):
        if random.randint(0, 2):
            _image = image.copy()
            _target = target.copy()
            try:
                _boxes = []
                _labels = []
                boxes = target["boxes"].numpy()
                labels = target["labels"].numpy()
                image = np.asarray(image)
                newImg = np.zeros_like(image)
                height, width = image.shape[:2]
                cy, cx = height // 2, width // 2

                x = random.randint(cx * (1 - self.alpha), cx * (1 + self.alpha))
                y = random.randint(cy * (1 - self.alpha), cy * (1 + self.alpha))

                # 左上角
                y1 = random.randint(0, height - y)
                x1 = random.randint(0, width - x)
                newImg[0:y, 0:x] = image[y1:y1 + y, x1:x1 + x]
                _target = {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}
                _target = filterByCenter(_target, x1, y1, x1 + x, y1 + y)

                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

                # 右上角
                y1 = random.randint(0, height - y)
                x1 = random.randint(0, x)
                newImg[0:y, x:width] = image[y1:y1 + y, x1:x1 + width - x]
                _target = {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}
                _target = filterByCenter(_target, x1, y1, x1 + width - x, y1 + y)

                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

                # 右下角
                y1 = random.randint(0, y)
                x1 = random.randint(0, x)
                newImg[y:height, x:width] = image[y1:y1 + height - y, x1:x1 + width - x]
                _target = {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}
                _target = filterByCenter(_target, x1, y1, x1 + width - x, y1 + height - y)
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

                # 左下角
                y1 = random.randint(0, y)
                x1 = random.randint(0, width - x)
                newImg[y:height, 0:x] = image[y1:y1 + height - y, x1:x1 + x]
                _target = {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}
                _target = filterByCenter(_target, x1, y1, x1 + x, y1 + height - y)
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

                target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
                target["labels"] = torch.tensor(_labels)

                target = filter(target, (height, width))
                h, w = newImg.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                image = newImg

                image = PIL.Image.fromarray(image)
                if target is None:
                    image, target = _image, _target
            except:
                image, target = _image, _target

        return image, target


class RandomMosaicV2:
    """单张图片做Mosaic"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, image2, target2):
        while True:
            image = image2.copy()
            target = target2.copy()

            _boxes = []
            _labels = []

            image = np.asarray(image)
            newImg = np.zeros_like(image)
            height, width = image.shape[:2]
            cy, cx = height // 2, width // 2

            x = random.randint(cx * (1 - self.alpha), cx * (1 + self.alpha))
            y = random.randint(cy * (1 - self.alpha), cy * (1 + self.alpha))

            # 左上角
            y1 = random.randint(0, height - y)
            x1 = random.randint(0, width - x)
            x2 = x1 + x
            y2 = y1 + y
            newImg[0:y, 0:x] = image[y1:y2, x1:x2]
            _target = filterByCenter(target.copy(), x1, y1, x2, y2)
            if _target is not None:
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x2) - x1
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y2) - y1

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

            # 右上角
            y1 = random.randint(0, height - y)
            x1 = random.randint(0, x)
            x2 = x1 + width - x
            y2 = y1 + y
            newImg[0:y, x:width] = image[y1:y2, x1:x2]
            _target = filterByCenter(target.copy(), x1, y1, x2, y2)
            if _target is not None:
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x2) - x1 + x
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y2) - y1

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

            # 右下角
            y1 = random.randint(0, y)
            x1 = random.randint(0, x)
            x2 = x1 + width - x
            y2 = y1 + height - y
            newImg[y:height, x:width] = image[y1:y2, x1:x2]
            _target = filterByCenter(target.copy(), x1, y1, x2, y2)
            if _target is not None:
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x2) - x1 + x
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y2) - y1 + y

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

            # 左下角
            y1 = random.randint(0, y)
            x1 = random.randint(0, width - x)
            x2 = x1 + x
            y2 = y1 + height - y
            newImg[y:height, 0:x] = image[y1:y2, x1:x2]
            _target = filterByCenter(target.copy(), x1, y1, x2, y2)
            if _target is not None:
                tboxes = _target['boxes'].numpy()
                tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x2) - x1
                tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y2) - y1 + y

                _boxes.extend(tboxes)
                _labels.extend(_target['labels'].numpy())

            if len(_labels) > 0:
                target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
                target["labels"] = torch.tensor(_labels)
                image = newImg
                h, w = newImg.shape[:2]
                target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clamp(0, w - 1)
                target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clamp(0, h - 1)
                image = PIL.Image.fromarray(image)

                return image, target


# 4张图片做 mosaic
def mosaicFourImg(self, idx, alpha=0.5):
    try:
        _boxes = []
        _labels = []

        index = torch.randperm(self.len).tolist()
        if idx + 3 >= self.len:
            idx = 0

        idx2 = index[idx + 1]
        idx3 = index[idx + 2]
        idx4 = index[idx + 3]

        img, mask, boxes, labels, img_path = self.load(idx)
        img2, mask2, boxes2, labels2, _ = self.load(idx2)
        img3, mask3, boxes3, labels3, _ = self.load(idx3)
        img4, mask4, boxes4, labels4, _ = self.load(idx4)

        boxes, labels = boxes.numpy(), labels.numpy()
        boxes2, labels2 = boxes2.numpy(), labels2.numpy()
        boxes3, labels3 = boxes3.numpy(), labels3.numpy()
        boxes4, labels4 = boxes4.numpy(), labels4.numpy()

        h1, w1, channel = img.shape
        h2, w2, _ = img2.shape
        h3, w3, _ = img3.shape
        h4, w4, _ = img4.shape

        height = min((h1, h2, h3, h4))
        width = min((w1, w2, w3, w4))
        # height = max((h1, h2, h3, h4))
        # width = max((w1, w2, w3, w4))

        newImg = np.zeros((height, width, channel), img.dtype)
        cy, cx = height // 2, width // 2

        x = random.randint(cx * (1 - alpha), cx * (1 + alpha))
        y = random.randint(cy * (1 - alpha), cy * (1 + alpha))

        # 左上角
        y1 = random.randint(0, h1 - y)
        x1 = random.randint(0, w1 - x)
        newImg[0:y, 0:x] = img[y1:y1 + y, x1:x1 + x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1
        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 右上角
        y1 = random.randint(0, h2 - y)
        x1 = random.randint(0, w2 + x - width)
        newImg[0:y, x:width] = img2[y1:y1 + y, x1:x1 + width - x]
        tboxes = boxes2.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

        _boxes.extend(tboxes)
        _labels.extend(labels2)

        # 右下角
        y1 = random.randint(0, h3 + y - height)
        x1 = random.randint(0, w3 + x - width)
        newImg[y:height, x:width] = img3[y1:y1 + height - y, x1:x1 + width - x]
        tboxes = boxes3.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels3)

        # 左下角
        y1 = random.randint(0, h4 + y - height)
        x1 = random.randint(0, w4 - x)
        newImg[y:height, 0:x] = img4[y1:y1 + height - y, x1:x1 + x]
        tboxes = boxes4.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels4)

        target = {}

        target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
        target["labels"] = torch.tensor(_labels)

        target = filter(target, (height, width))
        if target is None:
            img, mask, boxes, labels, img_path = self.load(idx)
        else:
            img = newImg
            boxes = target["boxes"]
            labels = target["labels"]

        return img, mask, boxes, labels, img_path
    except:
        img, mask, boxes, labels, img_path = self.load(idx)
        return img, mask, boxes, labels, img_path


# 2张图片做 mosaic
def mosaicTwoImg(self, idx, alpha=0.5):
    try:
        _boxes = []
        _labels = []

        index = torch.randperm(self.len).tolist()
        if idx + 1 >= self.len:
            idx = 0

        idx2 = index[idx + 1]

        img, mask, boxes, labels, img_path = self.load(idx)
        img2, mask2, boxes2, labels2, _ = self.load(idx2)

        boxes, labels = boxes.numpy(), labels.numpy()
        boxes2, labels2 = boxes2.numpy(), labels2.numpy()

        h1, w1, channel = img.shape
        h2, w2, _ = img2.shape

        height = min((h1, h2))
        width = min((w1, w2))
        # height = max((h1, h2))
        # width = max((w1, w2))

        newImg = np.zeros((height, width, channel), img.dtype)
        cy, cx = height // 2, width // 2

        x = random.randint(cx * (1 - alpha), cx * (1 + alpha))
        y = random.randint(cy * (1 - alpha), cy * (1 + alpha))

        # 左上角
        y1 = random.randint(0, h1 - y)
        x1 = random.randint(0, w1 - x)
        newImg[0:y, 0:x] = img[y1:y1 + y, x1:x1 + x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1
        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 右上角
        y1 = random.randint(0, h2 - y)
        x1 = random.randint(0, w2 + x - width)
        newImg[0:y, x:width] = img2[y1:y1 + y, x1:x1 + width - x]
        tboxes = boxes2.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

        _boxes.extend(tboxes)
        _labels.extend(labels2)

        # 右下角
        y1 = random.randint(0, h1 + y - height)
        x1 = random.randint(0, w1 + x - width)
        newImg[y:height, x:width] = img[y1:y1 + height - y, x1:x1 + width - x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 左下角
        y1 = random.randint(0, h2 + y - height)
        x1 = random.randint(0, w2 - x)
        newImg[y:height, 0:x] = img2[y1:y1 + height - y, x1:x1 + x]
        tboxes = boxes2.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels2)

        target = {}

        target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
        target["labels"] = torch.tensor(_labels)

        target = filter(target, (height, width))
        if target is None:
            img, mask, boxes, labels, img_path = self.load(idx)
        else:
            img = newImg
            boxes = target["boxes"]
            labels = target["labels"]

        return img, mask, boxes, labels, img_path
    except Exception as e:
        print(e)
        img, mask, boxes, labels, img_path = self.load(idx)
        return img, mask, boxes, labels, img_path


# 1张图片做 mosaic
def mosaicOneImg(self, idx, alpha=0.5):
    try:
        _boxes = []
        _labels = []

        img, mask, boxes, labels, img_path = self.load(idx)

        boxes, labels = boxes.numpy(), labels.numpy()

        height, width, channel = img.shape

        newImg = np.zeros_like(img)
        cy, cx = height // 2, width // 2

        x = random.randint(cx * (1 - alpha), cx * (1 + alpha))
        y = random.randint(cy * (1 - alpha), cy * (1 + alpha))

        # 左上角
        y1 = random.randint(0, height - y)
        x1 = random.randint(0, width - x)
        newImg[0:y, 0:x] = img[y1:y1 + y, x1:x1 + x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1
        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 右上角
        y1 = random.randint(0, height - y)
        x1 = random.randint(0, x)
        newImg[0:y, x:width] = img[y1:y1 + y, x1:x1 + width - x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 右下角
        y1 = random.randint(0, y)
        x1 = random.randint(0, x)
        newImg[y:height, x:width] = img[y1:y1 + height - y, x1:x1 + width - x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 左下角
        y1 = random.randint(0, y)
        x1 = random.randint(0, width - x)
        newImg[y:height, 0:x] = img[y1:y1 + height - y, x1:x1 + x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels)

        target = {}

        target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
        target["labels"] = torch.tensor(_labels)

        target = filter(target, (height, width))
        if target is None:
            img, mask, boxes, labels, img_path = self.load(idx)
        else:
            img = newImg
            boxes = target["boxes"]
            labels = target["labels"]

        return img, mask, boxes, labels, img_path
    except Exception as e:
        print(e)
        img, mask, boxes, labels, img_path = self.load(idx)
        return img, mask, boxes, labels, img_path


class ResizeV2():
    """PIL.Image 方式做缩放"""

    def __init__(self, size=(416, 416)):
        self.size = size  # h,w

    def __call__(self, img, target):
        """img : PIL.Image"""
        # resize
        w, h = img.size
        img = img.resize(self.size[::-1])
        if "boxes" in target:
            boxes = target['boxes']
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * self.size[1] / w
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * self.size[0] / h
            target['boxes'] = boxes

        return img, target


class ResizeAndAlign():
    """
    1、最大边缩放到指定尺寸，最小边等比例缩放
    2、在右边或者下边填充，对齐到指定尺寸 （目标的比例保持不变，且填充不改变坐标）
    3、PAD 是两边填充  （目标的比例保持不变，但填充会改变坐标）
    """

    def __init__(self, size=(416, 416), value=0):  # 114
        self.size = size  # h,w
        self.value = value

    def __call__(self, img, target):
        img, target = ResizeFixMax(self.size[0])(img, target)

        # 填充对齐
        img = np.array(img)
        h, w, c = img.shape
        tmp = np.ones([*self.size, c], np.uint8)
        tmp[:h, :w] = img
        img = PIL.Image.fromarray(tmp)

        return img, target
