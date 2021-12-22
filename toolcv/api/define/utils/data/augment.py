from torchvision import transforms as T
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
import random
import torch
from PIL import Image
import numpy as np
import math
import cv2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

__all__ = ['RandomHorizontalFlip', 'Resize', 'ResizeMax', 'Padding', "ResizeLimit", 'Crop', 'WarpAffine',
           'RandomBlur', 'RandomColorJitter', 'RandomNoise',
           'RandomChoice', 'Compose', 'ToTensor', 'Normalize',
           'mosaic', 'mosaicFourImg', 'mosaicTwoImg', 'mosaicOneImg',
           'batch'
           ]


class Resize():
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img, target):
        w, h = img.size
        img = img.resize((self.w, self.h))
        if target is not None:
            boxes = target['boxes']
            scale_w = self.w / w
            scale_h = self.h / h
            boxes *= torch.tensor([[scale_w, scale_h, scale_w, scale_h]], dtype=boxes.dtype, device=boxes.device)
            target['boxes'] = boxes

            if 'masks' in target:
                masks = target['masks']
                # if masks.ndim == 2: masks = masks[None]
                assert masks.ndim == 3
                masks = interpolate(masks[None], (self.h, self.w))[0]  # align_corners=False
                target['masks'] = masks

            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints[..., 0] *= scale_w
                keypoints[..., 1] *= scale_h
                target['keypoints'] = keypoints

        return img, target


class Padding:
    def __init__(self, size=(), mode='constant', value=0., pad_mode='both_sides'):
        """
        pad_mode='both_sides' 左右两边 或者 上下两边均匀填充
        pad_mode = "low_right" 右边或者下边填充
        """
        self.mode = mode
        self.value = value
        self.pad_mode = pad_mode
        self.size = size

    def __call__(self, img, target):
        img = np.asarray(img)
        h, w, c = img.shape
        max_v = max(w, h)
        if len(self.size) == 0:
            max_h = max_v
            max_w = max_v
        else:
            max_h, max_w = self.size
            assert max_h >= h and max_w >= w

        if self.pad_mode == 'both_sides':
            diff_w = max_w - w
            diff_h = max_h - h

            pad_list = [[diff_h // 2, diff_h - diff_h // 2], [diff_w // 2, diff_w - diff_w // 2], [0, 0]]
            img = np.pad(img, pad_list, mode=self.mode, constant_values=self.value)
            if target is not None:
                boxes = target['boxes']
                boxes += torch.tensor([[diff_w // 2, diff_h // 2, diff_w // 2, diff_h // 2]], dtype=boxes.dtype)
                target['boxes'] = boxes

                if 'masks' in target:
                    masks = target['masks']
                    # if masks.ndim == 2: masks = masks[None]
                    assert masks.ndim == 3
                    tmp_pad = torch.zeros([masks.size(0), max_h, max_w], dtype=masks.dtype)
                    tmp_pad[:, diff_h // 2:diff_h // 2 + h, diff_w // 2:diff_w // 2 + w] = masks
                    masks = tmp_pad
                    target['masks'] = masks

                if 'keypoints' in target:
                    keypoints = target['keypoints']
                    keypoints[..., 0] += diff_w // 2
                    keypoints[..., 1] += diff_h // 2
                    target['keypoints'] = keypoints

        else:
            tmp = np.zeros([max_h, max_w, c], img.dtype)
            tmp[:h, :w] = img
            img = tmp

            if 'masks' in target:
                masks = target['masks']
                # if masks.ndim == 2: masks = masks[None]
                assert masks.ndim == 3
                tmp_pad = torch.zeros([masks.size(0), max_h, max_w], dtype=masks.dtype)
                tmp_pad[:, :h, :w] = masks
                masks = tmp_pad
                target['masks'] = masks

        img = Image.fromarray(img)

        return img, target


class ResizeMax():
    """最大边resize到固定尺寸，最小边填充"""

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img, target):
        assert self.h == self.w
        w, h = img.size
        scale = self.h / max(w, h)
        new_w = min(math.ceil(w * scale), self.w)
        new_h = min(math.ceil(h * scale), self.h)

        img, target = Resize(new_h, new_w)(img, target)

        return img, target


class ResizeLimit:
    """
    最大边 不超过 1333  1000
    最小边 不超过 800   600
    """

    def __init__(self, min=800, max=1333):
        self.min = min
        self.max = max

    def __call__(self, img, target):
        w, h = img.size
        scale = self.min / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if max(new_w, new_h) > self.max:
            scale = self.max / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

        img, target = Resize(new_h, new_w)(img, target)

        return img, target


def batch(imgs, ratio=32):
    """
    imgs:[tensor,tensor,...]
    shape不一致，填充成一样大小，组成一个batch
    """
    h_list = []
    w_list = []
    for img in imgs:
        h, w = img.shape[-2:]
        h_list.append(h)
        w_list.append(w)
    max_w = max(w_list)
    max_h = max(h_list)

    # ratio整数倍
    new_w = math.ceil(max_w / ratio) * ratio
    new_h = math.ceil(max_h / ratio) * ratio

    tmp = torch.zeros([len(imgs), imgs[0].size(0), new_h, new_w], dtype=imgs[0].dtype, device=imgs[0].device)
    for i, img in enumerate(imgs):
        h, w = img.shape[-2:]
        tmp[i, :, :h, :w] = img

    return tmp


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            w, h = img.size
            img = F.hflip(img)
            boxes = target['boxes']
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes

            if 'masks' in target:
                masks = target['masks']
                # if masks.ndim == 2: masks = masks[None]
                assert masks.ndim == 3
                masks = torch.flip(masks, [2])
                target['masks'] = masks
            if 'keypoints' in target:
                keypoints = target['keypoints']  # [-1,17,3] 3:(x,y,v)
                keypoints[..., 0] = w - keypoints[..., 0]
                target['keypoints'] = keypoints

        return img, target


class ToTensor(object):
    def __call__(self, image, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image:Tensor
                target: Tensor
        """
        image = F.to_tensor(image)  # 0~1
        return image, target


class Normalize(object):
    def __init__(self, image_mean=None, image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # RGB格式
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]  # ImageNet std
            # image_std = [1.0, 1.0, 1.0]
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, image, target):
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
        image = T.Normalize(self.image_mean, self.image_std)(image)

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomChoice(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        t = random.choice(self.transforms)
        if isinstance(t, (tuple, list)): t = Compose(t)
        image, target = t(image, target)
        return image, target


class RandomErasing:
    """会改变框 未实现？？？"""

    def __init__(self):
        pass

    def __call__(self, image, target):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        image = T.RandomErasing()(image)
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))
        return image, target


# ------------------------------------------------
def _get_border(border, size):
    """Get the border size of the image"""
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


# 获得图片旋转以后的关键点的位置
def rotate_box(x, y, rot_mat):
    return (rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
            rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])


class CropWarpAffineResize:
    """
    随机裁剪 + 旋转 + 缩放 +仿射变换
    """

    def __init__(self, angle=5, scale=0.5, minpixel=15, resize=416):
        self.angle = angle
        self.scale = scale  # 缩放比例
        self.minpixel = minpixel  # 缩放比例
        self.resize = resize  # 缩放比例

    def __call__(self, img, target):
        img = np.array(img)
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        # while True:
        for _ in range(10):
            _img = img.copy()
            _boxes = boxes.copy()
            _labels = labels.copy()
            h, w, _ = img.shape
            sf = 0.3
            # crop
            while True:
                c = (random.uniform(sf, 1 - sf), random.uniform(sf, 1 - sf))  # 随机选择中心点
                cx = int(c[0] * w)
                cy = int(c[1] * h)

                x1 = cx - min(w - cx, cx)
                x2 = cx + min(w - cx, cx)
                y1 = cy - min(h - cy, cy)
                y2 = cy + min(h - cy, cy)

                # 裁剪
                _img = _img[y1:y2, x1:x2]
                h, w, _ = _img.shape

                # 过滤掉中心点不在范围内的
                # _boxes_cxcy = (_boxes[:, :2] + _boxes[:, 2:]) / 2
                # keep = []
                # for i, (cx, cy) in enumerate(_boxes_cxcy):
                #     if cx > x1 + minpixel and cx < x2 - minpixel and cy > y1 + minpixel and cy < y2 - minpixel:
                #         keep.append(i)

                keep = []
                _boxes2 = _boxes.copy()
                _boxes2[:, [0, 2]] = _boxes2[:, [0, 2]].clip(x1, x2)
                _boxes2[:, [1, 3]] = _boxes2[:, [1, 3]].clip(y1, y2)
                for i, ((_x1, _y1, _x2, _y2), (_x12, _y12, _x22, _y22)) in enumerate(zip(_boxes, _boxes2)):
                    if (_x22 - _x12) / (_x2 - _x1) < 0.3 or (_y22 - _y12) / (_y2 - _y1) < 0.3: continue
                    keep.append(i)

                if len(keep) > 0:
                    _boxes = _boxes[keep]
                    _labels = _labels[keep]
                    _boxes[:, [0, 2]] -= x1
                    _boxes[:, [1, 3]] -= y1

                    _boxes[:, [0, 2]] = _boxes[:, [0, 2]].clip(0, w - 1)
                    _boxes[:, [1, 3]] = _boxes[:, [1, 3]].clip(0, h - 1)
                    break

            # warpAffine
            while True:
                h, w, _ = _img.shape
                c = np.array([w / 2., h / 2.], dtype=np.float32)
                w_border = _get_border(128, w)
                h_border = _get_border(128, h)
                c[0] = np.random.randint(low=w_border, high=w - w_border)  # 裁剪后目标的中心点
                c[1] = np.random.randint(low=h_border, high=h - h_border)
                # a = random.randint(-10, 10)  # 裁剪后图片的旋转角度(旋转后 框有时不够准确)
                a = random.randint(-self.angle, self.angle)  # 裁剪后图片的旋转角度(旋转后 框有时不够准确)
                scale = random.uniform(1 - self.scale, 1 + self.scale)
                rot_mat = cv2.getRotationMatrix2D(tuple(c.astype(int)), a, scale)
                _img = cv2.warpAffine(_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

                new_boxes = []
                keep = []
                for i, box in enumerate(_boxes):
                    x1, y1, x2, y2 = box
                    x1, y1 = rotate_box(x1, y1, rot_mat)
                    x2, y2 = rotate_box(x2, y2, rot_mat)
                    if y2 - y1 > self.minpixel and x2 - x1 > self.minpixel:
                        new_boxes.append([x1, y1, x2, y2])
                        keep.append(i)
                if len(new_boxes) > 0:
                    _boxes = np.array(new_boxes)
                    _labels = _labels[keep]
                    h, w, _ = _img.shape
                    _boxes[:, [0, 2]] = _boxes[:, [0, 2]].clip(0, w - 1)
                    _boxes[:, [1, 3]] = _boxes[:, [1, 3]].clip(0, h - 1)

                    keep = ((_boxes[:, 2:] - _boxes[:, :2]) > self.minpixel).prod(-1) > 0
                    if keep.sum() > 0:
                        _boxes = _boxes[keep]
                        _labels = _labels[keep]
                        break

            # resize
            if random.randint(0, 1):
                h, w, _ = _img.shape
                _img = cv2.resize(_img, (self.resize, self.resize))
                _boxes *= np.array([[self.resize / w, self.resize / h, self.resize / w, self.resize / h]])
            else:
                # 等比例缩放 在填充
                h, w, _ = _img.shape
                scale = self.resize / max(w, h)
                _img = cv2.resize(_img, None, fx=scale, fy=scale)
                _boxes *= np.array([[scale, scale, scale, scale]])
                # pad
                h, w, _ = _img.shape
                py1, py2 = (self.resize - h) // 2, (self.resize - h) - (self.resize - h) // 2
                px1, px2 = (self.resize - w) // 2, (self.resize - w) - (self.resize - w) // 2
                _img = np.pad(_img, ((py1, py2), (px1, px2), (0, 0)), 'constant', constant_values=0)
                _boxes[:, [0, 2]] += px1
                _boxes[:, [1, 3]] += py1

            keep = ((_boxes[:, 2:] - _boxes[:, :2]) > self.minpixel).prod(-1) > 0
            if keep.sum() > 0:
                _boxes = _boxes[keep]
                _labels = _labels[keep]
                img, boxes, labels = _img, _boxes, _labels
                break

        img = Image.fromarray(img)
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = torch.from_numpy(labels)

        return img, target


class Crop:
    def __init__(self, sf=0.3, keep_ratio=0.3):
        self.sf = sf
        self.keep_ratio = keep_ratio

    def __call__(self, img, target):
        img = np.array(img)
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        h, w = img.shape[:2]

        # while True:
        for _ in range(10):
            _img = img.copy()
            _boxes = boxes.copy()
            _labels = labels.copy()
            c = (random.uniform(self.sf, 1 - self.sf), random.uniform(self.sf, 1 - self.sf))  # 随机选择中心点
            cx = int(c[0] * w)
            cy = int(c[1] * h)

            x1 = cx - min(w - cx, cx)
            x2 = cx + min(w - cx, cx)
            y1 = cy - min(h - cy, cy)
            y2 = cy + min(h - cy, cy)

            # 裁剪
            _img = _img[y1:y2, x1:x2]
            h, w, _ = _img.shape

            _boxes, _labels = keepByArea(_boxes, _labels, (x1, y1, x2, y2), self.keep_ratio)
            if len(_labels) > 0:
                img = _img
                boxes = _boxes
                labels = _labels
                break

        img = Image.fromarray(img)
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = torch.from_numpy(labels)

        return img, target


class WarpAffine:
    def __init__(self, angle=5, scale=0.5, minpixel=10, keep_ratio=0.3):
        self.angle = angle
        self.scale = scale  # 缩放比例
        self.minpixel = minpixel
        self.keep_ratio = keep_ratio

    def __call__(self, img, target):
        img = np.array(img)[..., ::-1]
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        h, w = img.shape[:2]

        # while True:
        for _ in range(10):
            _img = img.copy()
            _boxes = boxes.copy()
            _labels = labels.copy()

            c = np.array([w / 2., h / 2.], dtype=np.float32)
            w_border = _get_border(128, w)
            h_border = _get_border(128, h)
            c[0] = np.random.randint(low=w_border, high=w - w_border)  # 裁剪后目标的中心点
            c[1] = np.random.randint(low=h_border, high=h - h_border)
            # a = random.randint(-10, 10)  # 裁剪后图片的旋转角度(旋转后 框有时不够准确)
            a = random.randint(-self.angle, self.angle)  # 裁剪后图片的旋转角度(旋转后 框有时不够准确)
            scale = random.uniform(1 - self.scale, 1 + self.scale)
            rot_mat = cv2.getRotationMatrix2D(tuple(c.astype(int)), a, scale)
            _img = cv2.warpAffine(_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

            new_boxes = []
            keep = []
            for i, box in enumerate(_boxes):
                x1, y1, x2, y2 = box
                _x1, _y1 = rotate_box(x1, y1, rot_mat)
                _x2, _y2 = rotate_box(x2, y2, rot_mat)
                if (_y2 - _y1) / (y2 - y1) < self.keep_ratio or (_x2 - _x1) / (x2 - x1) < self.keep_ratio: continue
                new_boxes.append([_x1, _y1, _x2, _y2])
                keep.append(i)
            if len(new_boxes) > 0:
                _boxes = np.array(new_boxes)
                _labels = _labels[keep]
                h, w, _ = _img.shape
                _boxes[:, [0, 2]] = _boxes[:, [0, 2]].clip(0, w - 1)
                _boxes[:, [1, 3]] = _boxes[:, [1, 3]].clip(0, h - 1)

                keep = ((_boxes[:, 2:] - _boxes[:, :2]) > self.minpixel).prod(-1) > 0
                if keep.sum() > 0:
                    _boxes = _boxes[keep]
                    _labels = _labels[keep]

                img = _img
                boxes = _boxes
                labels = _labels
                break

        img = Image.fromarray(img[..., ::-1])
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = torch.from_numpy(labels)

        return img, target


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def mosaic(self, idx, h, w):
    # img, boxes, labels = self._load(idx)
    # w, h = img.size
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]

    len_d = self.__len__()
    while True:
        idx_ = np.random.randint(0, len_d)
        if idx_ != idx: break

    # annotation_line = [idx, (idx + 1) % len_d, (idx + 2) % len_d, (idx + 3) % len_d]
    annotation_line = [idx, idx_, (idx_ + 1) % len_d, (idx_ + 2) % len_d]

    # 统计高宽比与宽高比
    # h_w = []
    w_h = []
    for i, idx in enumerate(annotation_line):
        img, target = self._load(idx)

        image, target = self.transforms_mosaic(img, target)

        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()

        wh = boxes[..., 2:] - boxes[..., :2]
        w_h.extend(wh[..., 0] / wh[..., 1])
        # h_w.extend(wh[..., 1] / wh[..., 0])

        box = np.concatenate((boxes, labels[..., None]), -1)

        # 图片的大小
        iw, ih = image.size

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
            # image.show()
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh))

        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        # print(dx)
        dy = place_y[index]
        # print(dy)
        # new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] >= w] = w - 1
            box[:, 3][box[:, 3] >= h] = h - 1
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # >>> np.logical_and([True, False], [False, False])
            # array([False, False], dtype=bool)
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            # box_data = np.zeros((len(box), 5))
            # box_data[:len(box)] = box
            box_data = box

        image_datas.append(image_data)
        box_datas.append(box_data)

    # # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    img = Image.fromarray(new_image.astype(np.uint8))
    new_boxes = torch.tensor(new_boxes)

    # 按宽高比过滤
    w_h = torch.tensor(w_h)
    # h_w = torch.tensor(h_w)
    wh = new_boxes[..., 2:4] - new_boxes[..., :2]
    _w_h = wh[..., 0] / wh[..., 1]
    keep = torch.bitwise_and(_w_h < 1.3 * w_h.max(), _w_h > 0.7 * w_h.min())
    new_boxes = new_boxes[keep]

    target = {"boxes": new_boxes[..., :4], 'labels': new_boxes[..., 4].long()}

    return img, target


class RandomHSV:
    def __init__(self, hue=.1, sat=1.5, val=1.5):
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, img, target):
        # 进行色域变换
        hue = rand(-self.hue, self.hue)
        sat = rand(1, self.sat) if rand() < .5 else 1 / rand(1, self.sat)
        val = rand(1, self.val) if rand() < .5 else 1 / rand(1, self.val)
        x = rgb_to_hsv(np.array(img) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        img = hsv_to_rgb(x)

        img = Image.fromarray((img * 255).astype(np.uint8))

        return img, target


'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''


def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''


class RandomHSVCV:
    def __init__(self, hue_vari=10, sat_vari=0.1, val_vari=0.1):
        self.hue_vari = hue_vari
        self.sat_vari = sat_vari
        self.val_vari = val_vari

    def __call__(self, img, target):
        img = np.array(img)[..., ::-1]
        if random.random() > 0.5:
            hue_delta = np.random.randint(-self.hue_vari, self.hue_vari)
            sat_mult = 1 + np.random.uniform(-self.sat_vari, self.sat_vari)
            val_mult = 1 + np.random.uniform(-self.val_vari, self.val_vari)
            img = hsv_transform(img, hue_delta, sat_mult, val_mult)
        img = Image.fromarray(img[..., ::-1])

        return img, target


class RandomColor:
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, img, target):
        img = np.array(img)
        if random.random() > 0.5:
            b, g, r = random.uniform(1 - self.scale, 1 + self.scale), \
                      random.uniform(1 - self.scale, 1 + self.scale), \
                      random.uniform(1 - self.scale, 1 + self.scale)
            c = random.uniform(10, 150)
            img = (c + img * np.array([[b, g, r]])).clip(0, 255).astype(np.uint8)
            # img = (img*g).clip(0,255).astype(np.uint8)
        img = Image.fromarray(img)
        return img, target


class RandomColorJitter:
    """包含了 RandomColor，RandomHSVCV，RandomHSV"""

    def __init__(self, brightness=[0.3, 0.6], contrast=[0.3, 0.6], saturation=[0.3, 0.6], hue=[0, 0.5]):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, target):
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        saturation = random.uniform(*self.saturation)
        hue = random.uniform(*self.hue)
        img = T.ColorJitter(brightness, contrast, saturation, hue)(img)

        return img, target


class RandomBlur:
    def __init__(self, ksize=(5, 5)):
        self.ksize = ksize

    def dcall__(self, img, target):
        img = np.array(img)[..., ::-1]
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, self.ksize, 1)
        img = Image.fromarray(img[..., ::-1])
        return img, target

    def __call__(self, img, target):
        if random.random() > 0.5:
            img = T.GaussianBlur(self.ksize[0])(img)
        return img, target


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out


class RandomNoise:
    def __init__(self):
        pass

    def __call__(self, img, target):
        img = np.array(img)
        if random.random() > 0.5:
            if random.randint(0, 1):
                img = gasuss_noise(img, 0, random.uniform(0.01, 0.03))
            else:
                img = sp_noise(img, random.uniform(0.05, 0.1))

        img = Image.fromarray(img)
        return img, target


"""过滤掉很小的框"""


def filterBySize(target, imgSize=(), minhw=3):
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


# 中心点过滤

# 按裁剪前后长度比例过滤
def keepByArea(boxes, labels, x1y1x2y2, keep_ratio=0.3, mosaic=False):
    x1, y1, x2, y2 = x1y1x2y2
    keep = []
    _boxes = boxes.copy()
    _boxes[:, [0, 2]] = _boxes[:, [0, 2]].clip(x1, x2)
    _boxes[:, [1, 3]] = _boxes[:, [1, 3]].clip(y1, y2)
    for j, ((_x1, _y1, _x2, _y2), (_x12, _y12, _x22, _y22)) in enumerate(zip(boxes, _boxes)):
        # 裁剪前后的宽高 筛选
        if (_x22 - _x12) / (_x2 - _x1) < keep_ratio or (_y22 - _y12) / (_y2 - _y1) < keep_ratio: continue
        keep.append(j)

    if len(keep) > 0:
        if mosaic:
            boxes = _boxes[keep]
            labels = labels[keep]
        else:
            h, w = y2 - y1, x2 - x1
            boxes = boxes[keep]
            labels = labels[keep]
            boxes[:, [0, 2]] -= x1
            boxes[:, [1, 3]] -= y1

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    else:
        boxes = []
        labels = []

    return boxes, labels


# 4张图片做 mosaic
def mosaicFourImg(self, idx, height=416, width=416, alpha=0.7, keep_ratio=0.3):
    while True:
        boxes_list = []
        labels_list = []

        len_d = self.__len__()
        while True:
            idx_ = np.random.randint(0, len_d)
            if idx_ != idx: break

        # annotation_line = [idx, (idx + 1) % len_d, (idx + 2) % len_d, (idx + 3) % len_d]
        annotation_line = [idx, idx_, (idx_ + 1) % len_d, (idx_ + 2) % len_d]

        newImg = np.zeros((height, width, 3), np.uint8)
        cy, cx = height // 2, width // 2

        dx = random.randint(int(cx * (1 - alpha)), int(cx * (1 + alpha)))
        dy = random.randint(int(cy * (1 - alpha)), int(cy * (1 + alpha)))

        x1y1x2y2 = [[0, 0, dx, dy], [dx, 0, width - 1, dy], [0, dy, dx, height - 1], [dx, dy, width - 1, height - 1]]

        for i, idx in enumerate(annotation_line):
            img, target = self._load(idx)

            # 已经resize 到 height, width
            image, target = self.transforms_mosaic(img, target)
            image = np.array(image)
            assert image.shape[:2] == (height, width)
            x1, y1, x2, y2 = x1y1x2y2[i]
            newImg[y1:y2, x1:x2] = image[y1:y2, x1:x2]

            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()

            boxes, labels = keepByArea(boxes, labels, (x1, y1, x2, y2), keep_ratio, True)

            boxes_list.extend(boxes)
            labels_list.extend(labels)

        if len(labels_list) > 0:
            boxes = torch.from_numpy(np.stack(boxes_list, 0))
            labels = torch.from_numpy(np.stack(labels_list, 0))
            target = {"boxes": boxes, "labels": labels}
            img = Image.fromarray(newImg)
            return img, target


# 2张图片做 mosaic
def mosaicTwoImg(self, idx, height=416, width=416, alpha=0.7, keep_ratio=0.3):
    while True:
        boxes_list = []
        labels_list = []

        len_d = self.__len__()
        while True:
            idx_ = np.random.randint(0, len_d)
            if idx_ != idx: break

        # annotation_line = [idx, (idx + 1) % len_d, (idx + 2) % len_d, (idx + 3) % len_d]
        annotation_line = [idx, idx_, idx, idx_]

        newImg = np.zeros((height, width, 3), np.uint8)
        cy, cx = height // 2, width // 2

        dx = random.randint(int(cx * (1 - alpha)), int(cx * (1 + alpha)))
        dy = random.randint(int(cy * (1 - alpha)), int(cy * (1 + alpha)))

        x1y1x2y2 = [[0, 0, dx, dy], [dx, 0, width - 1, dy], [0, dy, dx, height - 1], [dx, dy, width - 1, height - 1]]

        for i, idx in enumerate(annotation_line):
            img, target = self._load(idx)

            # 已经resize 到 height, width
            image, target = self.transforms_mosaic(img, target)
            image = np.array(image)
            assert image.shape[:2] == (height, width)
            x1, y1, x2, y2 = x1y1x2y2[i]
            newImg[y1:y2, x1:x2] = image[y1:y2, x1:x2]

            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()

            boxes, labels = keepByArea(boxes, labels, (x1, y1, x2, y2), keep_ratio, True)

            boxes_list.extend(boxes)
            labels_list.extend(labels)

        if len(labels_list) > 0:
            boxes = torch.from_numpy(np.stack(boxes_list, 0))
            labels = torch.from_numpy(np.stack(labels_list, 0))
            target = {"boxes": boxes, "labels": labels}
            img = Image.fromarray(newImg)
            return img, target


# 1张图片做 mosaic
def mosaicOneImg(self, idx, height=416, width=416, alpha=0.7, keep_ratio=0.3):
    while True:
        boxes_list = []
        labels_list = []

        # len_d = self.__len__()
        annotation_line = [idx, idx, idx, idx]

        newImg = np.zeros((height, width, 3), np.uint8)
        cy, cx = height // 2, width // 2

        dx = random.randint(int(cx * (1 - alpha)), int(cx * (1 + alpha)))
        dy = random.randint(int(cy * (1 - alpha)), int(cy * (1 + alpha)))

        x1y1x2y2 = [[0, 0, dx, dy], [dx, 0, width - 1, dy], [0, dy, dx, height - 1], [dx, dy, width - 1, height - 1]]

        for i, idx in enumerate(annotation_line):
            img, target = self._load(idx)

            # 已经resize 到 height, width
            image, target = self.transforms_mosaic(img, target)
            image = np.array(image)
            assert image.shape[:2] == (height, width)
            x1, y1, x2, y2 = x1y1x2y2[i]
            newImg[y1:y2, x1:x2] = image[y1:y2, x1:x2]

            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()

            boxes, labels = keepByArea(boxes, labels, (x1, y1, x2, y2), keep_ratio, True)

            boxes_list.extend(boxes)
            labels_list.extend(labels)

        if len(labels_list) > 0:
            boxes = torch.from_numpy(np.stack(boxes_list, 0))
            labels = torch.from_numpy(np.stack(labels_list, 0))
            target = {"boxes": boxes, "labels": labels}
            img = Image.fromarray(newImg)
            return img, target
