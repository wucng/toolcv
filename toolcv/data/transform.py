import numpy as np
import torch
import random
import cv2
from PIL import Image
import os
import json
import math
import time

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from toolcv.tools.utils import wh_iou_np, box_iou_np, x1y1x2y22xywh_np


def gaussianValue(x, y, cx, cy, alpha=3):
    return math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * alpha ** 2))


def gaussian_radius(det_size, min_overlap=0.7):
    """
    :param det_size: boxes的[h,w]，已经所放到heatmap上
    :param min_overlap:
    :return:
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    :param heatmap: [128,128]
    :param center: [x,y]
    :param radius: int
    :param k:
    :return:
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def filterByCenter(boxes, rect=[]):
    """
    丢弃任何中心不在裁剪图像中的框
    :param boxes: np.array [m,4] [x1,y1,x2,y2]
    :param rect: [x1,y1,x2,y2]
    :return:
    """
    x1, y1, x2, y2 = rect
    # 计算中心点
    ct = 0.5 * (boxes[:, 2:] + boxes[:, :2])
    keep = []
    for i, box in enumerate(ct):
        x, y = box
        if x > x1 and x < x2 and y > y1 and y < y2:
            keep.append(i)

    return keep


def mosaicFourImgV2(self, idx, alpha=0.5, angle=None, imagu=False, advanced=False):
    try:
        _boxes = []
        _labels = []

        index = torch.randperm(self.__len__()).tolist()
        if idx + 3 >= self.__len__():
            idx = 0

        idx2 = index[idx + 1]
        idx3 = index[idx + 2]
        idx4 = index[idx + 3]

        img1, annotations1 = self.load(idx)
        img2, annotations2 = self.load(idx2)
        img3, annotations3 = self.load(idx3)
        img4, annotations4 = self.load(idx4)

        # 每张图片随机做仿射变换(待完善)
        # ....
        if imagu:
            if np.random.random() < 0.5:
                img1, annotations1 = simple_agu(img1, annotations1, int(time.time()), advanced=advanced)
            if np.random.random() < 0.5:
                img2, annotations2 = simple_agu(img2, annotations2, int(time.time()), advanced=advanced)
            if np.random.random() < 0.5:
                img3, annotations3 = simple_agu(img3, annotations3, int(time.time()), advanced=advanced)
            if np.random.random() < 0.5:
                img4, annotations4 = simple_agu(img4, annotations4, int(time.time()), advanced=advanced)

        if not imagu and angle is not None:
            if np.random.random() < 0.5:
                img1, annotations1 = random_roate(img1, annotations1, angle)
            if np.random.random() < 0.5:
                img2, annotations2 = random_roate(img2, annotations2, angle)
            if np.random.random() < 0.5:
                img3, annotations3 = random_roate(img3, annotations3, angle)
            if np.random.random() < 0.5:
                img4, annotations4 = random_roate(img4, annotations4, angle)

        img1, img2, img3, img4 = np.array(img1), np.array(img2), np.array(img3), np.array(img4)
        boxes1, labels1 = annotations1[..., 1:], annotations1[..., 0]
        boxes2, labels2 = annotations2[..., 1:], annotations2[..., 0]
        boxes3, labels3 = annotations3[..., 1:], annotations3[..., 0]
        boxes4, labels4 = annotations4[..., 1:], annotations4[..., 0]

        h1, w1, channel = img1.shape
        h2, w2, _ = img2.shape
        h3, w3, _ = img3.shape
        h4, w4, _ = img4.shape

        # boxes 已经归一化到 0~1
        boxes1 *= np.array([[w1, h1, w1, h1]])
        boxes2 *= np.array([[w2, h2, w2, h2]])
        boxes3 *= np.array([[w3, h3, w3, h3]])
        boxes4 *= np.array([[w4, h4, w4, h4]])

        height = min((h1, h2, h3, h4))
        width = min((w1, w2, w3, w4))
        # height = max((h1, h2, h3, h4))
        # width = max((w1, w2, w3, w4))

        newImg = np.ones((height, width, channel), img1.dtype)  # * 114.0
        cy, cx = height // 2, width // 2

        # x = random.randint(cx * (1 - alpha), cx * (1 + alpha))
        # y = random.randint(cy * (1 - alpha), cy * (1 + alpha))
        x = random.randint(int(cx * (1 - alpha)), int(cx * (1 + alpha)))
        y = random.randint(int(cy * (1 - alpha)), int(cy * (1 + alpha)))

        # 左上角
        y1 = random.randint(0, h1 - y)
        x1 = random.randint(0, w1 - x)
        newImg[0:y, 0:x] = img1[y1:y1 + y, x1:x1 + x]
        tboxes = boxes1.copy()
        # 丢弃任何中心不在裁剪图像中的框
        keep = filterByCenter(tboxes, [x1, y1, x1 + x, y1 + y])
        if len(keep) > 0:
            keep = np.array(keep, np.int)
            tboxes = tboxes[keep]
            labels = labels1[keep]
            tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
            tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1
            _boxes.extend(tboxes)
            _labels.extend(labels)

        # 右上角
        y1 = random.randint(0, h2 - y)
        x1 = random.randint(0, w2 + x - width)
        newImg[0:y, x:width] = img2[y1:y1 + y, x1:x1 + width - x]
        tboxes = boxes2.copy()
        # 丢弃任何中心不在裁剪图像中的框
        keep = filterByCenter(tboxes, [x1, y1, x1 + width - x, y1 + y])
        if len(keep) > 0:
            keep = np.array(keep, np.int)
            tboxes = tboxes[keep]
            labels2 = labels2[keep]
            tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
            tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

            _boxes.extend(tboxes)
            _labels.extend(labels2)

        # 右下角
        y1 = random.randint(0, h3 + y - height)
        x1 = random.randint(0, w3 + x - width)
        newImg[y:height, x:width] = img3[y1:y1 + height - y, x1:x1 + width - x]
        tboxes = boxes3.copy()

        # 丢弃任何中心不在裁剪图像中的框
        keep = filterByCenter(tboxes, [x1, y1, x1 + width - x, y1 + height - y])
        if len(keep) > 0:
            keep = np.array(keep, np.int)
            tboxes = tboxes[keep]
            labels3 = labels3[keep]
            tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
            tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

            _boxes.extend(tboxes)
            _labels.extend(labels3)

        # 左下角
        y1 = random.randint(0, h4 + y - height)
        x1 = random.randint(0, w4 - x)
        newImg[y:height, 0:x] = img4[y1:y1 + height - y, x1:x1 + x]
        tboxes = boxes4.copy()
        # 丢弃任何中心不在裁剪图像中的框
        keep = filterByCenter(tboxes, [x1, y1, x1 + x, y1 + height - y])
        if len(keep) > 0:
            keep = np.array(keep, np.int)
            tboxes = tboxes[keep]
            labels4 = labels4[keep]
            tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
            tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

            _boxes.extend(tboxes)
            _labels.extend(labels4)

        if len(_boxes) == 0:
            # img, boxes, labels, img_path = self.load(idx)
            # return img, boxes, labels, img_path
            return 0
        else:
            # visual
            # for x1, y1, x2, y2 in _boxes:
            #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #     cv2.rectangle(newImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow('test', newImg[..., ::-1])
            # cv2.waitKey(0)

            img = Image.fromarray(newImg)
            # h, w = img.shape[:2]
            w, h = img.size
            # boxes = np.array(_boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
            boxes = np.array(_boxes, dtype=np.float32)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
            boxes /= np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
            labels = np.array(_labels, dtype=np.float32)
            annotations = np.concatenate((labels[..., None], boxes), -1)

            return img, annotations
    except Exception as e:
        print(e)
        # img,  boxes, labels, img_path = self.load(idx)
        # return img,  boxes, labels, img_path
        return 0


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


def roate(img, boxes, angle=90, fourpoints=False, ienet=False):
    """angle 是角度 不是弧度"""
    # radian = angle /180.0 * np.pi
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    affine_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 求得旋转矩阵
    img = cv2.warpAffine(img, affine_matrix, (w, h))

    tmp = []
    for box in boxes:
        tmp.append(roate_boxes(box, center, angle, fourpoints, ienet))

    if fourpoints:
        boxes = np.stack(tmp, 0)
        boxes[..., [0, 2, 4, 6]] = boxes[..., [0, 2, 4, 6]].clip(0, w - 1)
        boxes[..., [1, 3, 5, 7]] = boxes[..., [1, 3, 5, 7]].clip(0, h - 1)

        return img, boxes

    if ienet:
        boxes = np.array(tmp)
        boxes[..., [0, 2, 4]] = boxes[..., [0, 2, 4]].clip(0, w - 1)
        boxes[..., [1, 3, 5]] = boxes[..., [1, 3, 5]].clip(0, h - 1)
        return img, boxes

    boxes = np.array(tmp)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)

    return img, boxes


def random_roate(img, annotations, angle_range=30, fourpoints=False, ienet=False):
    img_, annotations_ = img.copy(), annotations.copy()

    try:
        # angle = random.randint(-angle_range, angle_range)
        angle = random.choice(range(-angle_range, angle_range + 1, 5))

        img = np.array(img)
        h, w = img.shape[:2]
        labels, boxes = annotations[..., 0], annotations[..., 1:]
        boxes *= np.array([[w, h, w, h]])  # 恢复到原图大小

        # 转成4点坐标 x1, y1, x2, y2, x3, y3, x4, y4
        boxes_ = []
        for x1, y1, x2, y2 in boxes:
            x3, y3 = x2, y2
            x2, y2 = x3, y1
            x4, y4 = x1, y3

            boxes_.append([x1, y1, x2, y2, x3, y3, x4, y4])

        img, boxes = roate(img, boxes_, angle, fourpoints, ienet)
        h, w = img.shape[:2]
        labels = np.array(labels, dtype=np.float32)
        if fourpoints:
            boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h, w, h, w, h]], dtype=np.float32)
        elif ienet:
            boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h, w, h]], dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)

        return Image.fromarray(img), annotations

    except:
        return img_, annotations_


# ------------------------------------------------------------------------------
def run_seq():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    # if not isinstance(images,list):
    #     images=[images]

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),
            # sometimes(iaa.Crop(percent=(0, 0.05))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # rotate=(-45, 45),
                rotate=(-5, 5),
                # shear=(-16, 16),
                shear=(-5, 5),
                order=[0, 1],
                # cval=(0, 255),
                cval=0  # 144,  # 填充像素值
                # mode=ia.ALL # 默认常数值填充边界
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    # images_aug = seq(images=images)

    return seq


def run_seq2():
    # if not isinstance(images,list):
    #     images=[images]
    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # iaa.Crop(percent=(0, 0.05)),
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(0.5,
                      iaa.Affine(
                          scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                          translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                          rotate=(-5, 5),
                          # rotate=(-25, 25),
                          shear=(-5, 5),
                          order=[0, 1],
                          cval=0  # 144,  # 填充像素值
                      ))

    ], random_order=True)  # apply augmenters in random order

    # images_aug = seq(images=images)

    return seq


def simple_agu(image, annotations, seed=100, advanced=False, shape=()):
    """
    :param image:PIL image
    :param labels: [[x1,y1,x2,y2,class_id],[]]
    :return: image:PIL image
    """
    image_, annotations_ = image.copy(), annotations.copy()
    try:
        ia.seed(seed)
        image = np.array(image)
        h, w = image.shape[:2]
        labels, boxes = annotations[..., 0], annotations[..., 1:]
        boxes *= np.array([[w, h, w, h]])  # 恢复到原图大小
        labels = [[*box, label] for box, label in zip(boxes, labels)]

        temp = [BoundingBox(*item[:-1], label=item[-1]) for item in labels]
        bbs = BoundingBoxesOnImage(temp, shape=image.shape)

        seq = run_seq() if advanced else run_seq2()

        # Augment BBs and images.
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()  # 处理图像外的边界框
        if shape:
            image_aug = ia.imresize_single_image(image_aug, shape)
            bbs_aug = bbs_aug.on(image_aug)

        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        # for i in range(len(bbs.bounding_boxes)):
        #         before = bbs.bounding_boxes[i]
        #         after = bbs_aug.bounding_boxes[i]
        #         print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        #             i,
        #             before.x1, before.y1, before.x2, before.y2,
        #             after.x1, after.y1, after.x2, after.y2)
        #         )

        # image with BBs before/after augmentation (shown below)
        """
        image_before = bbs.draw_on_image(image, size=2)
        image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

        skimage.io.imshow(image_before)
        skimage.io.show()

        skimage.io.imshow(image_after)
        skimage.io.show()
        # """

        # return image_aug, [[item.x1, item.y1, item.x2, item.y2, item.label] for item in bbs_aug.bounding_boxes] if last_class_id \
        #     else [[item.label,item.x1, item.y1, item.x2, item.y2] for item in bbs_aug.bounding_boxes]

        image_aug = Image.fromarray(image_aug)

        box = []
        label = []
        for item in bbs_aug.bounding_boxes:
            box.append([item.x1, item.y1, item.x2, item.y2])
            label.append(item.label)

        if len(label) == 0:
            return image_, annotations_

        boxes = np.array(box)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1
        labels = np.array(label)

        annotations = np.concatenate((labels[..., None], boxes), -1)

        return image_aug, annotations

    except:
        return image_, annotations_


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------
def getitem_fourpoints(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, angle=30,
                       imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=None, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                img, annotations = simple_agu(img, annotations, int(time.time()), advanced)

    # if np.random.random() < 0.5:
    img, annotations = random_roate(img, annotations, angle, True)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1
        annotations[..., [5, 7]] = 1.0 - annotations[..., [7, 5]]  # 已经缩放到0~1

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # conf,x1,y1,x2,y2,x3,y3,x4,y4,class_id....
    featureMap = np.zeros([fh, fw, self.num_anchors, 9 + self.num_classes], np.float32)

    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = annot[1:]  # (x1,y1,x2,y2) 0~1
        x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0: continue

        tfx = cx * fw
        tfy = cy * fh

        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        # dx = tfx - fx
        # dy = tfy - fy

        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = x1 * fw - fx, y1 * fh - fy, x2 * fw - fx, y2 * fh - fy, \
                                                 x3 * fw - fx, y3 * fh - fy, x4 * fw - fx, y4 * fh - fy

        # yolov1
        featureMap[fy, fx, :, :9] = [1, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
        featureMap[fy, fx, :, 9:] = np.eye(self.num_classes, self.num_classes)[labels]

        if mode == "centernet":
            # CenterNet
            radius = gaussian_radius((h * fw, w * fw))
            radius = max(1, int(radius))
            featureMap[..., 0, 0] = draw_umich_gaussian(featureMap[..., 0, 0], (fx, fy), radius)

        elif mode == "centernetv2":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        # dx = tfx - x
                        # dy = tfy - y
                        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = x1 * fw - x, y1 * fh - y, x2 * fw - x, y2 * fh - y, \
                                                                 x3 * fw - x, y3 * fh - y, x4 * fw - x, y4 * fh - y
                        featureMap[y, x, :, 1:9] = [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
                        featureMap[y, x, :, 9:] = np.eye(self.num_classes, self.num_classes)[labels]

        elif mode == "fcos":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        cenerness = cenerness ** 2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)

        elif mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        # dx = tfx - x
                        # dy = tfy - y
                        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = x1 * fw - x, y1 * fh - y, x2 * fw - x, y2 * fh - y, \
                                                                 x3 * fw - x, y3 * fh - y, x4 * fw - x, y4 * fh - y
                        featureMap[y, x, :, 1:9] = [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
                        featureMap[y, x, :, 9:] = np.eye(self.num_classes, self.num_classes)[labels]

    return img, featureMap


# 参考 IENet 的标注方式
def getitem_ienet(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, angle=30,
                  imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=None, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                img, annotations = simple_agu(img, annotations, int(time.time()), advanced)

    # if np.random.random() < 0.5:
    img, annotations = random_roate(img, annotations, angle, False, True)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [5, 6]] = annotations[..., [3, 4]] - annotations[..., [1, 2]] - annotations[..., [5, 6]]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # conf,x,y,w,h,w1,h1,class_id....
    featureMap = np.zeros([fh, fw, self.num_anchors, 7 + self.num_classes], np.float32)

    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2, w1, h1 = annot[1:]  # (x1,y1,x2,y2) 0~1
        x_min, x_max = x1, x2
        y_min, y_max = y1, y2
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0: continue

        tfx = cx * fw
        tfy = cy * fh

        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        dx = tfx - fx
        dy = tfy - fy

        # yolov1
        featureMap[fy, fx, :, :7] = [1, dx, dy, w, h, w1 / w, h1 / h]
        featureMap[fy, fx, :, 7:] = np.eye(self.num_classes, self.num_classes)[labels]

        if mode == "centernet":
            # CenterNet
            radius = gaussian_radius((h * fw, w * fw))
            radius = max(1, int(radius))
            featureMap[..., 0, 0] = draw_umich_gaussian(featureMap[..., 0, 0], (fx, fy), radius)

        elif mode == "centernetv2":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y

                        featureMap[y, x, :, 1:7] = [dx, dy, w, h, w1 / w, h1 / h]
                        featureMap[y, x, :, 7:] = np.eye(self.num_classes, self.num_classes)[labels]

        elif mode == "fcos":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        cenerness = cenerness ** 2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)

        elif mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y_min * fh)), int(np.floor(y_max * fh))
            x_min, x_max = int(np.ceil(x_min * fw)), int(np.floor(x_max * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y

                        featureMap[y, x, :, 1:7] = [dx, dy, w, h, w1 / w, h1 / h]
                        featureMap[y, x, :, 7:] = np.eye(self.num_classes, self.num_classes)[labels]

    return img, featureMap


# -----------------------------------------------------------------------------------------------
def getitem_fcos(self, idx, resize, fh, fw, mode='exp', use_mosaic=True, fix_resize=False, angle=None,
                 imagu=False, advanced=False):
    """
    # 原始的fcos方法
    :param self:
    :param idx:
    :param mode: 'sigmoid','exp' (推荐 exp)
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes],
                          np.float32)  # conf,x,y,w,h,class_id....
    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0: continue
        tfx = cx * fw
        tfy = cy * fh
        fx = int(tfx)  # 缩放到 featureMap上
        fy = int(tfy)

        y1_ = y1 * self.fh
        y2_ = y2 * self.fh
        x1_ = x1 * self.fw
        x2_ = x2 * self.fw
        l, r = fx - x1_, x2_ - fx
        t, b = fy - y1_, y2_ - fy
        if l > 0 and r > 0 and t > 0 and b > 0:
            cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
        else:
            cenerness = 1
            l, t, r, b = max(0, l), max(0, t), max(0, r), max(0, b)
        featureMap[fy, fx, 0, 0] = cenerness
        # if cenerness > 0.3:  # 0
        if mode == "sigmoid":
            featureMap[fy, fx, :, 1:5] = [l / self.fw, t / self.fh, r / self.fw,
                                          b / self.fh]  # 0~1 sigmoid()
        elif mode == "exp":
            featureMap[fy, fx, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
        else:
            raise ('error!!')

        featureMap[fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        y_min, y_max = int(np.ceil(y1_)), int(np.floor(y2_))
        x_min, x_max = int(np.ceil(x1_)), int(np.floor(x2_))
        if y_max > y_min and x_max > x_min:
            for y in range(y_min, y_max):
                t, b = y - y1_, y2_ - y
                for x in range(x_min, x_max):
                    l, r = x - x1_, x2_ - x
                    cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                    # cenerness = cenerness**2
                    # featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                    featureMap[y, x, 0, 0] = cenerness
                    # if cenerness >= 0.5:
                    # if cenerness > 0.3:
                    if mode == "sigmoid":
                        featureMap[y, x, :, 1:5] = [l / self.fw, t / self.fh, r / self.fw,
                                                    b / self.fh]  # 0~1 sigmoid()
                    elif mode == "exp":
                        featureMap[y, x, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
                    else:
                        raise ('error!!')

                    featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

    return img, featureMap


def getitem_mutilscale_fcos(self, idx, batch_size, strides, mode='exp', use_mosaic=True, fix_resize=False, angle=None,
                            imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    fw = int(resize / strides)
    fh = int(resize / strides)
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_fcos(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, angle, imagu,
                                           advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_fcos(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, angle, imagu,
                                           advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


# 多分支
def getitem_fcosMS(self, idx, resize, strides=[8, 16, 32], mode='exp-v1', use_mosaic=True, fix_resize=False, angle=None,
                   imagu=False, advanced=False):
    """ # v2效果差"""
    assert "-" in mode
    mode, versions = mode.split("-")
    if versions == "v1":
        return getitem_fcosMSV1(self, idx, resize, strides, mode, use_mosaic, fix_resize,
                                angle, imagu, advanced)
    else: # v2效果差
        return getitem_fcosMSV2(self, idx, resize, strides, mode, use_mosaic, fix_resize,
                                angle, imagu, advanced)


def getitem_fcosMSV1(self, idx, resize, strides=[8, 16, 32], mode='exp', use_mosaic=True, fix_resize=False, angle=None,
                     imagu=False, advanced=False):
    """
    # 原始的fcos方法
    :param self:
    :param idx:
    :param mode: 'sigmoid','exp' (推荐 exp)
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    featureMap_list = []
    for stride in strides:
        fh, fw = resize // stride, resize // stride

        featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes],
                              np.float32)  # conf,x,y,w,h,class_id....
        for annot in annotations:
            labels = int(annot[0])
            x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0: continue
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
            # if cenerness > 0.3:
            if mode == "sigmoid":
                featureMap[fy, fx, :, 1:5] = [l / fw, t / fh, r / fw,
                                              b / fh]  # 0~1 sigmoid()
            elif mode == "exp":
                featureMap[fy, fx, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
            else:
                raise ('error!!')

            featureMap[fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

            y_min, y_max = int(np.ceil(y1_)), int(np.floor(y2_))
            x_min, x_max = int(np.ceil(x1_)), int(np.floor(x2_))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y1_, y2_ - y
                    for x in range(x_min, x_max):
                        l, r = x - x1_, x2_ - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        # featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        featureMap[y, x, 0, 0] = cenerness
                        # if cenerness >= 0.5:
                        # if cenerness > 0.3:
                        if mode == "sigmoid":
                            featureMap[y, x, :, 1:5] = [l / fw, t / fh, r / fw,
                                                        b / fh]  # 0~1 sigmoid()
                        elif mode == "exp":
                            featureMap[y, x, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
                        else:
                            raise ('error!!')

                        featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        featureMap_list.append(featureMap.reshape(-1, 5 + self.num_classes))

    return img, np.concatenate(featureMap_list, 0)


def getitem_mutilscale_fcosMS(self, idx, batch_size, strides=[8, 16, 32], mode='exp-v1', use_mosaic=True,
                              fix_resize=False, angle=None, imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_fcosMS(self, idx, resize, strides, mode, use_mosaic, fix_resize, angle,
                                             imagu, advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_fcosMS(self, idx, resize, strides, mode, use_mosaic, fix_resize, angle,
                                             imagu, advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


def getitem_fcosMSV2(self, idx, resize, strides=[8, 16, 32], mode='exp', use_mosaic=True, fix_resize=False, angle=None,
                     imagu=False, advanced=False):
    """
    效果差
    # 原始的fcos方法
    :param self:
    :param idx:
    :param mode: 'sigmoid','exp'
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    featureMap_list = [np.zeros([resize // stride, resize // stride, self.num_anchors, 5 + self.num_classes],
                                np.float32) for stride in strides]

    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0: continue

        bid = 0
        bstride = 0
        # 选择一个最好的stride
        for id, stride in enumerate(strides):
            if w <= stride * 15 / 416 and h <= stride * 15 / 416:
                bid = id
                bstride = stride
                break

        fw = resize / bstride
        fh = resize / bstride

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
        featureMap_list[bid][fy, fx, 0, 0] = cenerness
        # if cenerness > 0.3:
        if mode == "sigmoid":
            featureMap_list[bid][fy, fx, :, 1:5] = [l / fw, t / fh, r / fw,
                                                    b / fh]  # 0~1 sigmoid()
        elif mode == "exp":
            featureMap_list[bid][fy, fx, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
        else:
            raise ('error!!')

        featureMap_list[bid][fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        y_min, y_max = int(np.ceil(y1_)), int(np.floor(y2_))
        x_min, x_max = int(np.ceil(x1_)), int(np.floor(x2_))
        if y_max > y_min and x_max > x_min:
            for y in range(y_min, y_max):
                t, b = y - y1_, y2_ - y
                for x in range(x_min, x_max):
                    l, r = x - x1_, x2_ - x
                    cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                    # cenerness = cenerness**2
                    # featureMap_list[bid][y, x, 0, 0] = max(featureMap_list[bid][y, x, 0, 0], cenerness)
                    featureMap_list[bid][y, x, 0, 0] = cenerness
                    # if cenerness >= 0.5:
                    # if cenerness > 0.3:
                    if mode == "sigmoid":
                        featureMap_list[bid][y, x, :, 1:5] = [l / fw, t / fh, r / fw,
                                                              b / fh]  # 0~1 sigmoid()
                    elif mode == "exp":
                        featureMap_list[bid][y, x, :, 1:5] = [l, t, r, b]  # 0~+无穷 exp()
                    else:
                        raise ('error!!')

                    featureMap_list[bid][y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

    # featureMap_list.append(featureMap.reshape(-1, 5 + self.num_classes))
    featureMap_list = [featureMap.reshape(-1, 5 + self.num_classes) for featureMap in featureMap_list]

    return img, np.concatenate(featureMap_list, 0)


# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------


def getitem(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, angle=None,
            imagu=False, advanced=False,box_norm="log"):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :return:
    """
    img, annotations = self.load(idx)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    # keep = (annotations[..., [3, 4]] - annotations[..., [1, 2]]).prod(-1) > 0
    # annotations = annotations[keep]

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes],
                          np.float32)  # conf,x,y,w,h,class_id....
    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0: continue

        if box_norm == "log":
            _w = w
            _h = h
            w = np.log(_w)
            h = np.log(_h)
        elif box_norm == "logv2":
            _w = w
            _h = h
            w = np.log(_w/(self.strides/resize))
            h = np.log(_h/(self.strides/resize))
        elif box_norm == "sqrt":
            _w = w
            _h = h
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
        featureMap[fy, fx, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        if mode == "centernet":
            # CenterNet
            radius = gaussian_radius((_h * fw, _w * fw))
            radius = max(1, int(radius))
            featureMap[..., 0, 0] = draw_umich_gaussian(featureMap[..., 0, 0], (fx, fy), radius)

        elif mode == "centernetv2":
            y_min, y_max = int(np.ceil(y1 * fh)), int(np.floor(y2 * fh))
            x_min, x_max = int(np.ceil(x1 * fw)), int(np.floor(x2 * fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y
                        featureMap[y, x, :, 1:5] = [dx, dy, w, h]
                        # featureMap[y, x, :, 1:5] = [cx,cy, w, h] # 效果较差
                        featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        elif mode == "fcos":
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
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)

        elif mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * self.fh)), int(np.floor(y2 * self.fh))
            x_min, x_max = int(np.ceil(x1 * self.fw)), int(np.floor(x2 * self.fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        featureMap[y, x, 0, 0] = max(featureMap[y, x, 0, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y
                        featureMap[y, x, :, 1:5] = [dx, dy, w, h]
                        # featureMap[y, x, :, 1:5] = [cx,cy, w, h] # 效果较差
                        featureMap[y, x, :, 5:] = np.eye(self.num_classes, self.num_classes)[labels]


    return img, featureMap


def getitem_mutilscale(self, idx, batch_size, strides, mode='fcos', use_mosaic=True, fix_resize=False, angle=None,
                       imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    fw = int(resize / strides)
    fh = int(resize / strides)
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, angle, imagu, advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, angle, imagu, advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


# ------------------------------------------------------------------------------------------------
# 使用先验anchor
def getitem_yolov2(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, anchor=[], angle=None,
                   imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchor: 5x2 [w,h] 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    anchor = np.array(anchor)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)
    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes],
                          np.float32)  # conf,x,y,w,h,class_id....
    for annot in annotations:
        labels = int(annot[0])
        x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0: continue
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
        featureMap[fy, fx, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        if mode == "centernet":
            # CenterNet
            radius = gaussian_radius((h * fw, w * fw))
            radius = max(1, int(radius))
            featureMap[..., max_idx, 0] = draw_umich_gaussian(featureMap[..., max_idx, 0], (fx, fy), radius)

        elif mode == "centernetv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * self.fh)), int(np.floor(y2 * self.fh))
            x_min, x_max = int(np.ceil(x1 * self.fw)), int(np.floor(x2 * self.fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        cenerness = gaussianValue(x, y, cx, cy)
                        featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y
                        # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                        featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                        featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        elif mode == "fcos":
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

        elif mode == "fcosv2":
            # Fcos
            y_min, y_max = int(np.ceil(y1 * self.fh)), int(np.floor(y2 * self.fh))
            x_min, x_max = int(np.ceil(x1 * self.fw)), int(np.floor(x2 * self.fw))
            if y_max > y_min and x_max > x_min:
                for y in range(y_min, y_max):
                    t, b = y - y_min, y_max - y
                    for x in range(x_min, x_max):
                        l, r = x - x_min, x_max - x
                        cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                        # cenerness = cenerness**2
                        featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                        # if cenerness > 0.3:
                        dx = tfx - x
                        dy = tfy - y
                        # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                        featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                        featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

    return img, featureMap


def getitem_mutilscale_yolov2(self, idx, batch_size, strides, mode='fcos', use_mosaic=True, fix_resize=False,
                              anchor=[], angle=None, imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    fw = int(resize / strides)
    fh = int(resize / strides)
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_yolov2(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor, angle,
                                             imagu, advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_yolov2(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor, angle,
                                             imagu, advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


# -----------------------------------------------------------------------------------------------
def getitem_yolov3(self, idx, resize, strides=[8, 16, 32], mode='fcos', use_mosaic=True, fix_resize=False, anchors=[],
                   angle=None, imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchors: [[],[],[]] w,h 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    anchors = np.array(anchors)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)
    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    # 按面积从大到小排序
    # area = (annotations[...,[3,4]] - annotations[...,[1,2]]).prod(-1)
    annotations = np.stack(sorted(annotations, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True), 0)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap_list = []
    for ii, stride in enumerate(strides):
        anchor = anchors[ii]
        fh, fw = resize // stride, resize // stride

        featureMap = np.zeros([fh, fw, self.num_anchors, 5 + self.num_classes],
                              np.float32)  # conf,x,y,w,h,class_id....
        for annot in annotations:
            labels = int(annot[0])
            x1, y1, x2, y2 = annot[1:]  # (x1,y1,x2,y2) 0~1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0: continue
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
            featureMap[fy, fx, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

            if mode == "centernet":
                # CenterNet
                radius = gaussian_radius((h * fw, w * fw))
                radius = max(1, int(radius))
                featureMap[..., max_idx, 0] = draw_umich_gaussian(featureMap[..., max_idx, 0], (fx, fy), radius)

            elif mode == "centernetv2":
                # Fcos
                y_min, y_max = int(np.ceil(y1 * self.fh)), int(np.floor(y2 * self.fh))
                x_min, x_max = int(np.ceil(x1 * self.fw)), int(np.floor(x2 * self.fw))
                if y_max > y_min and x_max > x_min:
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            cenerness = gaussianValue(x, y, cx, cy)
                            featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                            # if cenerness > 0.3:
                            dx = tfx - x
                            dy = tfy - y
                            # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                            featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                            featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

            elif mode == "fcos":
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

            elif mode == "fcosv2":
                # Fcos
                y_min, y_max = int(np.ceil(y1 * self.fh)), int(np.floor(y2 * self.fh))
                x_min, x_max = int(np.ceil(x1 * self.fw)), int(np.floor(x2 * self.fw))
                if y_max > y_min and x_max > x_min:
                    for y in range(y_min, y_max):
                        t, b = y - y_min, y_max - y
                        for x in range(x_min, x_max):
                            l, r = x - x_min, x_max - x
                            cenerness = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                            # cenerness = cenerness**2
                            featureMap[y, x, max_idx, 0] = max(featureMap[y, x, max_idx, 0], cenerness)
                            # if cenerness > 0:
                            dx = tfx - x
                            dy = tfy - y
                            # featureMap[y, x, max_idx, 1:5] = [dx, dy, w, h]
                            featureMap[y, x, max_idx, 1:5] = [dx, dy, np.log(w / a_w), np.log(h / a_h)]
                            featureMap[y, x, max_idx, 5:] = np.eye(self.num_classes, self.num_classes)[labels]

        featureMap_list.append(featureMap.reshape(-1, 5 + self.num_classes))

    return img, np.concatenate(featureMap_list, 0)


def getitem_mutilscale_yolov3(self, idx, batch_size, strides=[8, 16, 32], mode='fcos', use_mosaic=True,
                              fix_resize=False, anchors=[], angle=None, imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_yolov3(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors, angle,
                                             imagu, advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_yolov3(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors, angle,
                                             imagu, advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


# -------------ssd-----------------------------------------------------------------
def getitem_ssd(self, idx, resize, fh, fw, mode='v1', use_mosaic=True, fix_resize=False, anchor=[], angle=None,
                imagu=False, advanced=False):
    """v2 效果差"""
    # assert mode in ['v1', 'v2']
    if mode not in ['v1', 'v2']: mode = 'v1'
    if mode == "v1":
        return getitem_ssdV1(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor, angle,
                             imagu, advanced)
    else:
        return getitem_ssdV2(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor,
                             angle, imagu, advanced)


def getitem_ssdV1(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, anchor=[], angle=None,
                  imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    # anchor = np.array(anchor)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

    anchor_xywh = x1y1x2y22xywh_np(anchor)
    labels = annotations[..., 0]
    gt_boxes = annotations[..., 1:]
    gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

    keep = gt_boxes_xywh[..., 2:].prod(-1) > 0
    labels = labels[keep]
    gt_boxes = gt_boxes[keep]
    gt_boxes_xywh = gt_boxes_xywh[keep]

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

    return img, featureMap


def getitem_mutilscale_ssd(self, idx, batch_size, strides, mode='v1', use_mosaic=True, fix_resize=False,
                           anchor=[], angle=None, imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    fw = int(resize / strides)
    fh = int(resize / strides)
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_ssd(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor, angle, imagu,
                                          advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_ssd(self, idx, resize, fh, fw, mode, use_mosaic, fix_resize, anchor, angle, imagu,
                                          advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


def getitem_ssdV2(self, idx, resize, fh, fw, mode='fcos', use_mosaic=True, fix_resize=False, anchor=[], angle=None,
                  imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    # anchor = np.array(anchor)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

    anchor_xywh = x1y1x2y22xywh_np(anchor)
    labels = annotations[..., 0]
    gt_boxes = annotations[..., 1:]
    gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

    keep = gt_boxes_xywh[..., 2:].prod(-1) > 0
    labels = labels[keep]
    gt_boxes = gt_boxes[keep]
    gt_boxes_xywh = gt_boxes_xywh[keep]

    iou = box_iou_np(anchor, gt_boxes)

    iou_v2 = iou.max(1)
    iou_i2 = iou.argmax(1)
    # for i, v in enumerate(iou_v):
    #     if v >= 0.5:  # 正
    #         x, y, w, h = gt_boxes_xywh[iou_i[i]]
    #         a_x, a_y, a_w, a_h = anchor_xywh[i]
    #         featureMap[i, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
    #         featureMap[i, -1] = int(labels[iou_i[i]])
    #     elif v < 0.5:  # 负
    #         featureMap[i, -1] = 0

    # 与gt_boxes最大iou的anchor为正
    # iou_v = iou.max(0)
    iou_i = iou.argmax(0)
    for i, idx in enumerate(iou_i):
        idxs = np.nonzero(np.bitwise_and(iou_i2 == i, iou_v2 > 0.5))[0]  # 大于0.5 但非最大的 忽略 （参考yolov2,3）
        for _idx in idxs:
            if _idx != idx: featureMap[_idx, -1] = -1
        x, y, w, h = gt_boxes_xywh[i]
        a_x, a_y, a_w, a_h = anchor_xywh[idx]
        featureMap[idx, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
        featureMap[idx, -1] = int(labels[i])

    return img, featureMap


# ----------------------------------------------------------------------------------1
def getitem_ssdMS(self, idx, resize, strides=[8, 16, 32], mode='v1', use_mosaic=True, fix_resize=False, anchors=[],
                  angle=None, imagu=False, advanced=False):
    """v2 效果差"""
    if mode not in ['v1', 'v2']: mode = 'v1'
    # assert mode in ['v1', 'v2']
    if mode == "v1":
        return getitem_ssdMSV1(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors,
                               angle, imagu, advanced)
    else:
        return getitem_ssdMSV2(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors,
                               angle, imagu, advanced)


def getitem_ssdMSV1(self, idx, resize, strides=[8, 16, 32], mode='fcos', use_mosaic=True, fix_resize=False, anchors=[],
                    angle=None, imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    # anchors = np.array(anchors)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    _featureMap_list = []
    for i, stride in enumerate(strides):
        anchor = anchors[i]
        fh, fw = resize // stride, resize // stride
        featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

        anchor_xywh = x1y1x2y22xywh_np(anchor)
        labels = annotations[..., 0]
        gt_boxes = annotations[..., 1:]
        gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

        keep = gt_boxes_xywh[..., 2:].prod(-1) > 0
        labels = labels[keep]
        gt_boxes = gt_boxes[keep]
        gt_boxes_xywh = gt_boxes_xywh[keep]

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

    return img, np.concatenate(_featureMap_list, 0)


def getitem_mutilscale_ssdMS(self, idx, batch_size, strides=[8, 16, 32], mode='fcos', use_mosaic=True,
                             fix_resize=False, anchors=[], angle=None, imagu=False, advanced=False):
    # 多尺度训练
    size = np.random.choice([9, 11, 13, 15, 17, 19])
    resize = size * 32
    img_, featureMap_ = [], []
    for i in range(batch_size):
        if i == 0:
            img, featureMap = getitem_ssdMS(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors, angle,
                                            imagu, advanced)
        else:
            idx = np.random.randint(self.__len__())
            img, featureMap = getitem_ssdMS(self, idx, resize, strides, mode, use_mosaic, fix_resize, anchors, angle,
                                            imagu, advanced)
        img_.append(img)
        featureMap_.append(featureMap)

    img_ = torch.stack(img_, 0)
    featureMap_ = np.stack(featureMap_, 0)

    return img_, featureMap_


def getitem_ssdMSV2(self, idx, resize, strides=[8, 16, 32], mode='fcos', use_mosaic=True, fix_resize=False, anchors=[],
                    angle=None, imagu=False, advanced=False):
    """
    :param self:
    :param idx:
    :param mode: 'fcosv2';'fcos';'centernet';'yolov1'
    :param anchor: 9x4 [x1,y1,x2,y2] 缩放到0~1
    :return:
    """
    img, annotations = self.load(idx)
    # anchors = np.array(anchors)

    if use_mosaic:
        # mosaic数据增强方式
        if np.random.random() < 0.5:
            tmp = mosaicFourImgV2(self, idx, alpha=0.5, angle=angle, imagu=imagu, advanced=advanced)
            if tmp != 0:
                img, annotations = tmp
        else:
            if np.random.random() < 0.5:
                if imagu:
                    img, annotations = simple_agu(img, annotations, int(time.time()), advanced=advanced)
                elif not imagu and angle is not None:
                    img, annotations = random_roate(img, annotations, angle)

    # 随机镜像
    if np.random.random() < 0.5:
        # print("------fliplr---------")
        img = np.fliplr(img).astype(np.uint8)
        img = Image.fromarray(img)
        # h,w = img.shape[:2]
        annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    w, h = img.size
    annotations[..., [1, 3]] = annotations[..., [1, 3]].clip(0, 1.0 - 1.0 / w)
    annotations[..., [2, 4]] = annotations[..., [2, 4]].clip(0, 1.0 - 1.0 / h)

    if fix_resize:
        # resize
        img = img.resize((resize, resize))
    else:
        # resize(等比例)
        img = np.array(img)
        # print(img.shape)
        h, w = img.shape[:2]
        scale = resize / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h1, w1 = img.shape[:2]
        # print(img.shape)
        tmp = np.zeros([resize, resize, 3], np.uint8)
        tmp[:h1, :w1] = img
        img = Image.fromarray(tmp)

    img = self.transform(img)

    # 随机镜像
    # if np.random.random() < 0.5:
    #     img = torch.fliplr(img)
    #     annotations[..., [1, 3]] = 1.0 - annotations[..., [3, 1]]  # 已经缩放到0~1

    _featureMap_list = []
    for i, stride in enumerate(strides):
        anchor = anchors[i]
        fh, fw = resize // stride, resize // stride
        featureMap = -1 * np.ones([fh * fw * self.num_anchors, 4 + 1], np.float32)  # x,y,w,h,class_id....

        anchor_xywh = x1y1x2y22xywh_np(anchor)
        labels = annotations[..., 0]
        gt_boxes = annotations[..., 1:]
        gt_boxes_xywh = x1y1x2y22xywh_np(gt_boxes)

        keep = gt_boxes_xywh[..., 2:].prod(-1) > 0
        labels = labels[keep]
        gt_boxes = gt_boxes[keep]
        gt_boxes_xywh = gt_boxes_xywh[keep]

        iou = box_iou_np(anchor, gt_boxes)

        iou_v2 = iou.max(1)
        iou_i2 = iou.argmax(1)
        # for i, v in enumerate(iou_v):
        #     if v >= 0.5:  # 正
        #         x, y, w, h = gt_boxes_xywh[iou_i[i]]
        #         a_x, a_y, a_w, a_h = anchor_xywh[i]
        #         featureMap[i, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
        #         featureMap[i, -1] = int(labels[iou_i[i]])
        #     elif v < 0.5:  # 负
        #         featureMap[i, -1] = 0

        # 与gt_boxes最大iou的anchor为正
        # iou_v = iou.max(0)
        iou_i = iou.argmax(0)
        for i, idx in enumerate(iou_i):
            idxs = np.nonzero(np.bitwise_and(iou_i2 == i, iou_v2 > 0.5))[0]  # 大于0.5 但非最大的 忽略 （参考yolov2,3）
            for _idx in idxs:
                if _idx != idx: featureMap[_idx, -1] = -1
            x, y, w, h = gt_boxes_xywh[i]
            a_x, a_y, a_w, a_h = anchor_xywh[idx]
            featureMap[idx, :4] = [(x - a_x) / a_w, (y - a_y) / a_h, np.log(w / a_w), np.log(h / a_h)]  # [x,y,w,h]
            featureMap[idx, -1] = int(labels[i])

        _featureMap_list.append(featureMap)

    return img, np.concatenate(_featureMap_list, 0)


# -----------------------------------------------------------------------------
# example
from torch.utils.data import DataLoader, Dataset  # ,TensorDataset


# from torchvision.datasets import ImageFolder,DatasetFolder

class CustomDataset(Dataset):
    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        self.root = root
        self.transform = transform
        self.classes = classes
        self.resize = resize
        self.strides = strides

        self.muilscale = muilscale
        self.batch_size = batch_size
        self.mode = mode
        self.use_mosaic = use_mosaic
        self.fix_resize = fix_resize

        self.angle = angle
        self.imagu = imagu
        self.advanced = advanced

        self.num_anchors = num_anchors
        self.num_classes = len(classes)

    def __len__(self):
        raise ("please implement this method")

    def load(self, idx):
        raise ("please implement this method")

    def __getitem__(self, idx):
        raise ("please implement this method")

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels
