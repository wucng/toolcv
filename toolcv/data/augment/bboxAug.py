"""
图片与bbox一起做数据增强，
如果不传入bbox 就是普通的只做图像增强
"""

import cv2
import numpy as np
import scipy.misc
import skimage
from skimage.transform import resize
import torch
import PIL.Image
from PIL import Image
import random

from torchvision.transforms import functional as F
from torch.nn import functional as F2
from torchvision.ops import misc as misc_nn_ops

try:
    from .uils import *
except:
    from uils import *

# ----------------------------------------------------------------------------------
class Augment(object):
    def __init__(self,advanced=False):
        self.advanced = advanced
    def __call__(self, image, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _image, _target=simple_agu(image.copy(),target.copy(),np.random.randint(0, int(1e5), 1)[0],self.advanced)
            if len(_target["boxes"])>0:
                # clip to image
                w,h=_image.size # PIL
                _target["boxes"][:,[0,2]] = _target["boxes"][:,[0,2]].clamp(min=0,max=w)
                _target["boxes"][:,[1,3]] = _target["boxes"][:,[1,3]].clamp(min=0,max=h)
                if (_target["boxes"][:,2:]-_target["boxes"][:,:2]>0).all():
                    image, target=_image, _target
            del _image
            del _target
        except:
            pass
        return image,target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image:Tensor
                target: Tensor
        """
        image = F.to_tensor(image) # 0~1
        return image, target

class Normalize(object):
    def __init__(self,image_mean=None,image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406] # RGB格式
        if image_std is None:
            image_std = [0.229, 0.224, 0.225] # ImageNet std
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
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.image_std, dtype=torch.float32, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image,target

class Pad(object):
    def __init__(self,mode='constant', value=128.):
        self.mode = mode
        self.value = value
    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        target["original_size"] = torch.as_tensor(img.shape[:2],dtype=torch.float32)
        img,target = self.pad_img(img,target)
        return img,target

    def pad_img(self, img,target):
        h, w = img.shape[:2]
        if "boxes" in target:
            boxes = target["boxes"]
        if h >= w:
            diff = h - w
            pad_list = [[0, 0], [diff // 2, diff - diff // 2], [0, 0]]
            if "boxes" in target:
                boxes = [[b[0] + diff // 2, b[1], b[2] + diff - diff // 2, b[3]] for b in boxes]
                boxes = torch.as_tensor(boxes,dtype=torch.float32)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints[...,0] += diff // 2 # x
                # keypoints[...,1] # y
                target["keypoints"] = keypoints


        else:
            diff = w - h
            pad_list = [[diff // 2, diff - diff // 2], [0, 0], [0, 0]]
            if "boxes" in target:
                boxes = [[b[0], b[1] + diff // 2, b[2], b[3] + diff - diff // 2] for b in boxes]
                boxes = torch.as_tensor(boxes,dtype=torch.float32)

            if "keypoints" in target:
                keypoints = target["keypoints"]
                # keypoints[...,0] = keypoints[...,0]+ diff // 2 # x
                keypoints[...,1] += diff // 2 # y
                target["keypoints"] = keypoints


        img = np.pad(img, pad_list, mode=self.mode, constant_values=self.value)
        if "masks" in target and target["masks"] is not None:
            masks = target["masks"].permute(1,2,0).cpu().numpy() # mask [c,h,w]格式
            masks = np.pad(masks, pad_list, mode=self.mode, constant_values=0)
            target["masks"] = torch.from_numpy(masks).permute(2,0,1)

        if "boxes" in target:
            target["boxes"] = boxes

        return PIL.Image.fromarray(img),target

class Resize(object):
    """
    适合先做pad（先按最长边填充成正方形），再做resize）
    也可以不pad，直接resize
    """
    def __init__(self,size=(224,224),multi_scale=False):
        self.size = size
        self.multi_scale = multi_scale
        if self.multi_scale: # 使用多尺度
            self.multi_scale_size = (32*np.arange(5,27,2)).tolist()

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        if self.multi_scale:
            choice_size = random.choice(self.multi_scale_size)
            self.size = (choice_size, choice_size)

        img = np.asarray(img)
        original_size = img.shape[:2]
        # target["original_size"] = torch.as_tensor(original_size)
        target["resize"] = torch.as_tensor(self.size,dtype=torch.float32)

        # img = scipy.misc.imresize(img,self.size,'bicubic') #  or 'cubic'
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)

        if "masks" in target and target["masks"] is not None:
            target["masks"] = misc_nn_ops.interpolate(target["masks"][None].float(), size=self.size,
                                                      mode="nearest")[0].byte()#.permute(1,2,0)


        if "boxes" in target:
            boxes = target["boxes"]

            boxes = resize_boxes(boxes,original_size,self.size)

            target["boxes"] = boxes

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,original_size,self.size)
            target["keypoints"] = keypoints

        return PIL.Image.fromarray(img), target


class ResizeAndPad(object):
    """先按比例resize，再pad"""
    def __init__(self,size=(224,224),multi_scale=False):
        self.size = size
        self.multi_scale = multi_scale
        if self.multi_scale:  # 使用多尺度
            self.multi_scale_size = (32 * np.arange(5, 27, 2)).tolist()

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        if self.multi_scale:
            choice_size = random.choice(self.multi_scale_size)
            self.size = (choice_size,choice_size)

        img = np.asarray(img)
        img_h, img_w = img.shape[:2]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)
        target["resize"] = torch.as_tensor(self.size,dtype=torch.float32)

        w, h = self.size
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))

        if new_w >= new_h:
            new_w = max(new_w, w)
        else:
            new_h = max(new_h, h)

        # img = scipy.misc.imresize(img, [new_h,new_w], 'bicubic')  # or 'cubic'
        img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        # pad
        img,target = Pad().pad_img(img,target)

        return img,target

class ResizeMinMax(object):
    """按最小边填充"""
    def __init__(self,min_size=600,max_size=1000): # 800,1333
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        img_h, img_w = img.shape[:2]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)

        # 按最小边填充
        min_size = min(img_w, img_h)
        max_size = max(img_w, img_h)
        scale = self.min_size/min_size
        if max_size*scale>self.max_size:
            scale = self.max_size /max_size

        new_w = int(scale * img_w)
        new_h = int(scale * img_h)

        target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)

        # img = scipy.misc.imresize(img, [new_h,new_w], 'bicubic')  # or 'cubic'
        img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        if "masks" in target and target["masks"] is not None:
            target["masks"] = misc_nn_ops.interpolate(target["masks"][None].float(),
                                                      size=(new_h,new_w), mode="nearest")[0].byte()#.permute(1,2,0)


        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,(img_h, img_w), (new_h,new_w))
            target["keypoints"] = keypoints


        return PIL.Image.fromarray(img),target


class ResizeFix(object):
    """按最小边填充"""
    def __init__(self,size=512):
        self.size = size

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        img_h, img_w = img.shape[:2]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)

        scale = self.size/max(img_w, img_h)

        new_w = int(scale * img_w)
        new_h = int(scale * img_h)

        target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)

        # img = scipy.misc.imresize(img, [new_h,new_w], 'bicubic')  # or 'cubic'
        img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        return PIL.Image.fromarray(img),target


class ResizeFixAndPad(object):
    """
    将长边填充到固定值，短边随机填充到32的倍数, Resize image to a 32-pixel-multiple rectangle
    """
    def __init__(self,new_shape=512,color=(114, 114, 114),auto=True, scaleFill=False, scaleup=True):
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup

    def __call__(self, img, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        target["original_size"] = torch.as_tensor(shape, dtype=torch.float32)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            if "boxes" in target:
                target["boxes"] = resize_boxes(target["boxes"], shape, new_unpad[::-1])
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        if "boxes" in target:
            target["boxes"] = target["boxes"] + torch.tensor([left, top, left, top]).unsqueeze(0)
        # return img, ratio, (dw, dh)

        target["resize"] = torch.as_tensor((new_unpad[1], new_unpad[0], left,top,right,bottom), dtype=torch.float32)

        return PIL.Image.fromarray(img), target


class RandomHSV(object):
    def __init__(self,p=0.5,hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        if random.random() < self.p:
            img = np.asarray(img,np.uint8)
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

            # Histogram equalization
            # if random.random() < 0.2:
            #     for i in range(3):
            #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])

            return PIL.Image.fromarray(img),target
        else:
            return img,target

class RandomCutout(object):
    """
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    """
    def __init__(self,p=0.5,thres_iou=0.6):
        self.p = p
        self.thres_iou = thres_iou

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """

        def bbox_ioa(box1, box2):
            # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
            box2 = box2.transpose()

            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

            # Intersection area
            inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                         (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

            # box2 area
            box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

            # Intersection over box2 area
            return inter_area / box2_area

        if random.random() < self.p:
            img = np.asarray(img,np.uint8)
            img2 = img.copy()
            img = img2
            h, w = img.shape[:2]

            # create random masks
            scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
            for s in scales:
                mask_h = random.randint(1, int(h * s))
                mask_w = random.randint(1, int(w * s))

                # box
                xmin = max(0, random.randint(0, w) - mask_w // 2)
                ymin = max(0, random.randint(0, h) - mask_h // 2)
                xmax = min(w, xmin + mask_w)
                ymax = min(h, ymin + mask_h)

                # apply random color mask
                img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

                gt_boxes = target["boxes"]
                labels = target["labels"]

                # return unobscured labels
                if len(labels) and s > 0.03:
                    box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                    ioa = bbox_ioa(box, gt_boxes.numpy())  # intersection over area
                    labels = labels[ioa < self.thres_iou]  # remove >60% obscured labels
                    gt_boxes = gt_boxes[ioa < self.thres_iou]
                    if len(labels)>0:
                        target["boxes"] = gt_boxes
                        target["labels"] = labels

            return PIL.Image.fromarray(img),target
        else:
            return img,target



class RandomDrop(object):
    def __init__(self,p=0.5,cropsize=(0.1,0.1)):
        """cropsize:从原图裁剪掉的像素值范围比例"""
        self.cropsize = cropsize
        self.p = p

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _img, _target = self.do(img.copy(),target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            img_h, img_w = img.shape[:2]
            y1,x1 = random.randint(0,int(self.cropsize[0]*img_h)),random.randint(0,int(self.cropsize[1]*img_w)) # 上面与左边裁剪的像素个数
            y2,x2 = random.randint(0,int(self.cropsize[0]*img_h)),random.randint(0,int(self.cropsize[1]*img_w)) # 下面与右边裁剪的像素个数

            # 裁剪后的图像
            img = img[y1:img_h-y2,x1:img_w-x2,:]
            new_h, new_w = img.shape[:2]

            # boxes也需做想要的裁剪处理
            boxes = target["boxes"]
            labels = target["labels"]
            new_boxes = []
            new_labels = []
            for i,b in enumerate(boxes):
                if b[2]-x1 <=0 or b[3]-y1 <=0: # box已经在裁剪图像的外
                    continue
                else:
                    new_boxes.append([max(0,b[0]-x1), max(0,b[1]-y1), min(new_w,b[2]-x1), min(new_h,b[3]-y1)])
                    new_labels.append(labels[i])

            new_boxes = torch.as_tensor(new_boxes,dtype=torch.float32)
            new_labels = torch.as_tensor(new_labels,dtype=torch.long)
            target["boxes"] = new_boxes
            target["labels"] = new_labels

            return PIL.Image.fromarray(img),target
        else:
            return img,target

class RandomCrop(object):
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            boxes = target["boxes"]  # .numpy()
            labels = target["labels"]  # .numpy()
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                target["boxes"] = boxes
                target["labels"] = labels
                img = PIL.Image.fromarray(bgr)
                return img, target
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]

            target["boxes"] = boxes_in
            target["labels"] = labels_in

            return PIL.Image.fromarray(img_croped),target
        else:
            return img,target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,target):
        return self.do(img, target)

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            height,width = img.shape[:2]
            img = img[:, ::-1, :]
            """
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img_center = torch.as_tensor(img_center,dtype=torch.float32)
            bboxes = target["boxes"]
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
            """
            bboxes = target["boxes"]
            bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]
            # """

            if not ((bboxes[:,2:]-bboxes[:,:2])>0).all():
                return PIL.Image.fromarray(img), target

            target["boxes"] = bboxes

            if "masks" in target and target["masks"] is not None:
                target["masks"] = target["masks"].flip(-1)  # [c,h,w]
                # target["masks"] = target["masks"].flip(1) # [h,w,c]


            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints

            return PIL.Image.fromarray(img), target
        else:
            return img,target

class RandomVerticallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img_center = torch.as_tensor(img_center,dtype=torch.float32)
            bboxes = target["boxes"]

            img = img[::-1, :, :]
            bboxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - bboxes[:, [1, 3]])

            box_h = abs(bboxes[:, 1] - bboxes[:, 3])

            bboxes[:, 1] -= box_h
            bboxes[:, 3] += box_h

            target["boxes"]=bboxes

            return PIL.Image.fromarray(img), target
        else:
            return img, target

class RandomScale(object):
    def __init__(self, p = 0.5,scale=0.2, diff=False):
        self.scale = scale
        self.p = p

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target

        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()
            img_shape = img.shape

            if self.diff:
                scale_x = random.uniform(*self.scale)
                scale_y = random.uniform(*self.scale)
            else:
                scale_x = random.uniform(*self.scale)
                scale_y = scale_x

            resize_scale_x = 1 + scale_x
            resize_scale_y = 1 + scale_y

            img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

            bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

            canvas = np.zeros(img_shape, dtype=np.uint8)

            y_lim = int(min(resize_scale_y, 1) * img_shape[0])
            x_lim = int(min(resize_scale_x, 1) * img_shape[1])

            canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

            img = canvas
            bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)
            img = PIL.Image.fromarray(img)
        return img, target

class RandomScale2(object):
    # #固定住高度，以0.8-1.2伸缩宽度，做图像形变
    def __init__(self, p = 0.5,scale=[0.8,1.2]):
        self.scale = random.uniform(*scale)
        self.p = p

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            height, width, c = img.shape
            boxes = target["boxes"]

            img = cv2.resize(img, (int(width * self.scale), height))
            scale_tensor = torch.FloatTensor([[self.scale, 1, self.scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor

            target["boxes"] = torch.as_tensor(boxes,dtype=torch.float32)
            img = PIL.Image.fromarray(img)

        return img, target

class RandomTranslate(object):
    """Randomly Translates the image """

    def __init__(self,p=0.5, translate=0.2, diff=False):
        self.translate = translate
        self.p = p

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random()<self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()
            img_shape = img.shape

            # translate the image

            # percentage of the dimension of the image to translate
            translate_factor_x = random.uniform(*self.translate)
            translate_factor_y = random.uniform(*self.translate)

            if not self.diff:
                translate_factor_y = translate_factor_x

            canvas = np.zeros(img_shape).astype(np.uint8)

            corner_x = int(translate_factor_x * img.shape[1])
            corner_y = int(translate_factor_y * img.shape[0])

            # change the origin to the top-left corner of the translated box
            orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                              min(img_shape[1], corner_x + img.shape[1])]

            mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
            canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
            img = canvas

            bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

            bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)

            return PIL.Image.fromarray(img), target
        else:
            return img,target


class RandomRotate(object):
    """Randomly rotates an image    """

    def __init__(self, p=0.5,angle=10):
        self.angle = angle
        self.p = p

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random()<self.p:
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()

            angle = random.uniform(*self.angle)

            w, h = img.shape[1], img.shape[0]
            cx, cy = w // 2, h // 2

            img = rotate_im(img, angle)

            corners = get_corners(bboxes)

            corners = np.hstack((corners, bboxes[:, 4:]))

            corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

            new_bbox = get_enclosing_box(corners)

            scale_factor_x = img.shape[1] / w

            scale_factor_y = img.shape[0] / h

            img = cv2.resize(img, (w, h))

            new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

            bboxes = new_bbox

            bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)

            return PIL.Image.fromarray(img), target
        else:
            return img,target


class RandomBrightness(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        return self.do(img, target)

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            hsv = BGR2HSV(img)
            h, s, v = cv2.split(hsv)
            adjust = random.choice(self.alpha)
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = HSV2BGR(hsv)
            img = PIL.Image.fromarray(img)

        return img, target


class RandomSaturation(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        return self.do(img, target)

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            hsv = BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice(self.alpha)
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = HSV2BGR(hsv)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomHue(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        return self.do(img, target)

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            hsv = BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = HSV2BGR(hsv)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomBlur(object):
    def __init__(self,p=0.5,kernel=(5,5)):
        self.p = p
        self.kernel = kernel

    def __call__(self,img,target):
        return self.do(img, target)

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            bgr = cv2.blur(bgr,self.kernel)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomShift(object):
    # 平移变换 （有问题）
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                # clip to image
                w, h = _img.size  # PIL
                _target["boxes"][:, [0, 2]] = _target["boxes"][:, [0, 2]].clamp(min=0, max=w)
                _target["boxes"][:, [1, 3]] = _target["boxes"][:, [1, 3]].clamp(min=0, max=h)
                if (_target["boxes"][:, 2:] - _target["boxes"][:, :2] > 0).all():
                    image, target = _image, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            boxes = target["boxes"]#.numpy()
            labels = target["labels"]#.numpy()
            center = (boxes[:, 2:] + boxes[:, :2]) / 2

            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                target["boxes"] = boxes
                target["labels"] = labels
                img = PIL.Image.fromarray(bgr)
                return img, target
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]

            target["target"] = boxes_in
            target["labels"] = labels_in
            img = PIL.Image.fromarray(after_shfit_image)

        return img, target


class ChannelMixer:
    """ Mix channels of multiple inputs in a single output image.
    This class works with opencv_ images (np.ndarray), and will mix the channels of multiple images into one new image.

    Args:
        num_channels (int, optional): The number of channels the output image will have; Default **3**

    Example:
        >>> # Replace the 3th channel of an image with a channel from another image
        >>> mixer = brambox.transforms.ChannelMixer()
        >>> mixer.set_channels([(0,0), (0,1), (1,0)])
        >>> out = mixer(img1, img2)
        >>> # out => opencv image with channels: [img0_channel0, img0_channel1, img1_channel0]
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        self.channels = [(0, i) for i in range(num_channels)]

    def set_channels(self, channels):
        """ Set from which channels the output image should be created.
        The channels list should have the same length as the number of output channels.

        Args:
            channels (list): List of tuples containing (img_number, channel_number)
        """
        if len(channels) != self.num_channels:
            raise ValueError('You should have one [image,channel] per output channel')
        self.channels = [(c[0], c[1]) for c in channels]

    def __call__(self, *imgs):
        """ Create and return output image.

        Args:
            *imgs: Argument list with all the images needed for the mix

        Warning:
            Make sure the images all have the same width and height before mixing them.
        """
        m = max(self.channels, key=lambda c: c[0])[0]
        if m >= len(imgs):
            raise ValueError('{} images are needed to perform the mix'.format(m))

        if isinstance(imgs[0], Image.Image):
            pil_image = True
            imgs = [np.array(img) for img in imgs]
        else:
            pil_image = False

        res = np.zeros([imgs[0].shape[0], imgs[0].shape[1], self.num_channels], 'uint8')
        for i in range(self.num_channels):
            if imgs[self.channels[i][0]].ndim >= 3:
                res[..., i] = imgs[self.channels[i][0]][..., self.channels[i][1]]
            else:
                res[..., i] = imgs[self.channels[i][0]]
        res = np.squeeze(res)

        if pil_image:
            return Image.fromarray(res)
        else:
            return res

# ----------------------------------------------------------------------------

class RandomChoice(object):
    def __init__(self):
        pass

    def __call__(self,img,target):
        choice = random.choice([
            RandomCrop(),
            RandomScale2(),
            RandomScale(),
            RandomDrop(cropsize=(0.05, 0.05)),
        ])

        return choice(img,target)