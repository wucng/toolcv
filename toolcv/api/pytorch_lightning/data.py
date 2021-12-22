import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset, Dataset
# from torchvision.datasets import ImageFolder,DatasetFolder
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pytorch_lightning as pl
import cv2
import random

from toolcv.tools.utils import glob_format
from toolcv.cls.data import load_mnist
from toolcv.data.augment.uils import mosaic_resize, mosaic_crop, mosaic_origin, mixup
from toolcv.data.augment.bboxAugv2 import mosaicFourImg
from toolcv.api.pytorch_lightning.transform import getitem_yolov1, getitem_yolov2, getitem_yolov3, getitem_ssd, \
    getitem_ssdMS, getitem_fcos


class _BaseDataset(Dataset):
    """mosaic_method=0 先做Mosaic 再做数据增强
       mosaic_method=1 先做数据增强 再做 Mosaic
    """

    def __init__(self, dir_data="", classes=[], transforms=None, useMosaic=False, mosaic_method=0,
                 filter_size=10):
        self.dir_data = dir_data
        self.classes = classes
        self.transforms = transforms
        self.filter_size = filter_size  # * self.resize / 224
        self.useMosaic = useMosaic
        self.mosaic_method = mosaic_method

    def __len__(self):
        raise ("error!")

    def _load(self, idx):
        """
        :param idx: 0~self.__len__()
        :return:
            img :PIL.Image
            masks :None
            boxes
            labels
            img_path
        """
        raise ("error!")

    def load(self, idx):
        """
        :param idx: 0~self.__len__()
        :return:
            img :PIL.Image
            masks :None
            boxes
            labels
            img_path
        """

        if self.mosaic_method == 0:
            return self._load(idx)
        else:
            img, masks, boxes, labels, img_path = self._load(idx)

            img = Image.fromarray(img)
            iscrowd = torch.zeros_like(labels, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["path"] = img_path

            if self.transforms is not None:
                while True:
                    try:
                        if self.multiscale:
                            self.transforms.target_size = self.resize
                            self.transforms.base_size = int(self.resize * 1.2)
                        _img, _target = self.transforms()(img.copy(), target.copy())
                        h, w = _img.shape[1:]
                        boxes = _target["boxes"]
                        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
                        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)
                        labels = _target["labels"]
                        if len(labels) == 0: raise Exception('error')
                        wh = boxes[..., 2:] - boxes[..., :2]
                        # keep = torch.bitwise_and(wh[..., 0] / wh[..., 1] < 8, wh[..., 1] / wh[..., 0] < 8)
                        # keep = torch.bitwise_and(keep, (wh > self.filter_size).prod(1) > 0)  # self.resize*0.05
                        keep = (wh > self.filter_size).prod(1) > 0

                        if sum(keep) == 0: raise Exception('error')
                        boxes = boxes[keep]
                        labels = labels[keep]
                        _target["boxes"] = boxes
                        _target["labels"] = labels
                        img, target = _img, _target
                        break
                    except Exception as e:
                        print("load", e)

            img = img.permute(1, 2, 0).cpu().numpy()

            return img, masks, boxes, labels, img_path

    def domosaic(self, idx, mosaic="origin"):
        try:
            if self.useMosaic:
                state = np.random.choice(["general", "mosaic", "mixup"])
                if state == "general":
                    img, masks, boxes, labels, img_path = self.load(idx)
                elif state == "mosaic":
                    if mosaic == "origin":
                        img, masks, boxes, labels, img_path = mosaic_origin(self, idx)
                    elif mosaic == "resize":
                        img, masks, boxes, labels, img_path = mosaic_resize(self, idx)
                    else:
                        # img, masks, boxes, labels, img_path = mosaic_crop(self, idx)
                        img, masks, boxes, labels, img_path = mosaicFourImg(self, idx)
                elif state == "mixup":
                    img, masks, boxes, labels, img_path = mixup(self, idx)
            else:
                img, masks, boxes, labels, img_path = self.load(idx)
        except Exception as e:
            print("domosaic", e)
            img, masks, boxes, labels, img_path = self.load(idx)

        return img, masks, boxes, labels, img_path

    def getitem0(self, idx):
        """ boxes并没有归一化 到 0~1 格式 [x1,y1,x2,y2]"""
        # img,masks, boxes, labels, img_path = self.load(idx)
        img, masks, boxes, labels, img_path = self.domosaic(idx)

        img = Image.fromarray(img)
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path

        if self.transforms is not None:
            while True:
                try:
                    if self.multiscale:
                        self.transforms.target_size = self.resize
                        self.transforms.base_size = int(self.resize * 1.2)
                    _img, _target = self.transforms()(img.copy(), target.copy())
                    h, w = _img.shape[1:]
                    boxes = _target["boxes"]
                    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
                    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)
                    labels = _target["labels"]
                    if len(labels) == 0: raise Exception('error')
                    wh = boxes[..., 2:] - boxes[..., :2]
                    # keep = torch.bitwise_and(wh[..., 0] / wh[..., 1] < 8, wh[..., 1] / wh[..., 0] < 8)
                    # keep = torch.bitwise_and(keep, (wh > self.filter_size).prod(1) > 0)  # self.resize*0.05
                    keep = (wh > self.filter_size).prod(1) > 0

                    if sum(keep) == 0: raise Exception('error')
                    boxes = boxes[keep]
                    labels = labels[keep]
                    _target["boxes"] = boxes
                    _target["labels"] = labels
                    img, target = _img, _target
                    break
                except Exception as e:
                    print("getitem0", e)

        return img, target

    def getitem1(self, idx):
        img, masks, boxes, labels, img_path = self.domosaic(idx, mosaic='crop')

        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path

        img = torch.tensor(img).permute(2, 0, 1)

        return img, target

    def getitem(self, idx):
        if self.mosaic_method == 0:
            return self.getitem0(idx)
        else:
            return self.getitem1(idx)

    def __getitem__(self, idx):
        return getitem

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels

    @staticmethod
    def collate_fnV2(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)

        return images, labels

    @staticmethod
    def collate_fn2(batch_data):
        data_list = []
        target_list = []
        for data, target in batch_data:
            data_list.append(data)
            target_list.append(target)

        return data_list, target_list

    def get_height_and_width(self, idx):
        img = np.array(self._load(idx)[0])
        return img.shape[:2]

    def show(self, nums=9, mode='cv2'):
        # for i in range(self.__len__()):
        idxs = np.random.choice(self.__len__(), nums).tolist()
        for idx in idxs:
            img, target = self.__getitem__(idx)
            # img = (img.permute(1, 2, 0) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
            img = ((img.permute(1, 2, 0) * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor(
                [[0.485, 0.456, 0.406]])) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
            boxes = target['boxes'].round().int().cpu().numpy()
            labels = target['labels'].cpu().numpy()

            # draw
            img = np.array(img.copy()[..., ::-1])  # RGB to BGR
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if mode == "cv2":
                cv2.imshow('test', img)
                cv2.waitKey(0)
            else:
                plt.imshow(img)
                plt.show()

        if mode == "cv2":
            cv2.destroyAllWindows()


class BaseDataset(_BaseDataset):
    """# for detecte"""

    def __init__(self, dir_data="", classes=[], transforms=None, useMosaic=False, mosaic_method=0,
                 filter_size=10, resize=416, strides=16, anchors=[], num_anchors=1, box_norm='log', mode='centernet',
                 method='yolov1', multiscale=False, batch_size=32, scales=[]):
        super().__init__(dir_data, classes, transforms, useMosaic, mosaic_method, filter_size)

        self.resize = resize
        self.strides = strides
        self.num_classes = len(classes)
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.box_norm = box_norm
        self.mode = mode
        self.method = method

        self.multiscale = multiscale
        self.batch_size = batch_size
        # if len(scales) == 0:
        #     scales = np.random.choice([9, 11, 13, 15, 17, 19]) * 32
        #     scales = int(max(min(np.random.normal(resize, resize // 4), 608), 288))  # 正态分布选择
        self.scales = scales
        self._resize = resize

    def __getitem__(self, idx):
        if self.multiscale:
            if len(self.scales) == 0:
                # resize = np.random.choice([9, 11, 13, 15, 17, 19]) * 32
                self.resize = int(max(min(np.random.normal(self._resize, self._resize // 4), 608), 288))  # 正态分布选择
            else:
                self.resize = np.random.choice(self.scales)
            imgs = []
            targets = []
            for i in range(self.batch_size):
                idx_ = idx * self.batch_size + i
                if idx_ >= self.len: continue
                img, target = self.dodetect(idx_)
                imgs.append(img)
                targets.append(target)

            img, target = torch.stack(imgs, 0), torch.stack(targets, 0)

            return img, target
        else:
            return self.dodetect(idx)

    def dodetect(self, idx):
        img, target = self.getitem(idx)

        if self.method == "yolov1":
            target = getitem_yolov1(self, target)
        elif self.method == "yolov2":
            target = getitem_yolov2(self, target)
        elif self.method == "yolov3":
            target = getitem_yolov3(self, target)
        elif self.method == "ssd":
            target = getitem_ssd(self, target)
        elif self.method == "ssdms":
            target = getitem_ssdMS(self, target)
        elif self.method == "fcos":
            target = getitem_fcos(self, target)

        if not isinstance(target, dict): target = torch.tensor(target)

        return img, target


class FruitsNutsDataset(BaseDataset):
    """
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, dir_data="", classes=[], transforms=None, useMosaic=False, mosaic_method=0, filter_size=10,
                 resize=416, strides=16, anchors=[], num_anchors=1, box_norm='log', mode='centernet', method='yolov1',
                 multiscale=False, batch_size=32, scales=[]):
        super().__init__(dir_data, classes, transforms, useMosaic, mosaic_method, filter_size, resize, strides, anchors,
                         num_anchors, box_norm, mode, method, multiscale, batch_size, scales)

        self.annotations = self.change_csv()

        self.len = len(self.annotations)

    def __len__(self):
        if self.multiscale:
            return max(self.len // self.batch_size, 1)
        else:
            return self.len

    def change_csv(self):
        json_file = os.path.join(self.dir_data, "trainval.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        annotations = imgs_anns["annotations"]

        # 找到每张图片所以标注，组成一个list
        result = []
        for k, v in images_dict.items():
            img_path = os.path.join(self.dir_data, "images", v)
            boxes = []
            labels = []
            iscrowd = []
            # image_id = []
            area = []
            segment = []
            for item in annotations:
                if item["image_id"] == k:
                    segment.append(item["segmentation"])
                    iscrowd.append(item["iscrowd"])
                    area.append(item["area"])
                    # boxes.append(item["bbox"])
                    bbox = item["bbox"]
                    # boxes.append([bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
                    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    labels.append(self.classes.index(categories_dict[item["category_id"]]))

            result.append({"img_path": img_path, "segment": segment, "iscrowd": iscrowd, "area": area, "boxes": boxes,
                           "labels": labels})

        return result

    def _load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        masks = None
        return img, masks, boxes, labels, img_path


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=32, num_workers=5,
                 use_cuda=None):
        super().__init__()
        self.batch_size = batch_size
        if use_cuda is None: use_cuda = torch.cuda.is_available()
        self.kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

        if train_dataset is not None:
            self.train_dataset = train_dataset
        if val_dataset is not None:
            self.val_dataset = val_dataset
        if test_dataset is not None:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)


class LitDataModule(BaseDataModule):
    def __init__(self, dir_path="./data", mode="mnist", batch_size=32):
        super().__init__(None, None, None, batch_size, 5, None)
        self.dir_path = dir_path
        self.mode = mode  # "fashion-mnist" or "mnist"

    def setup(self, stage):
        # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        train_images, train_labels, test_images, test_labels = load_mnist(self.dir_path, self.mode)
        train_images = torch.tensor(train_images / 255., dtype=torch.float32).unsqueeze(1)
        test_images = torch.tensor(test_images / 255., dtype=torch.float32).unsqueeze(1)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        if stage == 'fit' or stage is None:
            self.train_dataset = TensorDataset(train_images, train_labels)
            self.val_dataset = TensorDataset(test_images, test_labels)

        if stage == 'test' or stage is None:
            self.test_dataset = TensorDataset(test_images, test_labels)

    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass


def load_dataloader(dir_path=None, batch_size=32):
    if dir_path is None:
        dir_path = os.getcwd()
    # dataset = MNIST(dir_path, download=True, transform=transforms.ToTensor())
    # train, val = random_split(dataset, [55000, 5000])
    train = MNIST(dir_path, train=True, download=True, transform=transforms.ToTensor())
    val = MNIST(dir_path, train=False, download=True, transform=transforms.ToTensor())
    val, test = random_split(val, [9000, 1000])

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def smooth_label(label, num_classes):
    gamma = np.random.uniform(0.0, 0.4)
    I = np.eye(num_classes, num_classes)
    one_hot = I[label]

    return (one_hot - gamma) * one_hot + (1 - one_hot) * (gamma / (num_classes - 1))


class BaseDatasetCLS(Dataset):
    def __init__(self, dir_data, classes=[], transforms=None, mode="none", label_smooth=False):
        """mode ['mosaic','one_shot','triplet','none']"""
        assert mode in ['mosaic', 'one_shot', 'triplet', 'none']
        self.dir_data = dir_data
        self.classes = classes
        self.num_classes = len(classes)
        self.transforms = transforms
        self.mode = mode
        self.label_smooth = label_smooth

    def __len__(self):
        raise Exception("error")

    def __getitem__(self, idx):
        return self.domosaic(idx)

    def _load(self, idx):
        """
        :param idx:
        :return: img,target
        """
        raise Exception('error')

    def load(self, idx):
        img, target = self._load(idx)
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        img = img.permute(1, 2, 0).cpu().numpy()

        return img, target

    def domosaic(self, idx, alpha=0.9):
        if self.mode == "mosaic":
            state = np.random.choice(["general", "mosaic", "mixup"])
            if state == "general":
                img, target = self.load(idx)
                img = torch.tensor(img).permute(2, 0, 1)
                if self.label_smooth:
                    target = smooth_label(target, self.num_classes)
                return img, target
            elif state == "mosaic":
                img_list = []
                target_list = []
                img, target = self.load(idx)
                h, w = img.shape[:2]
                img_list.append(img)
                target_list.append(target)
                for _ in range(3):
                    while True:
                        # 循环直到一对图像不属于同一类别
                        idx = np.random.choice(self.__len__())
                        img, target = self.load(idx)
                        if target not in target_list:
                            img_list.append(img)
                            target_list.append(target)
                            break

                cx, cy = w // 2, h // 2
                while True:
                    x = int(np.random.randint(cx * (1 - alpha), cx * (1 + alpha)))
                    y = int(np.random.randint(cy * (1 - alpha), cy * (1 + alpha)))

                    # [:y,:x] [:y,x:],[y:,x:],[y:,:x]
                    area = np.array([x * y, (w - x) * y, (w - x) * (h - y), x * (h - y)]) / (w * h)
                    if max(area) > 0.6: break

                img = np.zeros([h, w, 3], np.float32)
                img[:y, :x] = img_list[0][:y, :x]
                img[:y, x:] = img_list[1][:y, x:]
                img[y:, x:] = img_list[2][y:, x:]
                img[y:, :x] = img_list[3][y:, :x]

                img = torch.tensor(img).permute(2, 0, 1)

                if self.label_smooth:
                    I = np.eye(self.num_classes, self.num_classes)
                    target = area[0] * I[target_list[0]] + area[1] * I[target_list[1]] + area[2] * I[target_list[2]] + \
                             area[3] * I[target_list[3]]

                    return img, target

                return img, np.array(target_list), area

            else:  # "mixup"
                gamma = np.random.uniform(0.1, 0.4)
                img, target = self.load(idx)
                while True:
                    # 循环直到一对图像不属于同一类别
                    idx = np.random.choice(self.__len__())
                    _img, _target = self.load(idx)
                    if _target != target:
                        break

                # img = cv2.addWeighted(img, gamma, _img, 1 - gamma, 0.0)
                img = img * gamma + _img * (1 - gamma)
                img = torch.tensor(img).permute(2, 0, 1)
                if self.label_smooth:
                    I = np.eye(self.num_classes, self.num_classes)
                    target = I[target] * gamma
                    _target = I[_target] * (1 - gamma)

                    target = target + _target

                    return img, target

                return img, target, _target, gamma

        elif self.mode == 'one_shot':
            img, target = self.load(idx)
            img = torch.tensor(img).permute(2, 0, 1)
            # 使得50%的训练数据为一对图像属于同一类别
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:  # 1
                while True:
                    # 循环直到一对图像属于同一类别
                    idx = np.random.choice(self.__len__())
                    _img, _target = self.load(idx)
                    if _target == target:
                        break
            else:
                while True:
                    # 循环直到一对图像属于不同的类别
                    idx = np.random.choice(self.__len__())
                    _img, _target = self.load(idx)
                    if _target != target:
                        break

            _img = torch.tensor(_img).permute(2, 0, 1)

            if self.label_smooth:
                target = smooth_label(target, self.num_classes)
                _target = smooth_label(_target, self.num_classes)

            return img, _img, target, _target, should_get_same_class  # 1 相同 ；0 不同

        elif self.mode == 'triplet':
            img, target = self.load(idx)
            img = torch.tensor(img).permute(2, 0, 1)
            while True:
                # 循环直到一对图像属于同一类别
                idx = np.random.choice(self.__len__())
                img1, target1 = self.load(idx)
                if target1 == target:
                    break
            while True:
                # 循环直到一对图像不属于同一类别
                idx = np.random.choice(self.__len__())
                img2, target2 = self.load(idx)
                if target2 != target:
                    break

            img1 = torch.tensor(img1).permute(2, 0, 1)
            img2 = torch.tensor(img2).permute(2, 0, 1)
            if self.label_smooth:
                target = smooth_label(target, self.num_classes)
                target1 = smooth_label(target1, self.num_classes)
                target2 = smooth_label(target2, self.num_classes)

            return img, img1, img2, target, target1, target2
        else:
            img, target = self.load(idx)
            img = torch.tensor(img).permute(2, 0, 1)
            if self.label_smooth:
                target = smooth_label(target, self.num_classes)

            return img, target

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels

    @staticmethod
    def collate_fn2(batch_data):
        data_list = []
        target_list = []
        for data, target in batch_data:
            data_list.append(data)
            target_list.append(target)

        return data_list, target_list

    def get_height_and_width(self, idx):
        img = np.array(self._load(idx)[0])
        return img.shape[:2]

    def show(self, nums=9, mode='cv2'):
        # for i in range(self.__len__()):
        idxs = np.random.choice(self.__len__(), nums).tolist()
        for idx in idxs:
            img, target = self.__getitem__(idx)
            # img = (img.permute(1, 2, 0) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
            img = ((img.permute(1, 2, 0) * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor(
                [[0.485, 0.456, 0.406]])) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
            labels = target

            # draw
            img = np.array(img.copy()[..., ::-1])  # RGB to BGR
            cv2.putText(img, str(labels), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if mode == "cv2":
                cv2.imshow('test', img)
                cv2.waitKey(0)
            else:
                plt.imshow(img)
                plt.show()

        if mode == "cv2":
            cv2.destroyAllWindows()


class FlowerPhotosDataset(BaseDatasetCLS):
    """
    5类鲜花数据集
    https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    """

    def __init__(self, dir_data, classes=[], transforms=None, mode="none", label_smooth=False):
        super().__init__(dir_data, classes, transforms, mode, label_smooth)

        self.paths = glob_format(dir_data)

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        path = self.paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(path)))
        image = np.array(Image.open(path).convert("RGB"))

        return image, label


if __name__ == "__main__":
    """
    import torchvision.transforms as T

    dataset = FlowerPhotosDataset(r"D:\data\flower_photos", ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'],
                                  T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                  mode='none', label_smooth=True)
    dataset.show(20)

    """
    from toolcv.api.pytorch_lightning.transform import Transforms
    from toolcv.data.augment import bboxAug, bboxAugv2

    # dm = LitDataModule()
    # dm.setup('fit')  # for train
    # dm.setup('test')  # for test
    dir_data = r"D:/data/fruitsNuts/"

    classes = ['date', 'fig', 'hazelnut']
    # dataset = FruitsNutsDataset(dir_data,classes,transforms=ToTensor())
    # transforms = get_transform(train=True, resize=(224, 224), useImgaug=True, advanced=False)
    resize = 224


    def trans(advanced=False, base_size=448, target_size=416):
        return bboxAug.Compose([
            bboxAug.RandomHorizontalFlip(),
            bboxAugv2.ResizeFixMinAndRandomCrop(base_size, (target_size, target_size)),
            # bboxAugv2.RandomRotate(),
            # bboxAugv2.RandomAffine(),
            # bboxAugv2.RandomDropPixelV2(),
            # bboxAugv2.RandomMosaic(),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])


    transforms = Transforms(trans, False, resize)
    filter_size = 15 * resize / 224
    dataset = FruitsNutsDataset(dir_data, classes, transforms, False, mosaic_method=0, filter_size=filter_size,
                                method='none', multiscale=False)
    dataset.show(15)
    # """
