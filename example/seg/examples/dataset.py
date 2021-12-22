import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from glob import glob
import numpy as np
import random
import json

# from toolcv.api.define.utils.data.data import FruitsNutsDataset
from toolcv.tools.segment.augment import ext_transforms as et
from toolcv.tools.tools import polygons2mask, mask2segmentation
from toolcv.tools import transform as T

import torchvision.transforms as T2


class WaferDataset(Dataset):
    def __init__(self, root, is_train=False):
        self.image_paths = glob(os.path.join(root, "image", "*.jpg"))
        self.mask_paths = glob(os.path.join(root, "mask", "*.jpg"))
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert('RGB')
        mask = Image.open(self.mask_paths[item])  # .convert('GRAY')

        if self.is_train:
            img, mask = et.ExtRandomScale((0.5, 2.0))(img, mask)
            img, mask = et.ExtRandomCrop(size=(352, 640), pad_if_needed=True)(img, mask)

        # resize
        img = img.resize((640, 352))
        mask = mask.resize((640, 352), 0)
        img = np.array(img, np.float32)
        mask = np.array(mask, np.float32)
        # 镜像
        if self.is_train:
            if random.random() < 0.5:
                img = np.flip(img, 1)
                mask = np.flip(mask, 1)

        # to tensor
        img = torch.tensor(img.copy() / 255.).permute(2, 0, 1)
        mask = torch.tensor(mask.copy()).long()
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]

        return img, mask


class FruitsNutsDataset(Dataset):
    def __init__(self, root, classes, transforms=None, mode="none"):
        self.root = root
        self.classes = classes
        self.transforms = transforms
        self.annotations = self.change_csv()
        self.mode = mode

    def __len__(self):
        return len(self.annotations)

    def _load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = Image.open(img_path).convert("RGB")  # , np.uint8
        # boxes = annotations["boxes"]
        labels = annotations["labels"]

        masks = annotations["segment"]
        masks = np.array([polygons2mask(img.size[::-1], mask) for mask in masks])
        masks = masks * np.array(labels)[:, None, None]
        # mask2segmentation(masks)
        masks = Image.fromarray(np.max(masks, 0).astype(np.uint8))

        if self.transforms is not None:
            img, masks = self.transforms(img, masks)

        return img, masks

    def __getitem__(self, idx):
        if self.mode == "none":
            img, masks = self._load(idx)
        elif self.mode == "mixup":
            img, masks = T.mixup(self, idx, 'seg',len(self.classes))
        elif self.mode == "cutmix":
            img, masks = T.cutmix(self, idx, 'seg')
        elif self.mode == "mosaic":
            img, masks = T.mosaic(self, idx, 'seg')
        elif self.mode == "mosaicTwo":
            img, masks = T.mosaicTwo(self, idx, 'seg')
        elif self.mode == "mosaicOne":
            img, masks = T.mosaicOne(self, idx, 'seg')
        else:
            img, masks = self._load(idx)

        return img, masks

    def change_csv(self):
        with open(os.path.join(self.root, "trainval.json")) as f:
            imgs_anns = json.load(f)

        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        annotations = imgs_anns["annotations"]

        # 找到每张图片所以标注，组成一个list
        result = []
        for k, v in images_dict.items():
            img_path = os.path.join(self.root, "images", v)
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


train_transform = et.ExtCompose([
    # et.ExtResize(size=opts.crop_size),
    # et.ExtRandomScale((0.5, 2.0)),
    # et.ExtRandomCrop(size=(256, 256), pad_if_needed=True),
    # et.ExtResize((256, 256)),
    # T.RandomCropAndResize((256,256)),
    # T.RandomCropV3(), T.RandomPerspective(), T.RandomAffine(60),
    # T.ResizeRatio((256, 256)),
    # T.RandomScale(),
    T.ResizeRatiov2((416,416),mode='center'),
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(),
    # T.RandomErasing(),
    # T.CutOut(),
    # T.MosaicOne('seg')
])
val_transform = et.ExtCompose([
    # et.ExtResize((256, 256)),
    T.ResizeRatio((256, 256)),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def mask2rgb(mask, color):
        mask = np.array(mask)
        h, w = mask.shape
        color = np.array(color)
        rgb_img = np.ones([h, w, 3])
        for i in range(len(color)):
            rgb_img[mask == i] *= color[i]
        return rgb_img.astype(np.uint8)


    root = r"D:\data\fruitsNuts"
    classes = ["__background__", 'date', 'fig', 'hazelnut']
    color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    dataset = FruitsNutsDataset(root, classes, train_transform,"none")
    for img, mask in dataset:
        plt.subplot(1, 2, 1)
        img = ((img.permute(1, 2, 0) * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor(
            [[0.485, 0.456, 0.406]])) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        # plt.imshow((np.array(mask)*30).astype(np.uint8))
        plt.imshow(mask2rgb(mask.cpu().numpy(), color))
        # print(np.unique(mask))
        plt.show()
