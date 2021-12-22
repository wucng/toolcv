import json
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
from tqdm import tqdm
import pickle

from toolcv.api.define.utils.data.augment import mosaic, mosaicFourImg, mosaicTwoImg, mosaicOneImg
from toolcv.api.define.utils.data import augment as aug
import toolcv.data.augment.bboxAugv2 as baug
from toolcv.tools.tools import polygons2mask, mask2polygons, compute_area_from_polygons
from toolcv.tools.vis import draw_rect, draw_mask, draw_segms, draw_keypoint, draw_keypointV2


def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data, target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list, target_list


class _BaseDataset(Dataset):
    @staticmethod
    def collate_fn(batch_data):
        data_list = []
        target_list = []
        for data, target in batch_data:
            data_list.append(data)
            target_list.append(target)

        return data_list, target_list

    def get_height_and_width(self, idx):
        img = np.array(self._load(idx)[0])
        return img.shape[:2]

    def show(self, nums=9, mode='cv2', draw='draw_mask', save_path=None):
        if draw == 'draw_rect':
            draw = draw_rect
        elif draw == 'draw_mask':
            draw = draw_mask
        elif draw == 'draw_segms':
            draw = draw_segms
        elif draw == 'draw_keypoint':
            draw = draw_keypointV2
        else:
            raise ('error!!')
        if nums > 0:
            idxs = np.random.choice(self.__len__(), nums).tolist()
        else:
            idxs = range(self.__len__())
        for idx in tqdm(idxs):
            img, target = self.__getitem__(idx)
            # img = (img.permute(1, 2, 0) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)
            img = ((img.permute(1, 2, 0) * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor(
                [[0.485, 0.456, 0.406]])) * 255.).clamp(0, 255).round().int().cpu().numpy().astype(np.uint8)

            """
            boxes = target['boxes'].round().int().cpu().numpy()
            labels = target['labels'].cpu().numpy()

            # draw
            img = np.array(img.copy()[..., ::-1])  # RGB to BGR
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            """
            img = draw(img, target)

            if save_path is not None:
                if not os.path.exists(save_path): os.makedirs(save_path)
                fitter = os.listdir('D:\data\coco\minicoco\images')

                jpg_name = target.pop('img_path').split('\\')[-1]
                if jpg_name not in fitter: continue

                cv2.imwrite(os.path.join(save_path, jpg_name), img)
                # pickle.dump(target, open(os.path.join(save_path, "%d.pkl" % idx), 'wb'))

                target['boxes'] = target['boxes'].cpu().numpy().tolist()
                target['labels'] = target['labels'].cpu().numpy().tolist()
                target['area'] = target['area'].cpu().numpy().tolist()
                target['iscrowd'] = target['iscrowd'].cpu().numpy().tolist()
                target['keypoints'] = target['keypoints'].cpu().numpy().tolist()
                # target['segm'] = target['segm']

                json.dump(target, open(os.path.join(save_path, jpg_name.replace('.jpg', '.json')), 'w'), indent=4)

            else:
                if mode == "cv2":
                    cv2.imshow('test', img[..., ::-1])
                    cv2.waitKey(0)
                else:
                    plt.imshow(img)
                    plt.show()

        if mode == "cv2" and save_path is None:
            cv2.destroyAllWindows()


class FruitsNutsDataset(_BaseDataset):
    """
    数据 是 mscoco 格式
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], transforms=None, max_samples=0, use_mosaic=False, transforms_mosaic=None,
                 h=416, w=416, mosaic_mode=0, do_clsify=False, expand_pixel=10, masks=False):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.max_samples = max_samples
        self.use_mosaic = use_mosaic
        self.transforms_mosaic = transforms_mosaic
        self.h = h
        self.w = w
        self.mosaic_mode = mosaic_mode
        self.do_clsify = do_clsify
        self.expand_pixel = expand_pixel
        self.masks = masks

        self.annotations = self.change_csv()

    def __len__(self):
        return len(self.annotations)

    def _load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = Image.open(img_path).convert("RGB")  # , np.uint8
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"labels": labels, 'boxes': boxes}
        if self.masks:
            masks = annotations["segment"]
            masks = torch.tensor([polygons2mask(img.size[::-1], mask) for mask in masks])
            target.update({'masks': masks})

        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        if 'masks' not in target:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([compute_area_from_polygons(mask2polygons(mask.cpu().numpy())) for mask in masks])

        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

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

    def __getitem__00(self, idx):
        if self.use_mosaic:
            if self.mosaic_mode == 0:
                img, target = mosaic(self, idx, self.h, self.w)
            elif self.mosaic_mode == 1:
                img, target = mosaicFourImg(self, idx, self.h, self.w)
            elif self.mosaic_mode == 2:
                img, target = mosaicTwoImg(self, idx, self.h, self.w)
            elif self.mosaic_mode == 3:
                img, target = mosaicOneImg(self, idx, self.h, self.w)
            else:
                raise ('error!!!!')
        else:
            img, target = self._load(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.max_samples > 0:
            bboxes = torch.cat((target["boxes"], target["labels"][..., None]), -1)
            tmp = -1 * torch.ones([self.max_samples, 5], dtype=bboxes.dtype, device=bboxes.device)
            min_len = min(self.max_samples, len(bboxes))
            tmp[:min_len] = bboxes[:min_len]
            bboxes = tmp

            return img, bboxes

        if self.do_clsify:
            # baug.Patch((self.h, self.w))
            c, img_h, img_w = img.shape
            img_list, boxes_list, labels_list = [], [], []
            boxes, labels = target["boxes"], target["labels"]
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.numpy()
                # 往外扩展几个像素 增加一定背景
                exp_x, exp_y = self.expand_pixel, self.expand_pixel
                # exp_x = math.floor((x2 - x1) / 2)
                # exp_y = math.floor((y2 - y1) / 2)
                x1 = max(math.floor(x1) - exp_x, 0)
                y1 = max(math.floor(y1) - exp_y, 0)
                x2 = min(math.ceil(x2) + exp_x, img_w - 1)
                y2 = min(math.ceil(y2) + exp_y, img_h - 1)
                crop = img[:, y1:y2, x1:x2]
                _, h, w = crop.shape

                x2 = x2 - x1 - exp_x
                y2 = y2 - y1 - exp_y
                x1 = exp_x
                y1 = exp_y

                # resizeMax
                size = np.random.choice(np.arange(min(img_h, img_w) // 2, min(img_h, img_w), 10))
                scale = size / max(w, h)
                new_w = min(math.ceil(w * scale), size)
                new_h = min(math.ceil(h * scale), size)
                crop = F.interpolate(crop[None], (new_h, new_w))[0]
                _, h, w = crop.shape
                x1 = x1 * scale
                y1 = y1 * scale
                x2 = x2 * scale
                y2 = y2 * scale
                # 随机patch
                tmp = torch.zeros_like(img)
                x = np.random.choice(np.arange(0, img_w - w)) if img_w - w > 0 else 0
                y = np.random.choice(np.arange(0, img_h - h)) if img_h - h > 0 else 0
                tmp[:, y:y + h, x:x + w] = crop

                img_list.append(tmp)
                boxes_list.append([x1 + x, y1 + y, x2 + x, y2 + y])
                labels_list.append(label)

            # img = torch.stack(img_list, 0)[0]
            # boxes = torch.tensor(boxes_list).float()[[0]]
            # labels = torch.tensor(labels_list)[[0]]

            img = torch.stack(img_list, 0)
            boxes = torch.tensor(boxes_list).float()
            labels = torch.tensor(labels_list)

            target = {'boxes': boxes, 'labels': labels}


        img, target = T.mixup(self,idx,'det')

        return img, target

    def __getitem__(self, idx):

        img, target = self._load(idx)
        # img, target = T.mixup(self,idx,'det')
        # img, target = T.cutmix(self,idx,'det')
        # img,target = T.mosaic(self,idx,'det')
        # img,target = T.mosaicTwo(self,idx,'det')
        # img,target = T.mosaicOne(self,idx,'det')

        return img, target


class LoadDataloader:
    def __init__(self, train_dataset, val_dataset=None, val_ratio=0.1, batch_size=32,
                 kwargs={'num_workers': 4, 'pin_memory': True}, collate_fn=collate_fn):
        # kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

        if 'collate_fn' in train_dataset.__dict__:
            self.collate_fn = train_dataset.collate_fn
        else:
            self.collate_fn = collate_fn

        if val_dataset is not None and val_ratio > 0:
            train_ratio = 1 - val_ratio
            num_datas = len(train_dataset)
            num_train = int(train_ratio * num_datas)
            indices = torch.randperm(num_datas).tolist()
            train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
            val_dataset = torch.utils.data.Subset(val_dataset, indices[num_train:])
        elif val_dataset is None and val_ratio > 0:
            nums = len(train_dataset)
            nums_val = int(nums * val_ratio)
            train_dataset, val_dataset = random_split(train_dataset, [nums - nums_val, nums_val])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.kwargs = kwargs

    def train_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_dataLoader = DataLoader(self.train_dataset, batch_size, True,
                                      collate_fn=self.collate_fn, **self.kwargs)
        return train_dataLoader

    def val_dataloader(self, batch_size=1):
        val_dataLoader = DataLoader(self.val_dataset, batch_size, False, collate_fn=self.collate_fn,  # self.batch_size
                                    **self.kwargs)

        return val_dataLoader


keypoints_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


class MScocoKeypointsDataset(_BaseDataset):
    """
    https://blog.csdn.net/u014734886/article/details/78830713
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    """

    def __init__(self, root="", classes=['person'], transforms=None):
        self.root = root
        self.transforms = transforms
        self.classes = classes

        self.annotations = self.parse()
        self.keys = list(self.annotations.keys())

    def parse(self):
        jdatas = json.load(open(os.path.join(self.root, 'annotations', 'person_keypoints_val2017.json')))
        jimages = jdatas['images']  # id
        jannotations = jdatas['annotations']  # image_id,category_id
        jcategories = jdatas['categories']  # id

        images = {}
        for item in tqdm(jimages):  # "id" 唯一
            images[item["id"]] = {"img_path": os.path.join(self.root, 'val2017', item["file_name"]),
                                  "height": item["height"], "width": item["width"]}

        categories = {}
        for item in tqdm(jcategories):  # "id" 唯一
            categories[item["id"]] = {"supercategory": item["supercategory"], "name": item["name"],
                                      "keypoints": item["keypoints"]}

        annotations = {}
        # 注意存在一张图片有多个注释（需要把属于同一张图的不同注释 合并到一起，以每张图片为单元）
        for item in tqdm(jannotations):  # "id" 唯一 但"image_id" 不唯一；category_id 也不一定唯一 除非一个类
            segm = item['segmentation']
            area = item['area']
            iscrowd = item['iscrowd']
            keypoints = item['keypoints']
            image_id = item['image_id']
            bbox = item['bbox']
            category_id = item['category_id']
            # id = item['id']
            name = categories[category_id]['name']

            if image_id not in annotations:
                annotations[image_id] = images[image_id]
                annotations[image_id]['segm'] = []
                annotations[image_id]['area'] = []
                annotations[image_id]['iscrowd'] = []
                annotations[image_id]['keypoints'] = []
                annotations[image_id]['boxes'] = []
                annotations[image_id]['labels'] = []

            annotations[image_id]['segm'].append(segm)
            annotations[image_id]['area'].append(area)
            annotations[image_id]['iscrowd'].append(iscrowd)
            annotations[image_id]['keypoints'].append(keypoints)
            annotations[image_id]['boxes'].append([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])
            annotations[image_id]['labels'].append(self.classes.index(name))

        return annotations

    def __len__(self):
        return len(self.keys)

    def _load(self, idx):
        annotations = self.annotations[self.keys[idx]]
        img_path = annotations["img_path"]
        img = Image.open(img_path).convert("RGB")  # , np.uint8
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        segm = annotations["segm"]
        keypoints = annotations["keypoints"]
        area = annotations["area"]
        iscrowd = annotations["iscrowd"]

        target = dict(boxes=torch.tensor(boxes, dtype=torch.float32),
                      labels=torch.tensor(labels, dtype=torch.int64),
                      # segm=torch.tensor(segm, dtype=torch.float32),
                      segm=segm,
                      keypoints=torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3),
                      area=torch.tensor(area, dtype=torch.float32),
                      iscrowd=torch.tensor(iscrowd),
                      img_path=img_path,
                      image_id=torch.tensor([idx])
                      )

        return img, target

    def __getitem__(self, idx):
        img, target = self._load(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class MinicocoKeypointsDataset(_BaseDataset):
    """
    MScocoKeypoints 筛选得到
    """

    def __init__(self, root="", classes=['person'], transforms=None):
        self.root = root
        self.transforms = transforms
        self.classes = classes

        self.json_dir = os.path.join(self.root, 'json')
        self.json_names = os.listdir(self.json_dir)

    def _load(self, idx):
        json_name = self.json_names[idx]
        json_path = os.path.join(self.json_dir, json_name)
        jdata = json.load(open(json_path, 'r'))
        img_path = os.path.join(self.root, 'images', json_name.replace('.json', '.jpg'))
        img = Image.open(img_path).convert("RGB")  # , np.uint8
        jdata['boxes'] = torch.tensor(jdata['boxes'])
        jdata['labels'] = torch.tensor(jdata['labels'])
        # jdata['segm'] = jdata['segm']
        jdata['keypoints'] = torch.tensor(jdata['keypoints'])
        jdata['area'] = torch.tensor(jdata['area'])
        jdata['iscrowd'] = torch.tensor(jdata['iscrowd'])
        jdata.pop('segm')
        jdata['image_id'] = torch.tensor([idx])

        return img, jdata

    def __len__(self):
        return len(self.json_names)

    def __getitem__(self, idx):
        img, target = self._load(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == "__main__":
    from toolcv.tools import transform as T
    from torchvision import transforms as T2

    """
    # transforms = aug.Compose([
    #     aug.RandomHorizontalFlip(),
    #     # aug.Resize(416, 416),
    #     aug.ResizeMax(416, 416), aug.Padding(),
    #     aug.ToTensor(), aug.Normalize()])
    transforms = T.Compose([
        # T.RandomCrop(),
        # T.RandomCropV2(),
        # T.RandomCropV3(),
        # T.RandomCropAndResize((416,416)),
        # T.Resize((416,416)),
        # T.ResizeRatio((416,416)),
        # T.RandomChoice([T.Resize((416, 416)), T.ResizeRatio((416, 416))]),
        # T.RandomRoate(60),
        # T.RandomHorizontalFlip(),
        # T.RandomShift(100,100,0,50),
        # T.RandomScale(),
        T.ZoomOut(),
        # T.ResizeRatiov2((416,416),mode="down"),
        T.ToTensor(), T.Normalize()
    ])

    # dataset = MScocoKeypointsDataset(r'D:\data\coco', transforms=transforms)
    # dataset.show(0, draw='draw_keypoint', save_path='output')

    dataset = MinicocoKeypointsDataset(r'D:\data\coco\minicoco', transforms=transforms)
    dataset.show(20, "plt", draw='draw_keypoint')
    #
    # exit(0)
    # """

    # import toolcv.api.define.utils.data.augment as aug
    # import toolcv.data.augment.bboxAugv2 as baug

    dir_data = r"D:/data/fruitsNuts/"
    classes = ['date', 'fig', 'hazelnut']
    """
    # transforms = Compose([RandomHorizontalFlip(), Resize(416, 416), ToTensor(), Normalize()])
    transforms = aug.Compose([
        # aug.RandomHorizontalFlip(),
        # aug.Resize(416, 416),
        # aug.ResizeMax(416, 416),
        # aug.Padding(),
        # aug.ResizeLimit(),
        # aug.Crop(),
        # aug.WarpAffine(),
        # aug.RandomBlur(),
        # aug.RandomHSV(),
        # aug.RandomHSVCV(),
        # aug.RandomChoice([aug.RandomColor(), aug.RandomNoise()]),
        # aug.RandomErasing(), # 会改变框
        # aug.RandomBlur(),
        # aug.RandomColorJitter(),
        aug.ToTensor(),
        aug.Normalize()])
    transforms_mosaic = aug.Compose([
        aug.RandomHorizontalFlip(),
        aug.Crop(),
        aug.WarpAffine(0),
        aug.ResizeMax(416, 416),
        aug.Padding(),
        # aug.Resize(416,416),
        aug.RandomBlur(),
        aug.RandomColorJitter([0, 0.8], [0, 0.8], [0, 0.8]),
        aug.RandomNoise()
    ])
    dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, True, transforms_mosaic, 416, 416)
    dataset.show()
    # for _ in range(10000):
    #     for img, target in dataset:
    #         print(img.shape)
    """
    # transforms = aug.Compose([
    #     baug.ResizeFixMinAndRandomCrop(512, (416, 416)),
    #     # baug.Resize((512,512)),
    #     # baug.RandomCrop((416,416)),
    #     # baug.RandomRotate(),
    #     # baug.RandomAffine(),
    #     # baug.RandomCutMix(),
    #     # baug.RandomDropPixelV2(0),
    #     # baug.RandomMosaic(),
    #     aug.ToTensor(),
    #     aug.Normalize()
    # ])
    #
    # dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, False, None, 416, 416)
    # dataset.show()

    """
    transforms = aug.Compose([
        aug.RandomHorizontalFlip(),
        # aug.Crop(),aug.WarpAffine(0),
        # aug.Resize(*resize),
        # aug.ResizeMax(*resize), aug.Padding(),
        aug.RandomChoice([baug.RandomBlur(), baug.RandomNoise(), aug.RandomColorJitter()]),
        # aug.RandomChoice([baug.RandomDropPixelV2(0), baug.RandomRotate(), baug.RandomMosaicV2()]),
        # aug.RandomChoice([[aug.ResizeMax(*resize), aug.Padding()],
        #                   [baug.ResizeFixMinAndRandomCropV2(ratio_range=[0.5, 0.6, 0.7, 0.8, 0.9]),
        #                    aug.Resize(*resize)]]),
        baug.ResizeFixMinAndRandomCropV2AndPatch([100, 200, 300, 400, 500, 600], [0.5, 0.6, 0.7, 0.8, 0.9]),
        aug.Resize(416, 416),
        aug.ToTensor(), aug.Normalize()])
    
    dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, False, None, 416, 416, 0, True)
    dataset.show()
    # """

    transforms = T.Compose([
        # aug.RandomHorizontalFlip(),
        # aug.Resize(416, 416),
        # aug.ResizeMax(416, 416), aug.Padding(),
        # RandomCrop(),
        # Resize((416,416)),
        # ResizeRatio((416,416)),
        # RandomCropAndResize((416,416)),
        # RandomCropV2((416,416)),
        # ResizeRatio((416, 416)),
        # T.RandomHorizontalFlip(),
        # T.RandomRoate(60),
        # T.RandomCropV3(center=False),
        # T.NotChangeLabel(mode="RandomApply"),
        # T.RandomPerspective(),
        # T.RandomAffine(30),
        # T.RandomShift(50,50,0,20),
        # T.ResizeRatiov2((416,416),mode="down"),
        T.ZoomOut(),
        T.ToTensor(), T.Normalize(),
        # T.RandomErasing(1),
        # T.CutOut()
        # T.MosaicOne('det')
    ])

    dataset = FruitsNutsDataset(dir_data, classes, transforms, 0, False, None, 416, 416, 0, False, 10, True)
    dataset.show(60, mode='plt', draw='draw_mask')
