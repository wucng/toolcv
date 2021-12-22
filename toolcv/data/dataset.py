from xml.dom.minidom import parse
import json
import pandas as pd
from PIL import Image
import numpy as np
import os
from glob import glob

from toolcv.data.transform import CustomDataset

from toolcv.data.transform import getitem, getitem_mutilscale
from toolcv.data.transform import getitem_fourpoints
from toolcv.data.transform import getitem_ienet
from toolcv.data.transform import getitem_ssd, getitem_mutilscale_ssd
from toolcv.data.transform import getitem_ssdMS, getitem_mutilscale_ssdMS
from toolcv.data.transform import getitem_yolov2, getitem_mutilscale_yolov2
from toolcv.data.transform import getitem_yolov3, getitem_mutilscale_yolov3
from toolcv.data.transform import getitem_fcos, getitem_mutilscale_fcos
from toolcv.data.transform import getitem_fcosMS, getitem_mutilscale_fcosMS

from toolcv.tools.utils import get_xml_data, glob_format


# yolov1 centernet fcos
class FruitsNutsDatasetYOLOV1(CustomDataset):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        self.json_file = os.path.join(root, "trainval.json")
        with open(self.json_file) as f:
            imgs_anns = json.load(f)

        self.annotations = self.change_csv(imgs_anns)

    def __len__(self):
        return len(self.annotations)

    def load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        # to 0~1
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1
        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)
        return img, annotations

    def change_csv(self, imgs_anns):
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

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class FruitsNutsDatasetFCOS(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='exp',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)


    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_fcos(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_fcos(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class FruitsNutsDatasetFCOSMS(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='exp',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)


    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_fcosMS(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_fcosMS(self, idx, self.resize, self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap

class FruitsNutsDatasetFourPoints(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=30,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

    def __getitem__(self, idx):
        img, featureMap = getitem_fourpoints(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                             self.angle, self.imagu, self.advanced)

        return img, featureMap


class FruitsNutsDatasetIENet(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=30,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

    def __getitem__(self, idx):
        img, featureMap = getitem_ienet(self, idx, self.resize, self.fh,
                                        self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                        self.angle, self.imagu, self.advanced)

        return img, featureMap


class FruitsNutsDatasetSSD(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            # self.anchor 也需要根据 resize的大小变化（未实现）
            img, featureMap = getitem_mutilscale_ssd(self, idx, self.batch_size, self.strides, self.mode,
                                                     self.use_mosaic, self.fix_resize, self.anchor, self.angle,
                                                     self.imagu, self.advanced)
        else:
            img, featureMap = getitem_ssd(self, idx, self.resize, self.fh,
                                          self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                          self.anchor, self.angle, self.imagu, self.advanced)

        return img, featureMap


class FruitsNutsDatasetSSDMS(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=[8, 16, 32], muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchors=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchors = anchors

    def __getitem__(self, idx):
        if self.muilscale:
            # self.anchor 也需要根据 resize的大小变化（未实现）
            img, featureMap = getitem_mutilscale_ssdMS(self, idx, self.batch_size, self.strides, self.mode,
                                                       self.use_mosaic, self.fix_resize, self.anchors,
                                                       self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_ssdMS(self, idx, self.resize, self.strides, self.mode, self.use_mosaic,
                                            self.fix_resize, self.anchors,
                                            self.angle, self.imagu, self.advanced)
        return img, featureMap


class FruitsNutsDatasetYOLOV2(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle, self.imagu, self.advanced)
        return img, featureMap


class FruitsNutsDatasetYOLOV3(FruitsNutsDatasetYOLOV1):
    """
    - https://www.kaggle.com/fengzhongyouxia/fruitsnuts
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", classes=[], resize=416, strides=[8, 16, 32], muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchors=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchors = anchors

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov3(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchors, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov3(self, idx, self.resize, self.strides, self.mode, self.use_mosaic,
                                             self.fix_resize, self.anchors, self.angle, self.imagu, self.advanced)

        return img, featureMap


# ----------------------------------------------------------

class CarPlateDatasetYOLOV1(CustomDataset):
    """
    https://www.kaggle.com/andrewmvd/car-plate-detection
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.xml_paths = root

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

    def __len__(self):
        return len(self.xml_paths)

    def load(self, idx):
        xml_path = self.xml_paths[idx]
        annotations = get_xml_data(xml_path, self.classes)
        img_path = xml_path.replace("annotations", "images").replace('.xml', '.png')
        img = Image.open(img_path).convert("RGB")
        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class CarPlateDatasetYOLOV2(CarPlateDatasetYOLOV1):
    """
    https://www.kaggle.com/andrewmvd/car-plate-detection
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle, self.imagu, self.advanced)
        return img, featureMap


# ----------------------------------------------------------
class CarDatasetYOLOV1(CustomDataset):
    """
    https://www.kaggle.com/sshikamaru/car-object-detection
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        # self.df = pd.read_csv(os.path.join(root, "train_solution_bounding_boxes (1).csv"))
        #
        # self.imgs = pd.unique(self.df['image'])  # 一张图片可能对应多个目标

        self.datas = self._parse()

    def _parse(self):
        df = pd.read_csv(os.path.join(self.root, "train_solution_bounding_boxes (1).csv"))
        datas = df.to_numpy()
        results = {}
        for name, x1, y1, x2, y2 in datas:
            if name not in results:
                results[name] = []
            results[name].append([x1, y1, x2, y2])

        results = [{k: v} for k, v in results.items()]

        return results

    def __len__(self):
        # return len(self.imgs)
        return len(self.datas)

    def load(self, idx):
        # name, boxes = None, None
        for k, v in self.datas[idx].items():
            name, boxes = k, v
        # name = self.imgs[idx]
        # boxes = self.datas[name]

        # tmp = self.df[self.df['image'] == name]
        # # boxes = []
        # # for i in range(len(tmp)):
        # #     _, x1, y1, x2, y2 = tmp.loc[i]
        # #     boxes.append([x1, y1, x2, y2])
        # boxes = tmp.to_numpy()[:, 1:]

        # name, x1, y1, x2, y2 = self.df.loc[idx]

        img_path = os.path.join(self.root, "training_images", name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        labels = [self.classes.index("car")] * len(boxes)
        # to 0~1
        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)
        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class CarDatasetYOLOV2(CarDatasetYOLOV1):
    """
    https://www.kaggle.com/sshikamaru/car-object-detection
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle, self.imagu, self.advanced)
        return img, featureMap


# ----------------------------------------------------------

class GarbageDatasetYOLOV1(CustomDataset):
    """
    https://www.kaggle.com/qingniany/garbage-detection-1500

    classes = ['Aerosol', 'Aluminium blister pack', 'Aluminium foil', 'Battery', 'Broken glass',
    'Carded blister pack', 'Cigarette', 'Clear plastic bottle', 'Corrugated carton', 'Crisp packet',
    'Disposable food container', 'Disposable plastic cup', 'Drink can', 'Drink carton', 'Egg carton',
    'Foam cup', 'Foam food container', 'Food Can', 'Food waste', 'Garbage bag', 'Glass bottle', 'Glass cup',
    'Glass jar', 'Magazine paper', 'Meal carton', 'Metal bottle cap', 'Metal lid', 'Normal paper',
    'Other carton', 'Other plastic', 'Other plastic bottle', 'Other plastic container', 'Other plastic cup',
    'Other plastic wrapper', 'Paper bag', 'Paper cup', 'Paper straw', 'Pizza box', 'Plastic bottle cap',
    'Plastic film', 'Plastic glooves', 'Plastic lid', 'Plastic straw', 'Plastic utensils', 'Polypropylene bag',
    'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Single-use carrier bag', 'Six pack rings', 'Spread tub',
    'Squeezable tube', 'Styrofoam piece', 'Tissues', 'Toilet tube', 'Tupperware', 'Unlabeled litter', 'Wrapping paper']
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        self.paths = self._parse()

        if classes is None or len(classes) == 0:
            self.classes = self._get_classes()
            self.num_classes = len(self.classes)
            print(self.classes)

    def _get_classes(self):
        classes = []
        for img_path in self.paths:
            xml_path = img_path.replace('.jpg', '.xml').replace('.JPG', '.xml')
            classes.extend(get_xml_data(xml_path)[:, 0].tolist())

        return list(sorted(set(classes)))

    def _parse(self):
        paths = glob_format(self.root, ('.jpg', '.jpeg', '.png', ".JPG"))
        return paths

    def __len__(self):
        return len(self.paths)

    def load(self, idx):
        img_path = self.paths[idx]
        xml_path = img_path.replace('.jpg', '.xml').replace('.JPG', '.xml')
        img = Image.open(img_path).convert("RGB")
        annotations = get_xml_data(xml_path, self.classes)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class GarbageDatasetYOLOV2(GarbageDatasetYOLOV1):
    """
    https://www.kaggle.com/qingniany/garbage-detection-1500
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle, self.imagu, self.advanced)
        return img, featureMap


# -------------------------------------------------------
class PascalVOCDatasetYOLOV1(CustomDataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        self.xml_paths = glob_format(os.path.join(self.root, "Annotations"), (".xml",))

    def __len__(self):
        return len(self.xml_paths)

    def load(self, idx):
        xml_path = self.xml_paths[idx]
        img_path = xml_path.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
        img = Image.open(img_path).convert("RGB")
        annotations = get_xml_data(xml_path, self.classes)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class FDDBDatasetDatasetYOLOV1(CustomDataset):
    """
    !wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
    !wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz

    http://vis-www.cs.umass.edu/fddb/
    人脸检测数据集，只有一个类别即人脸
    tar zxf FDDB-folds.tgz FDDB-folds
    tar zxf originalPics.tar.gz originalPics

    原数据注释是采用椭圆格式如下
    <major_axis_radius minor_axis_radius angle center_x center_y 1>
    转成矩形格式(x,y,w,h)为:
    [center_x center_y,minor_axis_radius*2,major_axis_radius*2]
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        assert "face" in self.classes

        self.img_path = os.path.join(root, "originalPics")
        self.annot_path = os.path.join(root, "FDDB-folds")

        self.annotations = self.change2csv()

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations = []
        txts = glob(os.path.join(self.annot_path, "*ellipseList.txt"))
        for txt in txts:
            fp = open(txt)
            datas = fp.readlines()
            tmp = {"image": "", "boxes": []}
            for data in datas:
                data = data.strip()  # 去掉末尾换行符
                if "img_" in data:
                    if len(tmp["image"]) > 0:
                        annotations.append(tmp)
                        tmp = {"image": "", "boxes": []}

                    tmp["image"] = os.path.join(self.img_path, data + ".jpg")
                elif len(data) < 8:
                    continue
                else:
                    tmp_box = []
                    box = list(map(float, filter(lambda x: len(x) > 0, data.split(" "))))
                    tmp_box.extend(box[3:5])  # cx,cy
                    # tmp_box.extend(box[:2]) # w,h
                    tmp_box.extend([2 * box[1], 2 * box[0]])  # w,h
                    # to x1y1x2y2
                    x1 = tmp_box[0] - tmp_box[2] / 2
                    y1 = tmp_box[1] - tmp_box[3] / 2
                    x2 = tmp_box[0] + tmp_box[2] / 2
                    y2 = tmp_box[1] + tmp_box[3] / 2
                    tmp_box = [x1, y1, x2, y2]
                    tmp_box.append(box[2])  # 角度
                    tmp["boxes"].append(tmp_box)
            fp.close()

        return annotations

    def __len__(self):
        return len(self.annotations)

    def load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["image"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = []
        labels = []

        for box in annotations["boxes"]:
            boxes.append(box[:4])
            labels.append(self.classes.index("face"))

        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1

        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class WIDERFACEDatasetYOLOV1(CustomDataset):
    """
    http://shuoyang1213.me/WIDERFACE/
    人脸检测数据集，只有一个类别即人脸
    unzip wider_face_split.zip -d wider_face_split
    unzip WIDER_train.zip -d WIDER_train

    原数据注释格式如下
    [left, top, width, height, score]
    x1, y1, w, h, 代表人脸框的位置（检测算法一般都要画个框框把人脸圈出来）
    blur：是模糊度，分三档：0，清晰；1：一般般；2：人鬼难分
    express：表达（什么鬼也没弄明白，反正我训这个用不着）
    illumination：曝光，分正常和过曝
    occlusion：遮挡，分三档。0，无遮挡；1，小遮挡；2，大遮挡；
    invalid：（没弄明白）
    pose：（疑似姿态？分典型和非典型姿态）

    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        self.img_path = os.path.join(root, "WIDER_train")
        self.annot_path = os.path.join(root, "wider_face_split")
        assert "face" in self.classes

        self.annotations = self.change2csv()

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations = []
        # txts=glob(os.path.join(self.annot_path,"*ellipseList.txt"))
        # for txt in txts:
        txt = os.path.join(self.annot_path, "wider_face_train_bbx_gt.txt")
        fp = open(txt)
        datas = fp.readlines()
        tmp = {"image": "", "boxes": []}
        for data in datas:
            data = data.strip()  # 去掉末尾换行符
            if ".jpg" in data:
                if len(tmp["image"]) > 0 and len(tmp["boxes"]) > 0:
                    annotations.append(tmp)
                    tmp = {"image": "", "boxes": []}

                if len(tmp["image"]) > 0 and len(tmp["boxes"]) == 0:
                    tmp = {"image": "", "boxes": []}

                tmp["image"] = os.path.join(self.img_path, "images", data)
            elif len(data) < 8:
                continue
            else:
                tmp_box = []
                box = list(map(float, filter(lambda x: len(x) > 0, data.split(" "))))
                if int(box[4]) == 2 or int(box[7]) == 2: continue
                if int(box[2]) * int(box[3]) < 120: continue
                tmp_box.extend(box[:2])  # x1,y1
                tmp_box.extend(box[2:4])  # w,h
                # to x1y1x2y2
                x1 = tmp_box[0]
                y1 = tmp_box[1]
                x2 = tmp_box[0] + tmp_box[2]
                y2 = tmp_box[1] + tmp_box[3]
                tmp_box = [x1, y1, x2, y2]
                tmp["boxes"].append(tmp_box)
        fp.close()

        return annotations

    def __len__(self):
        return len(self.annotations)

    def load(self, idx):
        annotations = self.annotations[idx]
        img_path = annotations["image"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = []
        labels = []

        for box in annotations["boxes"]:
            boxes.append(box[:4])
            labels.append(self.classes.index("face"))

        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1
        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class PennFudanDatasetYOLOV1(CustomDataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        assert "person" in self.classes

        # 确保imgs与masks相对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def load(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        mask = Image.open(mask_path)

        # instances are encoded as different colors
        # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
        obj_ids = np.unique(mask)  # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
        # first id is the background, so remove it
        # first id是背景，所以删除它
        obj_ids = obj_ids[1:]  # array([1, 2], dtype=uint8)

        # split the color-encoded mask into a set
        # of binary masks ,0,1二值图像
        # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
        masks = mask == obj_ids[:, None, None]  # shape (2, 536, 559)，2个mask
        # obj_ids[:, None, None] None为增加对应的维度，shape为 [2, 1, 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        labels = []
        for i in range(num_objs):  # mask反算对应的bbox
            pos = np.where(masks[i])  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index("person"))

        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h - 1)
        boxes /= np.array([[w, h, w, h]])  # to 0~1
        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        annotations = np.concatenate((labels[..., None], boxes), -1)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


# -------------------------------------------------------------------------------

class XrayDatasetYOLOV1(CustomDataset):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    classes2idx = {'knife': 1, 'scissors': 2, 'sharpTools': 3, 'expandableBaton': 4, 'smallGlassBottle': 5,
    'electricBaton': 6, 'plasticBeverageBottle': 7, 'plasticBottleWithaNozzle': 8, 'electronicEquipment': 9,
    'battery': 10, 'seal': 11, 'umbrella': 12 }
    classes = ['knife','scissors','sharpTools','expandableBaton','smallGlassBottle','electricBaton',
    'plasticBeverageBottle','plasticBottleWithaNozzle','electronicEquipment','battery','seal','umbrella']
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        if isinstance(strides, int):
            self.fw = int(resize / strides)
            self.fh = int(resize / strides)

        self.xml_paths = glob_format(self.root, (".xml",))

    def __len__(self):
        return len(self.xml_paths)

    def load(self, idx):
        xml_path = self.xml_paths[idx]
        img_path = xml_path.replace(".xml",".jpg").replace("/XML","").replace("\\XML","")

        img = Image.open(img_path).convert("RGB")
        # w, h = img.size

        annotations = get_xml_data(xml_path,self.classes)

        return img, annotations

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale(self, idx, self.batch_size,
                                                 self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                 self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem(self, idx, self.resize, self.fh,
                                      self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                      self.angle, self.imagu, self.advanced)

        return img, featureMap


class XrayDatasetSSD(XrayDatasetYOLOV1):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            # self.anchor 也需要根据 resize的大小变化（未实现）
            img, featureMap = getitem_mutilscale_ssd(self, idx, self.batch_size, self.strides, self.mode,
                                                     self.use_mosaic, self.fix_resize, self.anchor, self.angle,
                                                     self.imagu, self.advanced)
        else:
            img, featureMap = getitem_ssd(self, idx, self.resize, self.fh,
                                          self.fw, self.mode, self.use_mosaic, self.fix_resize,
                                          self.anchor, self.angle, self.imagu, self.advanced)

        return img, featureMap


class XrayDatasetSSDMS(XrayDatasetYOLOV1):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    """

    def __init__(self, root="", classes=[], resize=416, strides=[8, 16, 32], muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchors=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchors = anchors

    def __getitem__(self, idx):
        if self.muilscale:
            # self.anchor 也需要根据 resize的大小变化（未实现）
            img, featureMap = getitem_mutilscale_ssdMS(self, idx, self.batch_size, self.strides, self.mode,
                                                       self.use_mosaic, self.fix_resize, self.anchors,
                                                       self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_ssdMS(self, idx, self.resize, self.strides, self.mode, self.use_mosaic,
                                            self.fix_resize, self.anchors,
                                            self.angle, self.imagu, self.advanced)
        return img, featureMap


class XrayDatasetYOLOV2(XrayDatasetYOLOV1):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    """

    def __init__(self, root="", classes=[], resize=416, strides=32, muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchor=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchor = anchor

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov2(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchor, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov2(self, idx, self.resize, self.fh,
                                             self.fw, self.mode, self.use_mosaic, self.fix_resize, self.anchor,
                                             self.angle, self.imagu, self.advanced)
        return img, featureMap


class XrayDatasetYOLOV3(XrayDatasetYOLOV1):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    """

    def __init__(self, root="", classes=[], resize=416, strides=[8, 16, 32], muilscale=False,
                 batch_size=32, num_anchors=1, transform=None, mode='fcos',
                 use_mosaic=False, fix_resize=False, anchors=[], angle=None,
                 imagu=True, advanced=False):
        super().__init__(root, classes, resize, strides, muilscale, batch_size, num_anchors, transform,
                         mode, use_mosaic, fix_resize, angle, imagu, advanced)

        self.anchors = anchors

    def __getitem__(self, idx):
        if self.muilscale:
            img, featureMap = getitem_mutilscale_yolov3(self, idx, self.batch_size,
                                                        self.strides, self.mode, self.use_mosaic, self.fix_resize,
                                                        self.anchors, self.angle, self.imagu, self.advanced)
        else:
            img, featureMap = getitem_yolov3(self, idx, self.resize, self.strides, self.mode, self.use_mosaic,
                                             self.fix_resize, self.anchors, self.angle, self.imagu, self.advanced)

        return img, featureMap
