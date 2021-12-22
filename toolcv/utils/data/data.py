"""
    设置每个类别的权重
    1、weight = median(numsOfClass) / numsOfClass
    2、weight = 1/ln(c+numsOfClass)  c=1.02

    3、weight = mean(numsOfClass)/numsOfClass
    4、weight = sum(numsOfClass)/numsOfClass
    5、weight = max(numsOfClass)/numsOfClass
"""
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

from torchvision import transforms as T2

from toolcv.utils.data import transform as T
from toolcv.utils.tools.vis import draw_rect, draw_mask, draw_segms, draw_keypoint, draw_keypointV2
# from toolcv.utils.data.data import _BaseDataset, collate_fn
from toolcv.utils.tools.tools import polygons2mask, mask2polygons, compute_area_from_polygons, glob_format
from toolcv.utils.tools.tools2 import get_train_val_dataset, set_seed, DataLoaderX
from toolcv.tools.utils import get_xml_data


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

    def __getitem__(self, idx):

        if self.mode == "mixup":
            img, target = T.mixup(self, idx, 'det')
        elif self.mode == "cutmix":
            img, target = T.cutmix(self, idx, 'det')
        elif self.mode == "mosaic":
            img, target = T.mosaic(self, idx, 'det')
        elif self.mode == "mosaicTwo":
            img, target = T.mosaicTwo(self, idx, 'det')
        elif self.mode == "mosaicOne":
            img, target = T.mosaicOne(self, idx, 'det')
        else:
            img, target = self._load(idx)

        return img, target


class FruitsNutsDataset(_BaseDataset):
    """
    数据 是 mscoco 格式
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """
    classes = ['date', 'fig', 'hazelnut']

    def __init__(self, root="", classes=[], transforms=None, masks=False, mode="none"):
        self.root = root
        self.transforms = transforms
        if len(classes) > 0: self.classes = classes
        self.masks = masks
        self.mode = mode

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
            masks = target["masks"]
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


class BalloonDataset(_BaseDataset):
    """
    !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    !unzip balloon_dataset.zip > /dev/null
    """

    def __init__(self, root="", classes=["balloon"], transforms=None, masks=False, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.masks = masks
        self.mode = mode
        assert "balloon" in self.classes

        self.json_file = os.path.join(root, "via_region_data.json")
        with open(self.json_file) as f:
            self.imgs_anns = json.load(f)
        self.keys = list(self.imgs_anns.keys())

    def __len__(self):
        return len(self.keys)

    def _load(self, idx):
        tdata = self.imgs_anns[self.keys[idx]]
        img_path = os.path.join(os.path.dirname(self.json_file), tdata["filename"])
        img = Image.open(img_path).convert("RGB")

        annos = tdata["regions"]
        boxes = []
        labels = []
        if self.masks: masks = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            boxes.append([np.min(px), np.min(py), np.max(px), np.max(py)])
            labels.append(self.classes.index("balloon"))

            if self.masks:
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                masks.append(polygons2mask(img.size[::-1], [poly]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"labels": labels, 'boxes': boxes}
        if self.masks:
            target.update({'masks': torch.from_numpy(np.stack(masks, 0))})
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        if 'masks' not in target:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            masks = target["masks"]
            area = torch.tensor([compute_area_from_polygons(mask2polygons(mask.cpu().numpy())) for mask in masks])

        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class MScocoKeypointsDataset(_BaseDataset):
    """
    https://blog.csdn.net/u014734886/article/details/78830713
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    """

    def __init__(self, root="", classes=['person'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class MinicocoKeypointsDataset(_BaseDataset):
    """
    MScocoKeypoints 筛选得到
    """

    def __init__(self, root="", classes=['person'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

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

        if self.transforms is not None:
            img, jdata = self.transforms(img, jdata)

        return img, jdata

    def __len__(self):
        return len(self.json_names)


class PascalVOCDataset(_BaseDataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/

    classes = ["aeroplane","bicycle","bird","boat","bottle",
                "bus","car","cat","chair","cow",
                "diningtable","dog","horse","motorbike","person",
                "pottedplant","sheep","sofa","train","tvmonitor"
    ]
    # 训练集
    numsOfClass = [238,243,330,181,244,
                     186,713,337,445,141,
                     200,421,287,245,2008,
                     245,96,229,261,256]
    设置每个类别的权重
    1、weight = median(numsOfClass) / numsOfClass
    2、weight = 1/ln(c+numsOfClass)  c=1.02

    3、weight = mean(numsOfClass)/numsOfClass
    4、weight = sum(numsOfClass)/numsOfClass
    5、weight = max(numsOfClass)/numsOfClass
    """

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"
               ]

    def __init__(self, root="", xml_paths="", classes=[],
                 transforms=None, mode="none"):
        root = root
        self.transforms = transforms
        if len(classes) > 0: self.classes = classes
        self.mode = mode

        if len(xml_paths) > 0:
            if not os.path.exists(xml_paths): xml_paths = os.path.join(root, "ImageSets", "Main", xml_paths)
            _xml_paths = []
            with open(xml_paths) as fp:
                tmp = fp.read().split("\n")
            for t in tmp:
                if len(t) == 0: continue
                _xml_paths.append(os.path.join(root, "Annotations", "%s.xml" % t))
        else:
            _xml_paths = glob_format(os.path.join(root, "Annotations"), (".xml",))
        self.xml_paths = _xml_paths

    def __len__(self):
        return len(self.xml_paths)

    def _load(self, idx):
        xml_path = self.xml_paths[idx]
        img_path = xml_path.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
        img = Image.open(img_path).convert("RGB")
        annotations = get_xml_data(xml_path, self.classes)
        labels = torch.from_numpy(annotations[:, 0]).long()
        boxes = torch.from_numpy(annotations[:, 1:]).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CarPlateDataset(_BaseDataset):
    """
    https://www.kaggle.com/andrewmvd/car-plate-detection
    https://www.kaggle.com/andrewmvd/face-mask-detection
    # options={"with_mask":0,"without_mask":1,"mask_weared_incorrect":2}
    classes = ["with_mask","without_mask","mask_weared_incorrect"]
    """

    def __init__(self, root="", classes=['car'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

        self.xml_paths = glob_format(self.root)

    def __len__(self):
        return len(self.xml_paths)

    def _load(self, idx):
        xml_path = self.xml_paths[idx]
        annotations = get_xml_data(xml_path, self.classes)
        img_path = xml_path.replace("annotations", "images").replace('.xml', '.png')
        img = Image.open(img_path).convert("RGB")

        labels = torch.from_numpy(annotations[:, 0]).long()
        boxes = torch.from_numpy(annotations[:, 1:]).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class GarbageDataset(_BaseDataset):
    """
    https://www.kaggle.com/qingniany/garbage-detection-1500
    """
    classes = ['Aerosol', 'Aluminium blister pack', 'Aluminium foil', 'Battery', 'Broken glass',
               'Carded blister pack', 'Cigarette', 'Clear plastic bottle', 'Corrugated carton', 'Crisp packet',
               'Disposable food container', 'Disposable plastic cup', 'Drink can', 'Drink carton', 'Egg carton',
               'Foam cup', 'Foam food container', 'Food Can', 'Food waste', 'Garbage bag', 'Glass bottle', 'Glass cup',
               'Glass jar', 'Magazine paper', 'Meal carton', 'Metal bottle cap', 'Metal lid', 'Normal paper',
               'Other carton', 'Other plastic', 'Other plastic bottle', 'Other plastic container', 'Other plastic cup',
               'Other plastic wrapper', 'Paper bag', 'Paper cup', 'Paper straw', 'Pizza box', 'Plastic bottle cap',
               'Plastic film', 'Plastic glooves', 'Plastic lid', 'Plastic straw', 'Plastic utensils',
               'Polypropylene bag',
               'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Single-use carrier bag', 'Six pack rings',
               'Spread tub',
               'Squeezable tube', 'Styrofoam piece', 'Tissues', 'Toilet tube', 'Tupperware', 'Unlabeled litter',
               'Wrapping paper']

    def __init__(self, root="", classes=[], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        if len(classes) > 0: self.classes = classes
        self.mode = mode

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

    def _load(self, idx):
        img_path = self.paths[idx]
        xml_path = img_path.replace('.jpg', '.xml').replace('.JPG', '.xml')
        img = Image.open(img_path).convert("RGB")
        annotations = get_xml_data(xml_path, self.classes)

        labels = torch.from_numpy(annotations[:, 0]).long()
        boxes = torch.from_numpy(annotations[:, 1:]).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class FDDBDataset(_BaseDataset):
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

    def __init__(self, root="", classes=['face'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

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

    def _load(self, idx):
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

        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        labels = torch.from_numpy(labels).long()
        boxes = torch.from_numpy(boxes).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class WIDERFACEDataset(_BaseDataset):
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

    def __init__(self, root="", classes=['face'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

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

    def _load(self, idx):
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

        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        labels = torch.from_numpy(labels).long()
        boxes = torch.from_numpy(boxes).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class PennFudanDataset(_BaseDataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """

    def __init__(self, root="", classes=['person'], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.mode = mode

        assert "person" in self.classes

        # 确保imgs与masks相对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def _load(self, idx):
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

        # boxes = np.array(boxes, dtype=np.float32) / np.array([[w, h, w, h]], dtype=np.float32)  # to 0~1
        labels = np.array(labels, dtype=np.float32)

        labels = torch.from_numpy(labels).long()
        boxes = torch.from_numpy(boxes).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class XrayDataset(_BaseDataset):
    """
    http://challenge.xfyun.cn/topic/info?type=Xray-2021
    classes2idx = {'knife': 1, 'scissors': 2, 'sharpTools': 3, 'expandableBaton': 4, 'smallGlassBottle': 5,
    'electricBaton': 6, 'plasticBeverageBottle': 7, 'plasticBottleWithaNozzle': 8, 'electronicEquipment': 9,
    'battery': 10, 'seal': 11, 'umbrella': 12 }
    classes = ['knife','scissors','sharpTools','expandableBaton','smallGlassBottle','electricBaton',
    'plasticBeverageBottle','plasticBottleWithaNozzle','electronicEquipment','battery','seal','umbrella']
    """
    classes = ['knife', 'scissors', 'sharpTools', 'expandableBaton', 'smallGlassBottle', 'electricBaton',
               'plasticBeverageBottle', 'plasticBottleWithaNozzle', 'electronicEquipment', 'battery', 'seal',
               'umbrella']

    def __init__(self, root="", classes=[], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        if len(classes) > 0: self.classes = classes
        self.mode = mode

        self.xml_paths = glob_format(self.root, (".xml",))

    def __len__(self):
        return len(self.xml_paths)

    def _load(self, idx):
        xml_path = self.xml_paths[idx]
        img_path = xml_path.replace(".xml", ".jpg").replace("/XML", "").replace("\\XML", "")

        img = Image.open(img_path).convert("RGB")
        # w, h = img.size

        annotations = get_xml_data(xml_path, self.classes)

        labels = torch.from_numpy(annotations[:, 0]).long()
        boxes = torch.from_numpy(annotations[:, 1:]).float()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class HumanVehicleDataset(_BaseDataset):
    """
    https://www.datafountain.cn/competitions/552/datasets
    """
    classes = ['BiCyclist', 'Bike', 'CampusBus', 'Car', 'EngineTruck', 'HeavyTruck', 'LargeBus', 'LightTruck', 'MMcar',
               'Machineshop', 'MediumBus', 'MotorCyclist', 'Motorcycle', 'OtherCar', 'Pedestrian', 'PersonSitting',
               'Pickup', 'TricycleClosed', 'TricycleOpenHuman', 'TricycleOpenMotor', 'Truck', 'van']

    def __init__(self, root="", classes=[], transforms=None, mode="none"):
        self.root = root
        self.transforms = transforms
        if len(classes) > 0: self.classes = classes
        self.mode = mode

        self.annotations = self.parse(os.path.join(self.root, "train.json"))
        self.keys = list(self.annotations.keys())

    def parse(self, path):
        annotations = {}
        datas = json.load(open(path))["annotations"]
        for data in datas:
            filename = data["filename"]
            label = data["label"]
            truncated = data["truncated"]
            occluded = data["occluded"]
            box = data["box"]
            boxes = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
            if boxes[0] is None: continue
            if filename not in annotations:
                annotations[filename] = dict(labels=[], truncated=[], occluded=[], boxes=[])

            annotations[filename]["labels"].append(self.classes.index(label))
            annotations[filename]["truncated"].append(truncated)
            annotations[filename]["occluded"].append(occluded)
            annotations[filename]["boxes"].append(boxes)

        return annotations

    def __len__(self):
        return len(self.keys)

    def _load(self, idx):
        # img_path = os.path.join(self.root, self.keys[idx])
        img_path = os.path.join(self.root, "train_images",
                                self.keys[idx].split("\\")[-1])  # os.path.basename(self.keys[idx])
        annotations = self.annotations[self.keys[idx]]

        img = Image.open(img_path).convert("RGB")
        # w, h = img.size
        labels = torch.tensor(annotations["labels"]).long()
        boxes = torch.tensor(annotations["boxes"]).float()

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        target = {"labels": labels, 'boxes': boxes}
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class DataDefine:
    def __init__(self, root, classes, batch_size=8, size=416, train_ratio=0.8, drop_last=False):
        if isinstance(size, int): size = (size, size)
        self.size = size
        self.root, self.classes = root, classes
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.drop_last = drop_last

    def set_transform(self, train_transforms=None, val_transforms=None, level=0):
        if train_transforms is None:
            if level == 0:
                train_transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ResizeRatio(self.size),
                    T.ToTensor(),
                    T.Normalize()
                ])
            elif level == 1:
                train_transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCropV3([0.7, 0.8, 0.9, 0.95], ratio=0.7),
                    T.ResizeRatio(self.size),
                    T.ToTensor(),
                    T.Normalize()
                ])
            else:
                train_transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCropV3([0.7, 0.8, 0.9, 0.95], ratio=0.7),
                    T.ZoomOut([0.8, 0.9]),
                    T.RandomChoice([T.ResizeRatiov2(self.size, mode="none"), T.ResizeRatio(self.size)]),
                    # T.NotChangeLabel([T2.ColorJitter(0.5, 0.5, 0.5, 0.5), T2.GaussianBlur((5, 5))]),
                    # T.RandomApply(T.NotChangeLabel([T2.ColorJitter(0.5, 0.5, 0.5, 0.5)])),
                    T.RandomApply(T.NotChangeLabel([T2.GaussianBlur((5, 5))])),
                    # T.RandomApply([T.NotChangeLabel([T2.ColorJitter(0.5, 0.5, 0.5, 0.5)]),
                    #                T.NotChangeLabel([T2.GaussianBlur((5, 5))])]),
                    T.ToTensor(),
                    T.Normalize()
                ])

        if val_transforms is None:
            val_transforms = T.Compose([
                T.ResizeRatio(self.size),
                T.ToTensor(),
                T.Normalize()
            ])

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def get_dataloader(self, train_dataset=None, val_dataset=None):
        if train_dataset is None:
            train_dataset = FruitsNutsDataset(self.root, self.classes, self.train_transforms)
        if val_dataset is None:
            val_dataset = FruitsNutsDataset(self.root, self.classes, self.val_transforms)

        if self.train_ratio > 0:
            train_dataset, val_dataset = get_train_val_dataset(train_dataset, val_dataset, self.train_ratio)

        train_dataloader = DataLoaderX(train_dataset, self.batch_size, True, drop_last=self.drop_last,
                                       collate_fn=collate_fn)
        val_dataloader = DataLoaderX(val_dataset, 1, False, collate_fn=collate_fn)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


if __name__ == "__main__":
    # dir_data = r"D:/data/fruitsNuts/"
    # classes = ['date', 'fig', 'hazelnut']
    # dir_data = r'D:\data\coco\minicoco'
    # classes = ["person"]
    """
    dir_data = r"D:\data\Xray"
    classes = ['knife', 'scissors', 'sharpTools', 'expandableBaton', 'smallGlassBottle', 'electricBaton',
               'plasticBeverageBottle', 'plasticBottleWithaNozzle', 'electronicEquipment', 'battery', 'seal',
               'umbrella']
    ComDataset = XrayDataset
    data = DataDefine(dir_data, classes, 1, 416, 0)
    data.set_transform()
    train_dataset = ComDataset(os.path.join(dir_data, "train"), classes, data.train_transforms)
    val_dataset = ComDataset(os.path.join(dir_data, "test"), classes, data.val_transforms)
    data.get_dataloader(train_dataset, val_dataset)
    train_dataset.show(20, "plt", draw='draw_rect')
    """

    """
    dir_data = r"D:\practice\datas\balloon"
    classes = ["balloon"]
    data = DataDefine(dir_data, classes, 1, 416, 0)
    data.set_transform()
    train_dataset = BalloonDataset(os.path.join(dir_data, "train"), classes, data.train_transforms,True)
    val_dataset = BalloonDataset(os.path.join(dir_data, "val"), classes, data.val_transforms)
    data.get_dataloader(train_dataset, val_dataset)
    train_dataset.show(20, "plt", draw='draw_mask')
    """

    dir_data = r"D:\data\human_vehicle_detection"
    classes = ['BiCyclist', 'Bike', 'CampusBus', 'Car', 'EngineTruck', 'HeavyTruck', 'LargeBus', 'LightTruck',
               'MMcar',
               'Machineshop', 'MediumBus', 'MotorCyclist', 'Motorcycle', 'OtherCar', 'Pedestrian', 'PersonSitting',
               'Pickup', 'TricycleClosed', 'TricycleOpenHuman', 'TricycleOpenMotor', 'Truck', 'van']
    # data = DataDefine(dir_data, classes, 1, 416, 0)
    # data.set_transform()
    # train_dataset = HumanVehicleDataset(dir_data, data.classes, data.train_transforms)
    # print(data.classes)
    # # train_dataset = PascalVOCDataset(dir_data, "train.txt", classes,data.train_transforms)
    # # val_dataset = PascalVOCDataset(dir_data, "val.txt", classes,data.train_transforms)
    # # data.get_dataloader(train_dataset, val_dataset)
    # train_dataset.show(20, "plt", draw='draw_rect')

    # from toolcv.tools.anchor import gen_anchorv2
    # gen_anchorv2(HumanVehicleDataset(dir_data, classes), 15)

    resize = (416, 416)
    data = DataDefine(dir_data, classes, 1, resize, 0)
    data.set_transform(level=1)
    train_dataset = HumanVehicleDataset(dir_data, classes, data.train_transforms)
    val_dataset = HumanVehicleDataset(dir_data, classes, data.val_transforms)
    data.get_dataloader(train_dataset, val_dataset)

    train_dataset.show(100, 'cv2', 'draw_rect')
