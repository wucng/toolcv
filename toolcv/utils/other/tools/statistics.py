"""
对样本数据做统计

分类：
file_name |  img_W(原图片的宽)  |  img_H(原图片的高)  |  label   |

目标检测
pandas: 列标题为：
file_name |  img_W(原图片的宽)  |  img_H(原图片的高)  |  label   |  x1 |  y1  |  x2 |  y2  | w/h（gt_box宽高比）| gt_area(w*h) | iou
file_name |  img_W(原图片的宽)  |  img_H(原图片的高)  |  label   |  x1 |  y1  |  x2 |  y2  |  # gt_box，gt_area，iou 可选

coco 评估中：
small object: area < 32^2
medium object: 32^2 < area < 96^2
large object: area > 96^2
"""
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import cv2
import torch
import os
from glob import glob
from tqdm import tqdm
# from .tool import glob_format
from toolsmall.tools.utils.tools import box_iou

def static_cls(data_path,save_path=None,fmt_list = ('.jpg', '.jpeg', '.png',".xml")):
    if save_path is None:
        save_path = os.path.join(data_path,os.path.basename(data_path)+".csv")
    datasets = []
    classes = sorted(os.listdir(data_path))
    for cName in classes:
        paths = os.listdir(os.path.join(data_path,cName))
        for file_name in tqdm(paths):
            path = os.path.join(data_path,cName,file_name)
            fmt = os.path.splitext(file_name)[-1]
            if fmt in fmt_list:
                h,w = cv2.imread(path).shape[:2]
                datasets.append([file_name,w,h,cName])

    df = pd.DataFrame(np.array(datasets),columns=["file_name","img_W","img_H","label"])
    df.to_csv(save_path,index=False)

def static_obj_detecte(dataset:Dataset,save_path=None,priorBoxes=None):
    if save_path is None:
        save_path = "./static_obj.csv"
    result = []
    for image,target in tqdm(dataset):
        if isinstance(image,torch.Tensor):
            h,w = image.shape[-2:]
        else:
            h,w = np.array(image).shape[:2]
        file_name = os.path.basename(target["path"])
        labels = target["labels"].int().numpy()
        boxes = target["boxes"]#.float().numpy()
        if priorBoxes is not None:
            gt = boxes / torch.tensor([[w,h,w,h]],dtype=torch.float32)
            ious = box_iou(priorBoxes,gt)
            ious = ious.max(0)[0] # 计算每个gt对应的最大IOU

        boxes = boxes.float().numpy()
        for i,(label,box) in enumerate(zip(labels,boxes)):# boxes x1,y1,x2,y2 格式 未作缩放
            result.append([file_name,w,h,label,*box,ious[i].item() if priorBoxes is not None else 0])

    df = pd.DataFrame(np.array(result), columns=["file_name", "img_W", "img_H", "label",'x1','y1','x2','y2','iou'])
    df.to_csv(save_path, index=False)


if __name__=="__main__":
    data_path = "/media/wucong/225A6D42D4FA828F1/datas/flower_photos"
    static_cls(data_path)

    # from toolsmall.cls.statistics import static_obj
    # from toolsmall.tools.utils.anchor import getAnchorsV2_s
    # priorBoxes = torch.from_numpy(getAnchorsV2_s((img_size,img_size),stride,anchor_size,aspect_ratios))
    # test_transforms = get_transform_fixsize(False, img_size)
    # dataset = PascalVOCDatasetV2(data_path,test_transforms,classes)
    # static_obj_detecte(dataset,priorBoxes=priorBoxes)
    # exit(0)