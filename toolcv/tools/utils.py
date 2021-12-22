import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset  # ,TensorDataset
# from torchvision.datasets import ImageFolder,DatasetFolder
from torchvision.ops.boxes import batched_nms, nms
from torchvision import transforms as T
import numpy as np
import cv2
from PIL import Image
from fvcore.nn import sigmoid_focal_loss, giou_loss
import matplotlib.pyplot as plt
import os
import random
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import math
from tqdm import tqdm


def get_xml_dataV0(xml_file, classes, normal=False):
    """解析 xml格式 poscal voc 数据格式"""
    # out_file = open(out_file, 'w')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    img_box = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        # difficult = obj.find('difficult').text
        # if cls not in classes: #or int(difficult) == 1:
        #     continue
        if classes is not None:
            cls_id = classes.index(cls)
        else:
            cls_id = cls

        xmlbox = obj.find('bndbox')
        x1, y1, x2, y2 = min(max(0, float(xmlbox.find('xmin').text)), w - 1), \
                         min(max(0, float(xmlbox.find('ymin').text)), h - 1), \
                         min(max(0, float(xmlbox.find('xmax').text)), w - 1), \
                         min(max(0, float(xmlbox.find('ymax').text)), h - 1)
        if normal:
            img_box.append([cls_id, x1 / w, y1 / h, x2 / w, y2 / h])
        else:
            img_box.append([cls_id, x1, y1, x2, y2])
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return img_box


def get_xml_data(xml_file, classes=None, normal=False):
    """
    解析 xml格式 poscal voc 数据格式
    :param xml_file:
    :param classes:
    :return:
    """
    dom = parse(xml_file)
    root = dom.documentElement
    # img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = root.getElementsByTagName("size")[0]
    objects = root.getElementsByTagName("object")
    img_w = float(img_size.getElementsByTagName("width")[0].childNodes[0].data)
    img_h = float(img_size.getElementsByTagName("height")[0].childNodes[0].data)
    # img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    # print("img_name:", img_name)
    # print("image_info:(w,h,c)", img_w, img_h, img_c)
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        # to 0~1
        x1 = min(max(0, float(box.getElementsByTagName("xmin")[0].childNodes[0].data)), img_w - 1)
        y1 = min(max(0, float(box.getElementsByTagName("ymin")[0].childNodes[0].data)), img_h - 1)
        x2 = min(max(0, float(box.getElementsByTagName("xmax")[0].childNodes[0].data)), img_w - 1)
        y2 = min(max(0, float(box.getElementsByTagName("ymax")[0].childNodes[0].data)), img_h - 1)
        if normal:
            x1 /= img_w
            y1 /= img_h
            x2 /= img_w
            y2 /= img_h

        if classes is not None:
            img_box.append([classes.index(cls_name), x1, y1, x2, y2])
        else:
            img_box.append([cls_name, x1, y1, x2, y2])

    return np.array(img_box)


# fmt_list = ('.jpg', '.jpeg', '.png',".xml")
def glob_format(path, fmt_list=('.jpg', '.jpeg', '.png', ".JPG"), base_name=False):
    # print('--------pid:%d start--------------' % (os.getpid()))

    fs = []
    if not os.path.exists(path): return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:
                fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:
                fs.append(item)
    # print('--------pid:%d end--------------' % (os.getpid()))
    return fs


def _nms(heat, kernel=3):
    """
    :param heat: torch.tensor [bs,c,h,w]
    :param kernel:
    :return:
    """
    pad = (kernel - 1) // 2

    hmax = torch.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
    # return hmax == heat


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x0, y0, x1, y1) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x0, y0, x1, y1) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_iou_np(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


"""只使用w，h计算IOU"""


def wh_iou(wh1, wh2):
    """
    wh1 = torch.tensor([[3.62500,  2.81250],[4.87500,  6.18750],[11.65625, 10.18750]]) # anchor
    wh2 = torch.tensor([[9.37500, 8.48438]]) # gt

    print(wh_iou(wh1,wh2))

    # 等价于
    ious=box_iou(xywh2x1y1x2y2(torch.cat((torch.zeros_like(wh1),wh1),-1)),
            xywh2x1y1x2y2(torch.cat((torch.zeros_like(wh2),wh2),-1)))

    print(ious)

    # --------------------------
    tensor([[0.12818],
        [0.37923],
        [0.66983]])

    tensor([[0.12818],
            [0.37923],
            [0.66983]])
    """
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou_np(wh1, wh2):
    """
    wh1 = torch.tensor([[3.62500,  2.81250],[4.87500,  6.18750],[11.65625, 10.18750]]) # anchor
    wh2 = torch.tensor([[9.37500, 8.48438]]) # gt

    print(wh_iou(wh1,wh2))

    # 等价于
    ious=box_iou(xywh2x1y1x2y2(torch.cat((torch.zeros_like(wh1),wh1),-1)),
            xywh2x1y1x2y2(torch.cat((torch.zeros_like(wh2),wh2),-1)))

    print(ious)

    # --------------------------
    tensor([[0.12818],
        [0.37923],
        [0.66983]])

    tensor([[0.12818],
            [0.37923],
            [0.66983]])
    """
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2x1y1x2y2(boxes):
    """
    xywh->x1y1x2y2

    :param boxes: [...,4]
    :return:
    """
    x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
    x2y2 = boxes[..., :2] + boxes[..., 2:] / 2

    return torch.cat((x1y1, x2y2), -1)


def xywh2x1y1x2y2_np(boxes):
    """
    xywh->x1y1x2y2

    :param boxes: [...,4]
    :return:
    """
    x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
    x2y2 = boxes[..., :2] + boxes[..., 2:] / 2

    return np.concatenate((x1y1, x2y2), -1)


def x1y1x2y22xywh(boxes):
    """
    x1y1x2y2-->xywh

    :param boxes: [...,4]
    :return:
    """
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = boxes[..., 2:] - boxes[..., :2]

    return torch.cat((xy, wh), -1)


def x1y1x2y22xywh_np(boxes):
    """
    x1y1x2y2-->xywh

    :param boxes: [...,4]
    :return:
    """
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = boxes[..., 2:] - boxes[..., :2]

    return np.concatenate((xy, wh), -1)


# 效果差
def train2(model, optim, dataLoader, device, epoch, print_step=20, max_norm=0.1):
    reduction = "sum"
    model.train()
    for step, (img, target) in enumerate(dataLoader):
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()

        output = torch.sigmoid(model(img))

        # conf
        conf_keep = target[..., 0] == 1
        noconf_keep = target[..., 0] != 1

        output_conf = output[conf_keep]
        output_noconf = output[noconf_keep]
        target_conf = target[conf_keep]
        target_noconf = target[noconf_keep]

        conf_loss = F.binary_cross_entropy(output_conf[..., 0], target_conf[..., 0], reduction=reduction)
        noconf_loss = F.binary_cross_entropy(output_noconf[..., 0], target_noconf[..., 0], reduction=reduction) * 0.5
        conf_loss = conf_loss + noconf_loss

        # box
        boxes_loss = F.binary_cross_entropy(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)
        # boxes_loss = F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5],reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(output_conf[..., 5:], target_conf[..., 5:], reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        if step % print_step == 0:
            print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
                epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
            ))


# train_yolov1
def train(model, optim, dataLoader, device, epoch, muilscale=False, mode='fcosv2', print_step=20, reduction="sum",
          boxes_weight=5.0, max_norm=0.1, thres=0.3, box_norm="log"):
    model.train()

    # nums = len(dataLoader.dataset)
    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []
    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        if muilscale:
            img = img[0].to(device)
            target = target[0].to(device)
        else:
            img = img.to(device)
            target = target.to(device)

        # 多尺度训练
        # size = np.random.choice([9,11,13,15,17,19])
        # size = size*32
        # img = F.interpolate(img,(size,size),mode="bilinear",align_corners=True)
        # target = F.interpolate(target.squeeze(-2).permute(0,3,1,2),(size//stride,size//stride),mode="nearest",align_corners=True)
        # target = target.permute(0,2,3,1).unsqueeze(-2)

        optim.zero_grad()

        output = model(img)

        # conf
        if mode in ['fcosv2', 'centernetv2']:
            conf_keep = target[..., 0] > thres
        else:
            conf_keep = target[..., 0] == 1
        # output_conf = torch.sigmoid(output[conf_keep])
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        # if mode in ['fcosv2', 'centernetv2']:
        #     boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)
        # else:
        #     # boxes_loss = boxes_weight*F.binary_cross_entropy(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],reduction=reduction)
        #     boxes_loss = boxes_weight * F.mse_loss(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],
        #                                            reduction=reduction)
        if 'log' not in box_norm and mode not in ['fcosv2', 'centernetv2']:
            boxes_loss = boxes_weight * F.binary_cross_entropy(torch.sigmoid(output_conf[..., 1:5]),
                                                               target_conf[..., 1:5], reduction=reduction)
        else:
            boxes_loss = boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5],
                                                         reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        desc = "epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item())
        # if step % print_step == 0:
        #     print(desc)
        pbar.set_description(desc)  # 设置进度条左边显示的信息
        # pbar.set_postfix(epoch=epoch, step=step, loss=loss.item(), conf_loss=conf_loss.item(),
        #                  boxes_loss=boxes_loss.item(),cls_loss=cls_loss.item())  # 设置进度条右边显示的信息

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict(model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
            fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3,
            box_norm="log"):
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))

    conf = torch.sigmoid(output[..., 0])
    keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

    # if mode in ['fcosv2', 'centernetv2']:
    #     boxes = output[..., 1:5]
    # else:
    #     boxes = torch.sigmoid(output[..., 1:5])

    if 'log' not in box_norm and mode not in ['fcosv2', 'centernetv2']:
        boxes = torch.sigmoid(output[..., 1:5])
    else:
        boxes = output[..., 1:5]

    shift_x = np.arange(0, fw)
    shift_y = np.arange(0, fh)
    X, Y = np.meshgrid(shift_x, shift_y)
    xy = np.stack((X, Y), -1)
    boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

    if box_norm == "log":
        boxes[..., 2:] = boxes[..., 2:].exp()
    elif box_norm == "logv2":
        boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor((1 / fw, 1 / fh), dtype=torch.float32, device=device)
    elif box_norm == "sqrt":
        boxes[..., 2:] = boxes[..., 2:] ** 2

    scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

    conf = conf * scores
    # keep = conf > conf_threshold
    keep = torch.bitwise_and(conf > conf_threshold, keep)

    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]
    conf = conf[keep]

    # nms
    boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(boxes, scores, labels, iou_threshold)
    keep = batched_nms(boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# -----------------------------------------------------------------------------------------------------
def train_fcos(model, optim, dataLoader, device, epoch, muilscale=False, mode='exp', print_step=20,
               reduction="sum", boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        if muilscale:
            img = img[0].to(device)
            target = target[0].to(device)
        else:
            img = img.to(device)
            target = target.to(device)

        optim.zero_grad()

        output = model(img)

        # conf
        conf_keep = target[..., 0] > thres
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        if mode == "sigmoid":
            boxes_loss = boxes_weight * F.mse_loss(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],
                                                   reduction=reduction)
        elif mode == "exp":
            boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:5].exp(), target_conf[..., 1:5],
                                                   reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        # if step % print_step == 0:
        #     print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
        #         epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
        #     ))

        desc = "epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item())
        pbar.set_description(desc)  # 设置进度条左边显示的信息

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_fcos(model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
                 fix_resize=False, mode='exp', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))

    conf = torch.sigmoid(output[..., 0])
    keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

    fwhwh = torch.tensor([[fw, fh, fw, fh]], dtype=torch.float32, device=device)
    if mode == "sigmoid":
        boxes = output[..., 1:5].sigmoid() * fwhwh
    elif mode == "exp":
        boxes = output[..., 1:5].exp()

    shift_x = np.arange(0, fw)
    shift_y = np.arange(0, fh)
    X, Y = np.meshgrid(shift_x, shift_y)
    xy = np.stack((X, Y), -1)
    # boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    # boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

    # x1y1x2y2
    tmp = torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    boxes[..., :2] = tmp - boxes[..., :2]
    boxes[..., 2:] += tmp
    boxes /= fwhwh  # 0~1

    scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

    conf = conf * scores
    # keep = conf > conf_threshold
    keep = torch.bitwise_and(conf > conf_threshold, keep)

    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]
    conf = conf[keep]

    # nms
    # boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(boxes, scores, labels, iou_threshold)
    keep = batched_nms(boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# -----------------------------------------------------------------------------
# 多分支
def train_fcosMS(model, optim, dataLoader, device, epoch, muilscale=False, mode='exp', print_step=20,
                 reduction="sum", boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    for step, (img, target) in enumerate(dataLoader):
        if muilscale:
            img = img[0].to(device)
            target = target[0].to(device)
        else:
            img = img.to(device)
            target = target.to(device)

        optim.zero_grad()

        output = model(img)
        x8, x16, x32 = output
        bs = x8.size(0)
        c = x8.size(-1)
        output = torch.cat((x8.contiguous().view(bs, -1, c), x16.contiguous().view(bs, -1, c),
                            x32.contiguous().view(bs, -1, c)), 1)

        # conf
        conf_keep = target[..., 0] > thres
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        mode = mode.split("-")[0]
        if mode == "sigmoid":
            boxes_loss = boxes_weight * F.mse_loss(torch.sigmoid(output_conf[..., 1:5]), target_conf[..., 1:5],
                                                   reduction=reduction)
        elif mode == "exp":
            boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:5].exp(), target_conf[..., 1:5],
                                                   reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        if step % print_step == 0:
            print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
                epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
            ))

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_fcosMS(model, img_path, transform, resize, device, visual=True, strides=[8, 16, 32],
                   fix_resize=False, mode='exp', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    mode = mode.split("-")[0]
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    outputs = model(img[None].to(device))

    _boxes = []
    _conf = []
    _scores = []
    _labels = []

    for i, stride in enumerate(strides):
        output = outputs[i]
        fw, fh = resize // stride, resize // stride

        conf = torch.sigmoid(output[..., 0])
        keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

        fwhwh = torch.tensor([[fw, fh, fw, fh]], dtype=torch.float32, device=device)
        if mode == "sigmoid":
            boxes = output[..., 1:5].sigmoid() * fwhwh
        elif mode == "exp":
            boxes = output[..., 1:5].exp()

        shift_x = np.arange(0, fw)
        shift_y = np.arange(0, fh)
        X, Y = np.meshgrid(shift_x, shift_y)
        xy = np.stack((X, Y), -1)
        # boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        # boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

        # x1y1x2y2
        tmp = torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        boxes[..., :2] = tmp - boxes[..., :2]
        boxes[..., 2:] += tmp
        boxes /= fwhwh  # 0~1

        scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

        conf = conf * scores
        # keep = conf > conf_threshold
        keep = torch.bitwise_and(conf > conf_threshold, keep)

        if keep.sum() == 0:
            return "no object detecte"
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]
        conf = conf[keep]

        _boxes.append(boxes)
        _scores.append(scores)
        _labels.append(labels)
        _conf.append(conf)

    boxes = torch.cat(_boxes, 0)
    scores = torch.cat(_scores, 0)
    labels = torch.cat(_labels, 0)
    conf = torch.cat(_conf, 0)

    # nms
    # boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(boxes, scores, labels, iou_threshold)
    keep = batched_nms(boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# -----------------------------------------------------------------------------------------------------
def train_fourpoints(model, optim, dataLoader, device, epoch, mode='fcosv2', print_step=20, reduction="sum",
                     boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    for step, (img, target) in enumerate(dataLoader):

        img = img.to(device)
        target = target.to(device)

        optim.zero_grad()

        output = model(img)

        # conf
        if mode in ['fcosv2', 'centernetv2']:
            conf_keep = target[..., 0] > thres
        else:
            conf_keep = target[..., 0] == 1
        # output_conf = torch.sigmoid(output[conf_keep])
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:9], target_conf[..., 1:9], reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 9:]), target_conf[..., 9:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        if step % print_step == 0:
            print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
                epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
            ))

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_fourpoints(model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
                       fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3,
                       fourpoints=True, angle_range=45):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    if angle_range > 0:
        # angle = random.randint(-angle_range, angle_range)
        angle = random.choice(range(-angle_range, angle_range + 1, 5))
        img = img.rotate(angle)
    _img = np.array(img).copy()

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))

    conf = torch.sigmoid(output[..., 0])
    keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

    boxes = output[..., 1:9]

    shift_x = np.arange(0, fw)
    shift_y = np.arange(0, fh)
    X, Y = np.meshgrid(shift_x, shift_y)
    xy = np.stack((X, Y, X, Y, X, Y, X, Y), -1)
    boxes += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    boxes /= torch.tensor([[fw, fh, fw, fh, fw, fh, fw, fh]], dtype=torch.float32, device=device)
    boxes = boxes.clamp(0, 1)
    scores, labels = torch.sigmoid(output[..., 9:]).max(-1)

    conf = conf * scores
    # keep = conf > conf_threshold
    keep = torch.bitwise_and(conf > conf_threshold, keep)

    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]
    conf = conf[keep]

    # nms
    # 从4点转成2点
    _boxes = boxes.clone().view(-1, 4, 2)
    x1y1 = _boxes.min(1)[0]
    x2y2 = _boxes.max(1)[0]
    _boxes = torch.cat((x1y1, x2y2), -1)
    # boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(_boxes, scores, labels, iou_threshold)
    keep = batched_nms(_boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"

    if fourpoints:  # 4点（左上角 右上角 右下角 左下角）
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]
        boxes *= torch.tensor([[w, h, w, h, w, h, w, h]], dtype=boxes.dtype, device=boxes.device)
        boxes[..., [0, 2, 4, 6]] = boxes[..., [0, 2, 4, 6]].clamp(0, w - 1)
        boxes[..., [1, 3, 5, 7]] = boxes[..., [1, 3, 5, 7]].clamp(0, h - 1)

        img = _img  # cv2.imread(img_path)
        img = drawImg_fourpoints(img, boxes, labels, scores)

    else:  # 原来的2点（左上角与右下角）
        boxes, scores, labels, conf = _boxes[keep], scores[keep], labels[keep], conf[keep]
        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

        img = _img  # cv2.imread(img_path)
        img = drawImg(img, boxes, labels, scores)

    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img)
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img[..., ::-1])

    return "%d object detecte" % (len(labels))


# ---------------------------------------------------------------------------------------------------

def train_ienet(model, optim, dataLoader, device, epoch, mode='fcosv2', print_step=20, reduction="sum",
                boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    for step, (img, target) in enumerate(dataLoader):

        img = img.to(device)
        target = target.to(device)

        optim.zero_grad()

        output = model(img)

        # conf
        if mode in ['fcosv2', 'centernetv2']:
            conf_keep = target[..., 0] > thres
        else:
            conf_keep = target[..., 0] == 1
        # output_conf = torch.sigmoid(output[conf_keep])
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        if mode in ['fcosv2', 'centernetv2']:
            boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:7], target_conf[..., 1:7], reduction=reduction)
        else:
            boxes_loss = boxes_weight * F.mse_loss(torch.sigmoid(output_conf[..., 1:7]), target_conf[..., 1:7],
                                                   reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 7:]), target_conf[..., 7:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        if step % print_step == 0:
            print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
                epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
            ))

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_ienet(model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
                  fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3,
                  ienet=True, angle_range=45):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    if angle_range > 0:
        # angle = random.randint(-angle_range, angle_range)
        angle = random.choice(range(-angle_range, angle_range + 1, 5))
        img = img.rotate(angle)
    _img = np.array(img).copy()

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))

    conf = torch.sigmoid(output[..., 0])
    keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

    if mode in ['fcosv2', 'centernetv2']:
        boxes = output[..., 1:7]
    else:
        boxes = torch.sigmoid(output[..., 1:7])

    shift_x = np.arange(0, fw)
    shift_y = np.arange(0, fh)
    X, Y = np.meshgrid(shift_x, shift_y)
    xy = np.stack((X, Y), -1)
    boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)
    boxes[..., -2:] *= boxes[..., 2:-2]
    boxes = boxes.clamp(0, 1)
    scores, labels = torch.sigmoid(output[..., 7:]).max(-1)

    conf = conf * scores
    # keep = conf > conf_threshold
    keep = torch.bitwise_and(conf > conf_threshold, keep)

    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]
    conf = conf[keep]

    # nms
    _boxes = xywh2x1y1x2y2(boxes[..., :4])
    # keep = batched_nms(_boxes, scores, labels, iou_threshold)
    keep = batched_nms(_boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"

    if ienet:  # 4点（左上角 右上角 右下角 左下角）
        boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]
        boxes[..., :4] = xywh2x1y1x2y2(boxes[..., :4])
        boxes *= torch.tensor([[w, h, w, h, w, h]], dtype=boxes.dtype, device=boxes.device)
        boxes[..., [0, 2, 4]] = boxes[..., [0, 2, 4]].clamp(0, w - 1)
        boxes[..., [1, 3, 5]] = boxes[..., [1, 3, 5]].clamp(0, h - 1)

        # 转成4点坐标
        x_min, y_min, x_max, y_max, w1, h1 = boxes.split(1, -1)
        w = x_max - x_min
        h = y_max - y_min

        x1 = x_min
        y1 = y_max - h1
        x2 = x_max - w1
        y2 = y_min
        x3 = x_max
        y3 = y_min + h1
        x4 = x_min + w1
        y4 = y_max
        boxes = torch.cat((x1, y1, x2, y2, x3, y3, x4, y4), -1)

        img = _img  # cv2.imread(img_path)
        img = drawImg_fourpoints(img, boxes, labels, scores)

    else:  # 原来的2点（左上角与右下角）
        boxes, scores, labels, conf = _boxes[keep], scores[keep], labels[keep], conf[keep]
        # 回复到原始输入图像大小
        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

        img = _img  # cv2.imread(img_path)
        img = drawImg(img, boxes, labels, scores)

    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img)
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img[..., ::-1])

    return "%d object detecte" % (len(labels))


# -----------------------------------------------------------------------------------------------------

# b, g, r = np.random.randint(32, 256, [3])
# base_v = np.random.randint(64, 128)

b, g, r = np.random.choice(np.arange(32, 256, 20), 3, replace=False)
base_v = np.random.choice(np.arange(64, 128, 10))


def drawImg(img, boxes, labels, scores):
    boxes = boxes.round().int().cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        b1, g1, r1 = int((b + base_v * label) % 255), int((g + base_v * label) % 255), int((r + base_v * label) % 255)
        x1, y1, x2, y2 = box
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (b1, g1, r1), 2)
        cv2.putText(img, '%d:%.3f' % (label, score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def drawImg_fourpoints(img, boxes, labels, scores):
    boxes = boxes.round().int().cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        b1, g1, r1 = int((b + base_v * label) % 255), int((g + base_v * label) % 255), int((r + base_v * label) % 255)
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        # cv2.polylines(img, box.reshape(1,4,2), 1, (b1, g1, r1), 2)
        cv2.line(img, (x1, y1), (x2, y2), (b1, g1, r1), 2)
        cv2.line(img, (x2, y2), (x3, y3), (b1, g1, r1), 2)
        cv2.line(img, (x3, y3), (x4, y4), (b1, g1, r1), 2)
        cv2.line(img, (x4, y4), (x1, y1), (b1, g1, r1), 2)
        cv2.putText(img, '%d:%.3f' % (label, score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


# ----------------------------------------------------------------------------------------
def train_yolov2(model, optim, dataLoader, device, epoch, muilscale=False, mode='fcosv2', print_step=20,
                 reduction="sum", boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        if muilscale:
            img = img[0].to(device)
            target = target[0].to(device)
        else:
            img = img.to(device)
            target = target.to(device)

        optim.zero_grad()

        output = model(img)

        # conf
        if mode in ['fcosv2', 'centernetv2']:
            conf_keep = target[..., 0] > thres
        else:
            conf_keep = target[..., 0] == 1
        # output_conf = torch.sigmoid(output[conf_keep])
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        no_ignore = target[..., 0] != -1

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0][no_ignore], target[..., 0][no_ignore], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)
        # boxes_loss = boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        # if step % print_step == 0:
        #     print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
        #         epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
        #     ))
        desc = "epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item())
        pbar.set_description(desc)  # 设置进度条左边显示的信息

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_yolov2(anchor, model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
                   fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    anchor = np.array(anchor)
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))

    conf = torch.sigmoid(output[..., 0])
    keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

    boxes = output[..., 1:5]

    boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor(anchor, dtype=torch.float32, device=device)[None, None]

    shift_x = np.arange(0, fw)
    shift_y = np.arange(0, fh)
    X, Y = np.meshgrid(shift_x, shift_y)
    xy = np.stack((X, Y), -1)
    boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
    boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

    scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

    conf = conf * scores
    # keep = conf > conf_threshold
    keep = torch.bitwise_and(conf > conf_threshold, keep)

    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]
    conf = conf[keep]

    # nms
    boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(boxes, scores, labels, iou_threshold)
    keep = batched_nms(boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# -------------------------------------------------------------------------------------------

def train_yolov3(model, optim, dataLoader, device, epoch, muilscale=False, mode='fcosv2', print_step=20,
                 reduction="sum", boxes_weight=5.0, max_norm=0.1, thres=0.3):
    model.train()

    conf_loss_list = []
    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        if muilscale:
            img = img[0].to(device)
            target = target[0].to(device)
        else:
            img = img.to(device)
            target = target.to(device)

        optim.zero_grad()

        output = model(img)
        x8, x16, x32 = output
        bs = x8.size(0)
        c = x8.size(-1)
        output = torch.cat((x8.contiguous().view(bs, -1, c), x16.contiguous().view(bs, -1, c),
                            x32.contiguous().view(bs, -1, c)), 1)

        # conf
        if mode in ['fcosv2', 'centernetv2']:
            conf_keep = target[..., 0] > thres
        else:
            conf_keep = target[..., 0] == 1
        # output_conf = torch.sigmoid(output[conf_keep])
        output_conf = output[conf_keep]
        target_conf = target[conf_keep]

        no_ignore = target[..., 0] != -1

        # conf_loss = sigmoid_focal_loss(output[..., 0], target[..., 0], 0.4, 2, reduction)
        # conf_loss = sigmoid_focal_loss(output[..., 0][no_ignore], target[..., 0][no_ignore], 0.4, 2, reduction)
        conf_loss = sigmoid_focal_loss(output[..., 0][no_ignore], target[..., 0][no_ignore], 0.4, 2, "none")
        if reduction == "sum":
            conf_loss = conf_loss.sum()
        elif reduction == "mean":
            conf_loss = conf_loss.sum() / conf_keep.sum()

        # box
        boxes_loss = boxes_weight * F.mse_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)
        # boxes_loss = boxes_weight * F.smooth_l1_loss(output_conf[..., 1:5], target_conf[..., 1:5], reduction=reduction)

        # class
        cls_loss = F.binary_cross_entropy(torch.sigmoid(output_conf[..., 5:]), target_conf[..., 5:],
                                          reduction=reduction)

        loss = conf_loss + boxes_loss + cls_loss

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        # if step % print_step == 0:
        #     print("epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
        #         epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item()
        #     ))

        desc = "epoch:%d step:%d loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), conf_loss.item(), boxes_loss.item(), cls_loss.item())
        pbar.set_description(desc)  # 设置进度条左边显示的信息

        conf_loss_list.append(conf_loss.item())
        boxes_loss_list.append(boxes_loss.item())
        cls_loss_list.append(cls_loss.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list), "conf_loss": np.mean(conf_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f conf_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                                         loss_dict["total_loss"],
                                                                                         loss_dict["conf_loss"],
                                                                                         loss_dict["boxes_loss"],
                                                                                         loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_yolov3(anchors, model, img_path, transform, resize, device, visual=True, strides=[8, 16, 32],
                   fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3, conf_threshold=0.3):
    anchors = np.array(anchors)
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    outputs = model(img[None].to(device))

    _boxes = []
    _conf = []
    _scores = []
    _labels = []
    for i, stride in enumerate(strides):
        output = outputs[i]
        anchor = anchors[i]
        fw, fh = resize // stride, resize // stride

        conf = torch.sigmoid(output[..., 0])
        keep = _nms(conf.permute(0, 3, 1, 2), 3).permute(0, 2, 3, 1)

        boxes = output[..., 1:5]

        boxes[..., 2:] = boxes[..., 2:].exp() * torch.tensor(anchor, dtype=torch.float32, device=device)[None, None]

        shift_x = np.arange(0, fw)
        shift_y = np.arange(0, fh)
        X, Y = np.meshgrid(shift_x, shift_y)
        xy = np.stack((X, Y), -1)
        boxes[..., :2] += torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-2)
        boxes[..., :2] /= torch.tensor([[fw, fh]], dtype=torch.float32, device=device)

        scores, labels = torch.sigmoid(output[..., 5:]).max(-1)

        conf = conf * scores
        # keep = conf > conf_threshold
        keep = torch.bitwise_and(conf > conf_threshold, keep)

        if keep.sum() == 0:
            return "no object detecte"
        boxes = boxes[keep]
        scores, labels = scores[keep], labels[keep]
        conf = conf[keep]

        _boxes.append(boxes)
        _scores.append(scores)
        _labels.append(labels)
        _conf.append(conf)

    boxes = torch.cat(_boxes, 0)
    scores = torch.cat(_scores, 0)
    labels = torch.cat(_labels, 0)
    conf = torch.cat(_conf, 0)

    # nms
    boxes = xywh2x1y1x2y2(boxes)
    # keep = batched_nms(boxes, scores, labels, iou_threshold)
    keep = batched_nms(boxes, conf, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels, conf = boxes[keep], scores[keep], labels[keep], conf[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# ----------------------------------------------------------------------------------------
def train_ssd(model, optim, dataLoader, device, epoch, reduction="sum", print_step=20, focal_loss=False,
              boxes_weight=5.0, max_norm=0.1):
    model.train()

    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        img = img.to(device)
        target = target.to(device).view(-1, 5)

        optim.zero_grad()

        output = model(img)

        boxes = output[..., :4]
        cls = output[..., 4:]

        gt_boxes = target[..., :4]
        gt_cls = target[..., 4]
        positive = gt_cls > 0
        nums_positive = positive.sum()
        if focal_loss:
            keep = gt_cls >= 0
            loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
                                          reduction="none")
            if reduction == "sum":
                loss_cls = loss_cls.sum()
            elif reduction == "mean":
                loss_cls = loss_cls.sum() / nums_positive

        else:
            with torch.no_grad():
                loss = -F.log_softmax(cls, -1)[..., 0]  # 对应 softmax ,第0列对应背景
                loss[positive] = -np.inf
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            gt_cls[~positive] = -1
            gt_cls[negindex[:3 * nums_positive]] = 0  # 负样本

            # loss_cls = F.cross_entropy(cls, gt_cls.long(), ignore_index=-1, reduction=reduction)

            keep = gt_cls >= 0
            loss_cls = F.cross_entropy(cls[keep], gt_cls[keep].long(), reduction=reduction)

        loss_boxes = boxes_weight * F.smooth_l1_loss(boxes[positive], gt_boxes[positive], reduction=reduction)
        # loss_boxes = boxes_weight * F.mse_loss(boxes[positive], gt_boxes[positive], reduction=reduction)

        loss = loss_cls + loss_boxes

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        # if step % print_step == 0:
        #     print("epoch:%d step:%d loss:%.5f loss_cls:%.5f loss_boxes:%.5f" % (
        #         epoch, step, loss.item(), loss_cls.item(), loss_boxes.item()))

        desc = "epoch:%d step:%d loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), loss_boxes.item(), loss_cls.item())
        pbar.set_description(desc)  # 设置进度条左边显示的信息

        boxes_loss_list.append(loss_boxes.item())
        cls_loss_list.append(loss_cls.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                          loss_dict["total_loss"],
                                                                          loss_dict["boxes_loss"],
                                                                          loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_ssd(anchor, model, img_path, transform, resize, device, visual=True, fh=13, fw=13,
                fix_resize=False, save_path="./output", iou_threshold=0.3, conf_threshold=0.3, focal_loss=False):
    # anchor = np.array(anchor)
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))
    boxes = output[..., :4]
    if focal_loss:
        cls = output[..., 4:].sigmoid()
    else:
        cls = output[..., 4:].softmax(-1)

    anchor_xywh = torch.tensor(x1y1x2y22xywh_np(anchor), dtype=torch.float32, device=device)

    boxes[..., 2:] = boxes[..., 2:].exp() * anchor_xywh[..., 2:]
    boxes[..., :2] = boxes[..., :2] * anchor_xywh[..., 2:] + anchor_xywh[..., :2]

    scores, labels = cls.max(-1)

    keep = torch.bitwise_and(scores > conf_threshold, labels > 0)  # 0 为背景
    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]

    # nms
    boxes = xywh2x1y1x2y2(boxes)
    keep = batched_nms(boxes, scores, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# ----------------------------------------------------------------------------------------
def train_ssdMS(model, optim, dataLoader, device, epoch, reduction="sum", print_step=20, focal_loss=False,
                boxes_weight=5.0, max_norm=0.1):
    model.train()

    boxes_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    pbar = tqdm(enumerate(dataLoader))
    for step, (img, target) in pbar:
        img = img.to(device)
        target = target.to(device).view(-1, 5)

        optim.zero_grad()

        output = model(img)

        boxes = output[..., :4]
        cls = output[..., 4:]

        gt_boxes = target[..., :4]
        gt_cls = target[..., 4]
        positive = gt_cls > 0
        nums_positive = positive.sum()
        if focal_loss:
            keep = gt_cls >= 0
            # loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
            #                               reduction=reduction)
            loss_cls = sigmoid_focal_loss(cls[keep], F.one_hot(gt_cls[keep].long(), cls.size(-1)).float(), 0.4, 2,
                                          reduction="none")
            if reduction == "sum":
                loss_cls = loss_cls.sum()
            elif reduction == "mean":
                loss_cls = loss_cls.sum() / nums_positive

        else:
            with torch.no_grad():
                loss = -F.log_softmax(cls, -1)[..., 0]  # 对应 softmax ,第0列对应背景
                loss[positive] = -np.inf
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            gt_cls[~positive] = -1
            gt_cls[negindex[:3 * nums_positive]] = 0  # 负样本

            # loss_cls = F.cross_entropy(cls, gt_cls.long(), ignore_index=-1, reduction=reduction)

            keep = gt_cls >= 0
            loss_cls = F.cross_entropy(cls[keep], gt_cls[keep].long(), reduction=reduction)

        loss_boxes = boxes_weight * F.smooth_l1_loss(boxes[positive], gt_boxes[positive], reduction=reduction)
        # loss_boxes = boxes_weight * F.mse_loss(boxes[positive], gt_boxes[positive], reduction=reduction)

        loss = loss_cls + loss_boxes

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_([parm for parm in model.parameters() if parm.requires_grad],
                                           max_norm=max_norm)
        optim.step()

        # if step % print_step == 0:
        #     print("epoch:%d step:%d loss:%.5f loss_cls:%.5f loss_boxes:%.5f" % (
        #         epoch, step, loss.item(), loss_cls.item(), loss_boxes.item()))

        desc = "epoch:%d step:%d loss:%.5f boxes_loss:%.5f cls_loss:%.5f" % (
            epoch, step, loss.item(), loss_boxes.item(), loss_cls.item())
        pbar.set_description(desc)  # 设置进度条左边显示的信息

        boxes_loss_list.append(loss_boxes.item())
        cls_loss_list.append(loss_cls.item())
        total_loss_list.append(loss.item())

    loss_dict = {"total_loss": np.mean(total_loss_list),
                 "boxes_loss": np.mean(boxes_loss_list), 'cls_loss': np.mean(cls_loss_list)}
    print('-' * 60)
    print("| epoch:%d total_loss:%.5f boxes_loss:%.5f cls_loss:%.5f |" % (epoch,
                                                                          loss_dict["total_loss"],
                                                                          loss_dict["boxes_loss"],
                                                                          loss_dict["cls_loss"]))
    print('-' * 60)

    return loss_dict


@torch.no_grad()
def predict_ssdMS(anchors, model, img_path, transform, resize, device, visual=True, strides=[8, 16, 32],
                  fix_resize=False, save_path="./output", iou_threshold=0.3, conf_threshold=0.3, focal_loss=False):
    # anchor = np.array(anchor)
    model.eval()
    img = Image.open(img_path).convert("RGB")

    if fix_resize:
        # resize
        w, h = img.size
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

    img = transform(img)

    output = model(img[None].to(device))
    boxes = output[..., :4]
    if focal_loss:
        cls = output[..., 4:].sigmoid()
    else:
        cls = output[..., 4:].softmax(-1)

    anchor = np.concatenate(anchors, 0)
    anchor_xywh = torch.tensor(x1y1x2y22xywh_np(anchor), dtype=torch.float32, device=device)

    boxes[..., 2:] = boxes[..., 2:].exp() * anchor_xywh[..., 2:]
    boxes[..., :2] = boxes[..., :2] * anchor_xywh[..., 2:] + anchor_xywh[..., :2]

    scores, labels = cls.max(-1)

    keep = torch.bitwise_and(scores > conf_threshold, labels > 0)  # 0 为背景
    if keep.sum() == 0:
        return "no object detecte"
    boxes = boxes[keep]
    scores, labels = scores[keep], labels[keep]

    # nms
    boxes = xywh2x1y1x2y2(boxes)
    keep = batched_nms(boxes, scores, labels, iou_threshold)
    if len(keep) == 0:
        return "no object detecte"
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # 回复到原始输入图像大小
    boxes[..., 0] *= w
    boxes[..., 1] *= h
    boxes[..., 2] *= w
    boxes[..., 3] *= h

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h - 1)

    img = cv2.imread(img_path)
    img = drawImg(img, boxes, labels, scores)
    if visual:
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        if not os.path.exists(save_path): os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img)

    return "%d object detecte" % (len(labels))


# --------------------------------------------------------------------------------------
def dtrain(model, optim, dataLoader, device, epoch, muilscale=False, mode='fcosv2', print_step=20, reduction="sum",
           boxes_weight=5.0, max_norm=0.1, method="yolov1", focal_loss=False, thres=0.3):
    """
    1、method="yolov1"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    2、method="fcos"     mode = 'sigmoid', 'exp'
    3、method="fcosms"   mode = 'sigmoid-v1', 'exp-v1'  ;'v2'效果差
    4、method="yolov2"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    5、method="yolov3"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    6、method="ssd"      mode = 'v1', 'v2' ;'v2'效果差
    7、method="ssdms"    mode = 'v1', 'v2' ;'v2'效果差
    """

    method = method.lower()

    if method == "yolov1":
        assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
        loss_dict = train(model, optim, dataLoader, device, epoch, muilscale, mode, print_step, reduction, boxes_weight,
                          max_norm, thres)
    elif method == "fcos":
        assert mode in ['sigmoid', 'exp']
        loss_dict = train_fcos(model, optim, dataLoader, device, epoch, muilscale, mode, print_step, reduction,
                               boxes_weight, max_norm, thres)

    elif method == "fcosms":
        assert mode.split("-")[0] in ['sigmoid', 'exp']
        assert mode.split("-")[1] in ['v1', 'v2']  # 'v2'效果差
        loss_dict = train_fcosMS(model, optim, dataLoader, device, epoch, muilscale, mode, print_step, reduction,
                                 boxes_weight, max_norm, thres)
    elif method == "yolov2":
        assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
        loss_dict = train_yolov2(model, optim, dataLoader, device, epoch, muilscale, mode, print_step, reduction,
                                 boxes_weight, max_norm, thres)
    elif method == "yolov3":
        assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
        loss_dict = train_yolov3(model, optim, dataLoader, device, epoch, muilscale, mode, print_step, reduction,
                                 boxes_weight, max_norm, thres)
    elif method == "ssd":  # 单分支
        assert mode in ['v1', 'v2']  # 'v2'效果差
        loss_dict = train_ssd(model, optim, dataLoader, device, epoch, reduction, print_step, focal_loss, boxes_weight,
                              max_norm)
    elif method == "ssdms":  # 多分支
        assert mode in ['v1', 'v2']  # 'v2'效果差
        loss_dict = train_ssdMS(model, optim, dataLoader, device, epoch, reduction, print_step, focal_loss,
                                boxes_weight, max_norm)
    else:
        raise ("error!!")

    return loss_dict


def evalute(model, img_paths, transform=None, resize=416, device='cpu', visual=True, strides=16,
            fix_resize=False, mode='fcosv2', save_path="./output", iou_threshold=0.3,
            conf_threshold=0.3, method="yolov1", anchors=[], focal_loss=False):
    """
    1、method="yolov1"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    2、method="fcos"     mode = 'sigmoid', 'exp'
    3、method="fcosms"   mode = 'sigmoid-v1', 'exp-v1'  ;'v2'效果差
    4、method="yolov2"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    5、method="yolov3"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    6、method="ssd"      mode = 'v1', 'v2' ;'v2'效果差
    7、method="ssdms"    mode = 'v1', 'v2' ;'v2'效果差
    """

    method = method.lower()

    save_model = os.path.join(save_path, 'weight.pth')
    if os.path.exists(save_model):
        model.load_state_dict(torch.load(save_model, map_location=device))
        print("load weight successful!!!")

    if transform is None:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    for img_path in img_paths:
        if method == "yolov1":
            assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
            fh, fw = resize // strides, resize // strides
            info = predict(model, img_path, transform, resize, device, visual, fh, fw,
                           fix_resize, mode, save_path, iou_threshold, conf_threshold)

        elif method == "fcos":
            assert mode in ['sigmoid', 'exp']
            fh, fw = resize // strides, resize // strides
            info = predict_fcos(model, img_path, transform, resize, device, visual, fh, fw,
                                fix_resize, mode, save_path, iou_threshold, conf_threshold)

        elif method == "fcosms":
            # assert mode in ['sigmoid', 'exp']
            assert mode.split("-")[0] in ['sigmoid', 'exp']
            assert mode.split("-")[1] in ['v1', 'v2']  # 'v2'效果差
            info = predict_fcosMS(model, img_path, transform, resize, device, visual, strides,
                                  fix_resize, mode, save_path, iou_threshold, conf_threshold)

        elif method == "yolov2":
            assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
            fh, fw = resize // strides, resize // strides
            info = predict_yolov2(anchors, model, img_path, transform, resize, device, visual, fh, fw,
                                  fix_resize, mode, save_path, iou_threshold, conf_threshold)

        elif method == "yolov3":
            assert mode in ['yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2']
            assert isinstance(strides, (tuple, list))
            info = predict_yolov3(anchors, model, img_path, transform, resize, device, visual, strides,
                                  fix_resize, mode, save_path, iou_threshold, conf_threshold)

        elif method == "ssd":  # 单分支
            assert mode in ['v1', 'v2']  # 'v2'效果差
            fh, fw = resize // strides, resize // strides
            info = predict_ssd(anchors, model, img_path, transform, resize, device, visual, fh, fw,
                               fix_resize, save_path, iou_threshold, conf_threshold, focal_loss)

        elif method == "ssdms":  # 多分支
            assert mode in ['v1', 'v2']  # 'v2'效果差
            assert isinstance(strides, (tuple, list))
            info = predict_ssdMS(anchors, model, img_path, transform, resize, device, visual, strides,
                                 fix_resize, save_path, iou_threshold, conf_threshold, focal_loss)

        else:
            raise ("error!!")

        print(info)


class History():
    def __init__(self):
        self.epoch = []
        self.history = {}

    # 打印训练结果信息
    # @staticmethod
    def show_final_history(self):
        # len_ = len(self.history)
        # fig, ax = plt.subplots(1, len_, figsize=(15, 5))
        # for i,(k,v) in enumerate(self.history.items()):
        #     ax[i].set_title(k)
        #     ax[i].plot(self.epoch, v, label=k)
        #     ax[i].legend()
        #     ax[i].grid()

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        for k, v in self.history.items():
            ax.set_title("loss")
            ax.plot(self.epoch, v, label=k)
            ax.legend()
            ax.grid()

        plt.show()


def fit(model, dataset, optim=None, scheduler=None, train=None,
        epochs=50, batch_size=32, muilscale=False, lr=5e-4, weight_decay=5e-5,
        seed=100, device="cuda:0", save_path='./output', mode='centernet',
        print_step=20, reduction="sum", boxes_weight=5.0, max_norm=0.1, method='yolov1',
        focal_loss=False, draw=True, thres=0.3):
    """
    1、method="yolov1"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    2、method="fcos"     mode = 'sigmoid', 'exp'
    3、method="fcosms"   mode = 'sigmoid-v1', 'exp-v1'  ;'v2'效果差
    4、method="yolov2"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    5、method="yolov3"   mode = 'yolov1', 'centernet', 'centernetV2', 'fcos', 'fcosV2'
    6、method="ssd"      mode = 'v1', 'v2' ;'v2'效果差
    7、method="ssdms"    mode = 'v1', 'v2' ;'v2'效果差
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if draw:
        history = History()

    if train is None:
        train = dtrain

    # if transform is None:
    #     transform = T.Compose(
    #         [
    #             # T.Resize((resize,resize)),
    #             # T.RandomApply([T.GaussianBlur(5)]),
    #             # T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)]),
    #             # T.RandomApply([T.ColorJitter(0.125, 0.5, 0.5, 0.05)]),
    #             T.ToTensor(),
    #             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #             # T.RandomErasing()
    #         ])

    if not os.path.exists(save_path): os.makedirs(save_path)

    save_model = os.path.join(save_path, 'weight.pth')
    if os.path.exists(save_model):
        model.load_state_dict(torch.load(save_model, map_location=device))
        print("load weight successful!!!")

    dataLoader = DataLoader(dataset, 1 if muilscale else batch_size, True, collate_fn=dataset.collate_fn)

    if optim is None:
        # optim = torch.optim.RMSprop([parm for parm in model.parameters() if parm.requires_grad],
        #                             lr=lr, weight_decay=weight_decay, momentum=0.95)
        optim = torch.optim.AdamW([parm for parm in model.parameters() if parm.requires_grad],
                                  lr=lr, weight_decay=weight_decay)

    if scheduler is None:
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.8)
        lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lf)

    for epoch in range(epochs):
        loss_dict = train(model, optim, dataLoader, device, epoch, muilscale, mode, print_step,
                          reduction, boxes_weight, max_norm, method, focal_loss, thres)
        scheduler.step()
        torch.save(model.state_dict(), save_model)

        if draw:
            history.epoch.append(epoch)
            for k, v in loss_dict.items():
                if k not in history.history:
                    history.history[k] = []
                history.history[k].append(v)

    if draw:
        history.show_final_history()
