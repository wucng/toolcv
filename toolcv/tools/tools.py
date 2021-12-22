import os
import torch
from torch import nn
import numpy as np
import random
import cv2, math
import PIL.Image
from PIL import Image, ImageDraw
from torchvision.ops.boxes import batched_nms
from torch.nn import functional as F
from itertools import product
from torch.hub import load_state_dict_from_url


def _initParmas(modules, std=0.01, mode='normal'):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  # , nn.Linear
            if mode == 'normal':
                nn.init.normal_(m.weight, std=std)
            elif mode == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, 0, std=std)
        #     if m.bias is not None:
        #         # nn.init.zeros_(m.bias)
        #         nn.init.constant_(m.bias, 0)


def _initParmasV2(modules, std=0.01, mode='normal'):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):  # , nn.Linear
            if mode == 'normal':
                nn.init.normal_(m.weight, std=std)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, a=1)
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def load_weight(model, weight_path='weight.pth', url="", device="cpu"):
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, device)
        else:
            state_dict = load_state_dict_from_url(url, map_location=device)
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in model.state_dict().items():
            if k in state_dict and state_dict[k].numel() == v.numel():
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = v
                print("%s not load weight" % k)
        model.load_state_dict(new_state_dict)
        print("---------load weight successful-----------")
        del state_dict
        del new_state_dict
    except Exception as e:
        print(e)
        print("---------load weight fail-----------")


# ------------------------数据转换------------------------------------
def compute_area_from_polygons(polygons):
    """
    使用该方式计算的结果：15584.661332227523
    coco格式中的 area： 15584.661085606262

    coco数据 `area` 是通过计算边界多边形的面积，如果没有 边界多边形；则使用bounding box的面积替代
    :param polygons: list(list(x,y)) 边界点(边界多边形)
    :return:
    """
    contour = np.array(polygons[0]).reshape(-1, 2).astype(np.float32)

    return abs(cv2.contourArea(contour, True))


def polygons2mask(img_shape, polygons):
    """
    :param img_shape [h,w]
    :param polygons: list(list(x,y)) 边界点(边界多边形)
    :return: 0,1二值图像(0为背景)，大小为图像的大小 channel=1
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    # xy = list(map(tuple, polygons))
    xy = np.array(polygons).reshape(-1, 2).tolist()
    xy = list(map(tuple, xy))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    # visual
    # Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8)).show()
    return mask.astype(np.uint8)


def mask2polygons(mask):
    """
    :param mask: 0,1二值图像(0为背景)，大小为图像的大小 channel=1
    :return: list(list(x,y)) 边界点(边界多边形)
    """
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
    if len(contours) == 3:
        return [contours[1][0][:, 0, :].reshape(-1).tolist()]
    elif len(contours) == 2:
        return [contours[0][0].reshape(-1).tolist()]


def getbbox_from_polygons(polygons):
    """
    :param polygons: list(list(x,y)) 边界点(边界多边形)
    :return: bbox [x_min,y_min,x_max,y_max]
    """
    polygons = np.array(polygons).reshape(-1, 2)

    x_min = np.min(polygons[:, 0])
    y_min = np.min(polygons[:, 1])
    x_max = np.max(polygons[:, 0])
    y_max = np.max(polygons[:, 1])

    return [x_min, y_min, x_max, y_max]


def getbbox_from_mask(mask):
    """
    :param mask: 0,1二值图像(0为背景)，大小为图像的大小 channel=1
    :return: bbox [x_min,y_min,x_max,y_max]
    """
    pos = np.where(mask)  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]


def mask2segmentation(masks, labels):
    """
    :param masks: [(0,1 channel=1的二值图像 大小为图像的大小)，(0,1 channel=1的二值图像 大小为图像的大小)，...] ; list[np.array]
    :param labels: [1,2,...]
    :return: np.array 值取0～num_classes(0为背景),大小为图像的大小，channel=1 图像
    """
    masks = np.stack(masks, 0)  # shape:m,h,w value:0,1
    labels = np.array(labels)  # shape:m
    target = masks * labels[:, None, None]
    return target


def segmentation2mask(segmentation):
    """
    注意：如果 segment有多个相同类别的目标对象，会把它们（类别相同的目标）分到同一个mask上，
    必须结合（类别相同的目标 的）bounding box（扣出对应的部分） 这样才能得到每个目标对应一个mask

    :param segmentation: np.array 值取0～num_classes(0为背景),大小为图像的大小，channel=1 图像
    :return: :param masks: [(0,1 channel=1的二值图像 大小为图像的大小)，(0,1 channel=1的二值图像 大小为图像的大小)，...] ; list[np.array]
             :param labels: [1,2,...]
    """
    # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
    obj_ids = np.unique(segmentation)  # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
    # first id是背景，所以删除它
    obj_ids = obj_ids[1:]  # array([1, 2], dtype=uint8)
    # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
    masks = segmentation == obj_ids[:, None, None]  # shape (2, 536, 559)，2个mask
    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    labels = []

    for i in range(num_objs):  # mask反算对应的bbox
        boxes.append(getbbox_from_mask(masks[i]))
        labels.append(obj_ids[i])

    return masks, boxes, labels


def segmentation2maskV2(segmentation: np.array, bboxs: []):
    """
    注意：如果 segment有多个相同类别的目标对象，会把它们（类别相同的目标）分到同一个mask上，
    必须结合（类别相同的目标 的）bounding box（扣出对应的部分） 这样才能得到每个目标对应一个mask

    :param segmentation: np.array 值取0～num_classes(0为背景),大小为图像的大小，channel=1 图像
    :return: :param masks: [(0,1 channel=1的二值图像 大小为图像的大小)，(0,1 channel=1的二值图像 大小为图像的大小)，...] ; list[np.array]
             :param labels: [1,2,...]
    """

    masks = []
    h, w = segmentation.shape
    for box in bboxs:
        _mask = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        _mask[y1:y2, x1:x2] = (segmentation[y1:y2, x1:x2] > 0).astype(np.uint8)  # 变成 0,1二值图像
        masks.append(_mask)

    return masks


# ---------------------------------------------------------------------------

def glob_format(path, fmt_list=('.jpg', '.jpeg', '.png', ".xml"), base_name=False):
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


def batch(imgs: list, stride=32):
    nums = len(imgs)
    new_imgs = imgs
    # for i in range(nums):
    #     new_imgs.append(resizeMinMax(imgs[i],min_size,max_size))

    # 获取最大的 高和宽
    max_h = max([img.size(1) for img in new_imgs])
    max_w = max([img.size(2) for img in new_imgs])

    # 扩展到 stride的倍数(32倍数 GPU可以加速，且网络的最终stride=32)
    max_h = int(np.ceil(1.0 * max_h / stride) * stride)
    max_w = int(np.ceil(1.0 * max_w / stride) * stride)

    # 初始一个tensor 用于填充
    batch_img = torch.ones([nums, 3, max_h, max_w], device=imgs[0].device) * 114
    for i, img in enumerate(new_imgs):
        c, h, w = img.size()
        batch_img[i, :, :h, :w] = img  # 从左上角往下填充

    return batch_img


def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data, target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list, target_list


def xywh2x1y1x2y2(boxes):
    """
    xywh->x1y1x2y2

    :param boxes: [...,4]
    :return:
    """
    x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
    x2y2 = boxes[..., :2] + boxes[..., 2:] / 2

    return torch.cat((x1y1, x2y2), -1)


def x1y1x2y22xywh(boxes):
    """
    x1y1x2y2-->xywh

    :param boxes: [...,4]
    :return:
    """
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = boxes[..., 2:] - boxes[..., :2]

    return torch.cat((xy, wh), -1)


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


# fastercnn/rfcn
"""
多分支（propose）：
1.每个分支 按分数排序 选择前 2000个
2.合并#1 （5个分支大概 10000个）做nms  阈值0.7 取前 1000个做最后的建议框 

单分支（propose）：
1. 按分数排序 选择前 12000个
2. 做nms  阈值0.7 取前 1000个做最后的建议框 
"""


def nonempty(box, threshold: float = 0.0) -> torch.Tensor:
    """
    Find boxes that are non-empty.
    A box is considered empty, if either of its side is no larger than threshold.

    Returns:
        Tensor:
            a binary vector which represents whether each box is empty
            (False) or non-empty (True).
    """
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def propose_boxes(logits, bbox_reg, anchors, training, targets, device,
                  rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                  rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                  rpn_nms_thresh=0.7
                  ):
    bs = logits.size(0)
    input_h, input_w, scale = targets[0]["resize"]

    # Normalize 0~1
    anchors_x1y1x2y2 = anchors

    # 选择scores最大的前12000个候选区域
    if training:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_train]
    else:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_test]

    # candidate_box_x1y1x2y2 = candidate_box_x1y1x2y2[index]
    anchors_x1y1x2y2 = torch.stack([anchors_x1y1x2y2[i][index[i]] for i in range(bs)], 0)
    logits = torch.stack([logits[i][index[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][index[i]] for i in range(bs)], 0)

    # 先修正anchor 得到建议框
    # anchors修正得到候选框
    anchors_xywh = x1y1x2y22xywh(anchors_x1y1x2y2)
    candidate_box_xywh = torch.zeros_like(anchors_xywh)
    candidate_box_xywh[..., :2] = (anchors_xywh[..., 2:] * bbox_reg[..., :2]) + anchors_xywh[..., :2]
    candidate_box_xywh[..., 2:] = anchors_xywh[..., 2:] * torch.exp(bbox_reg[..., 2:])
    proposal = xywh2x1y1x2y2(candidate_box_xywh).clamp(0., 1.)  # 裁剪到图像内

    # filter empty boxes
    # keep = nonempty(proposal)
    # proposal = proposal

    # nms 过滤
    # keep = batched_nms(candidate_box_x1y1x2y2,logits_score,torch.ones_like(logits_score),rpn_nms_thresh)
    if training:
        keep = torch.stack([batched_nms(proposal[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_train]
                            for i in range(bs)], 0)
    else:
        keep = torch.stack([batched_nms(proposal[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_test]
                            for i in range(bs)], 0)

    anchors_xywh = torch.stack([anchors_xywh[i][keep[i]] for i in range(bs)], 0)
    proposal = torch.stack([proposal[i][keep[i]] for i in range(bs)], 0)

    logits = torch.stack([logits[i][keep[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][keep[i]] for i in range(bs)], 0)

    return anchors_xywh[0], logits[0], bbox_reg[0], proposal[0].detach() * \
           torch.tensor([input_w, input_h, input_w, input_h],
                        dtype=torch.float32, device=device)[None]


def propose_boxes_batch(logits, bbox_reg, anchors, training, targets, device,
                        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                        rpn_nms_thresh=0.7
                        ):
    bs = logits.size(0)

    # Normalize 0~1
    anchors_x1y1x2y2 = anchors

    # 选择scores最大的前12000个候选区域
    if training:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_train]
    else:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_test]

    # candidate_box_x1y1x2y2 = candidate_box_x1y1x2y2[index]
    anchors_x1y1x2y2 = torch.stack([anchors_x1y1x2y2[i][index[i]] for i in range(bs)], 0)
    logits = torch.stack([logits[i][index[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][index[i]] for i in range(bs)], 0)

    # 先修正anchor 得到建议框
    # anchors修正得到候选框
    anchors_xywh = x1y1x2y22xywh(anchors_x1y1x2y2)
    candidate_box_xywh = torch.zeros_like(anchors_xywh)
    candidate_box_xywh[..., :2] = (anchors_xywh[..., 2:] * bbox_reg[..., :2]) + anchors_xywh[..., :2]
    candidate_box_xywh[..., 2:] = anchors_xywh[..., 2:] * torch.exp(bbox_reg[..., 2:])
    proposal = xywh2x1y1x2y2(candidate_box_xywh).clamp(0., 1.)  # 裁剪到图像内

    # filter empty boxes
    # keep = nonempty(proposal)
    # proposal = proposal

    # nms 过滤
    # keep = batched_nms(candidate_box_x1y1x2y2,logits_score,torch.ones_like(logits_score),rpn_nms_thresh)
    if training:
        keep = torch.stack([batched_nms(proposal[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_train]
                            for i in range(bs)], 0)
    else:
        keep = torch.stack([batched_nms(proposal[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_test]
                            for i in range(bs)], 0)

    anchors_xywh = torch.stack([anchors_xywh[i][keep[i]] for i in range(bs)], 0)
    proposal = torch.stack([proposal[i][keep[i]] for i in range(bs)], 0)

    logits = torch.stack([logits[i][keep[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][keep[i]] for i in range(bs)], 0)

    return anchors_xywh, logits, bbox_reg, proposal


def positiveAndNegative(ious, miniBactch=256, rpn=True, logits=None,
                        rpn_fg_iou_thresh=0.7,
                        rpn_bg_iou_thresh=0.3,
                        rpn_positive_fraction=0.5,
                        box_fg_iou_thresh=0.5,
                        box_bg_iou_thresh=0.5,
                        box_positive_fraction=0.25
                        ):
    # 每个anchor与gt对应的iou
    per_anchor_to_gt, per_anchor_to_gt_index = ious.max(1)
    # 与gt对应的最大IOU的anchor
    per_gt_to_anchor, per_gt_to_anchor_index = ious.max(0)

    # 每个anchor对应的gt
    gt_indexs = per_anchor_to_gt_index

    indexs = torch.ones_like(per_anchor_to_gt) * (-1)

    if rpn:
        indexs[per_anchor_to_gt > rpn_fg_iou_thresh] = 1  # 正样本
        indexs[per_anchor_to_gt < rpn_bg_iou_thresh] = 0  # 负样本

        # 随机选择256个anchors,正负样本比例为1:1
        new_positive = int(miniBactch * rpn_positive_fraction)

    else:  # rcnn
        indexs[per_anchor_to_gt >= box_fg_iou_thresh] = 1  # 正样本
        indexs[torch.bitwise_and(per_anchor_to_gt < box_bg_iou_thresh, per_anchor_to_gt > 0.1)] = 0  # 负样本

        #  25% 的前景,75% 的背景
        new_positive = int(miniBactch * box_positive_fraction)

    # 与gt对应的最大IOU的anchor 也是正样本
    # for i, idx in enumerate(per_gt_to_anchor_index):
    #     indexs[idx] = 1
    #     gt_indexs[idx] = i

    # 包括上面
    pred_inds_with_highest_quality, gt_inds_ = (ious == per_gt_to_anchor[None, :]).nonzero(as_tuple=True)
    for i, idx in enumerate(pred_inds_with_highest_quality):
        indexs[idx] = 1
        gt_indexs[idx] = gt_inds_[i]

    # -1 忽略
    # nums_positive = (indexs == 1).sum()
    # nums_negative = (indexs == 0).sum()
    idx_positive = torch.nonzero(indexs == 1).squeeze(-1)
    idx_negative = torch.nonzero(indexs == 0).squeeze(-1)
    nums_positive = len(idx_positive)
    nums_negative = len(idx_negative)

    new_negative = miniBactch - min(nums_positive, new_positive)
    # new_negative = 3*min(nums_positive,new_positive) # 正负样本固定为1:3

    if logits is None:  # 随机选
        if nums_positive < new_positive:
            # 选择负样本个数为
            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1

        elif nums_positive > new_positive:
            posindex = list(range(nums_positive))
            random.shuffle(posindex)
            indexs[idx_positive[posindex[new_positive:]]] = -1

            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1
        else:
            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1

    else:  # 根据负样本的loss 选 从大到小选
        if nums_positive < new_positive:
            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1

        elif nums_positive > new_positive:
            posindex = list(range(nums_positive))
            random.shuffle(posindex)
            indexs[idx_positive[posindex[new_positive:]]] = -1

            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1
        else:
            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1

    return indexs, gt_indexs


"""
多分支：
1.每个分支 随机选择 256个

单分支：
1. 随机选择 256/512
"""


def positiveAndNegativeV2(ious, miniBactch=256, rpn=True, logits=None,
                          rpn_fg_iou_thresh=0.7,
                          rpn_bg_iou_thresh=0.3,
                          rpn_positive_fraction=0.5,
                          box_fg_iou_thresh=0.5,
                          box_bg_iou_thresh=0.5,
                          box_positive_fraction=0.25
                          ):
    # 每个anchor与gt对应的iou
    per_anchor_to_gt, per_anchor_to_gt_index = ious.max(1)
    # 与gt对应的最大IOU的anchor
    per_gt_to_anchor, per_gt_to_anchor_index = ious.max(0)

    # 每个anchor对应的gt
    gt_indexs = per_anchor_to_gt_index

    indexs = torch.ones_like(per_anchor_to_gt) * (-1)

    if rpn:
        indexs[per_anchor_to_gt > rpn_fg_iou_thresh] = 1  # 正样本
        indexs[per_anchor_to_gt < rpn_bg_iou_thresh] = 0  # 负样本

    else:  # rcnn
        indexs[per_anchor_to_gt >= box_fg_iou_thresh] = 1  # 正样本
        indexs[torch.bitwise_and(per_anchor_to_gt < box_bg_iou_thresh, per_anchor_to_gt > 0.1)] = 0  # 负样本

    # 与gt对应的最大IOU的anchor 也是正样本
    # for i, idx in enumerate(per_gt_to_anchor_index):
    #     indexs[idx] = 1
    #     gt_indexs[idx] = i

    # 包括上面
    pred_inds_with_highest_quality, gt_inds_ = (ious == per_gt_to_anchor[None, :]).nonzero(as_tuple=True)
    for i, idx in enumerate(pred_inds_with_highest_quality):
        indexs[idx] = 1
        gt_indexs[idx] = gt_inds_[i]

    positive = (indexs > 0).nonzero(as_tuple=True)[0]
    negative = (indexs == 0).nonzero(as_tuple=True)[0]

    if rpn:
        num_pos = int(miniBactch * rpn_positive_fraction)
    else:
        num_pos = int(miniBactch * box_positive_fraction)

    num_pos = min(positive.numel(), num_pos)
    num_neg = miniBactch - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    pos_idx = positive[perm1]
    neg_idx = negative[perm2]

    indexs.fill_(-1)
    indexs.scatter_(0, pos_idx, 1)
    indexs.scatter_(0, neg_idx, 0)

    return indexs, gt_indexs


def _nms(heat, kernel=3):
    """
    :param heat: torch.tensor [bs,c,h,w]
    :param kernel:
    :return:
    """
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


# -----------------------centernet----------------------------------------------
def heatmap2index(heatmap: torch.tensor, heatmap2: torch.tensor = None, thres=0.5, has_background=False):
    """
    heatmap[0,0,:]只能属于一个类别
    :param heatmap: [bs,h,w,num_classes] or [h,w,num_classes]
    :param heatmap2 : [bs,h,w,num_classes,4] or [h,w,num_classes,4]
    :param threds:
    :param has_background: 是否包含背景
    :return: scores, labels,cycx(中心点坐标)
    """
    scores, labels = heatmap.max(-1)  # [bs,h,w] or [h,w]
    if has_background:
        keep = torch.bitwise_and(scores > thres, labels > 0)  # 0为背景，包括背景
    else:
        keep = scores > thres
    scores, labels = scores[keep], labels[keep]
    cycx = torch.nonzero(keep)
    if heatmap2 is not None:
        heatmap2 = heatmap2[keep, labels]

    return scores, labels, cycx, keep, heatmap2


def heatmap2indexV2(heatmap: torch.tensor, heatmap2: torch.tensor = None, thres=0.5, has_background=False, topK=5):
    """
    heatmap[0,0,:]可以属于多个类别
    :param heatmap: [bs,h,w,num_classes] or [h,w,num_classes]
    :param heatmap2 : [bs,h,w,num_classes,4] or [h,w,num_classes,4]
    """
    scores, labels = heatmap.topk(topK, -1)

    if heatmap2 is not None:
        h, w, c = labels.shape
        new_heatmap2 = torch.zeros((h, w, c, heatmap2.shape[-1]), device=heatmap2.device)
        # for i in range(h):
        #     for j in range(w):
        #         for k in range(c):
        #             l = labels[i,j,k]
        #             new_heatmap2[i,j,k,:] = heatmap2[i,j,l,:]
        # for i in range(h):
        #     for j in range(w):
        for i, j in product(range(h), range(w)):
            new_heatmap2[i, j] = heatmap2[i, j, labels[i, j]]

    if has_background:
        keep = torch.bitwise_and(scores > thres, labels > 0)  # 0为背景，包括背景
    else:
        keep = scores > thres
    scores, labels = scores[keep], labels[keep]
    cycx = torch.nonzero(keep)[..., :2]
    if heatmap2 is not None:
        new_heatmap2 = new_heatmap2[keep]
        heatmap2 = new_heatmap2

    return scores, labels, cycx, keep, heatmap2


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


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    """
    :param heatmap: [128,128]
    :param center: [x,y]
    :param sigma: int
    :return:
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


# ----------------------------------------------------


"""centernet 的方式标记 heatmap (只标记中心点及其附近点)"""


def drawHeatMapV1(hm: torch.tensor, box: torch.tensor, device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    hm = hm.cpu().numpy()
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    cx, cy = (x2 + x1) / 2., (y2 + y1) / 2.
    cx, cy = cx.int().item(), cy.int().item()
    h, w = h.item(), w.item()
    radius = gaussian_radius((h, w))
    # radius = math.sqrt(h*w) # 不推荐
    radius = max(1, int(radius))
    # hm = torch.from_numpy(draw_msra_gaussian(hm, (cx,cy), radius)).to(device) # 等价于 drawHeatMapV2
    hm = torch.from_numpy(draw_umich_gaussian(hm, (cx, cy), radius)).to(device)
    return hm


"""参考 centernet 的方式标记 heatmap 并改进"""


def drawHeatMapV2(hm: torch.tensor, box: torch.tensor, device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    # hm = hm.cpu().numpy()
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    cx, cy = (x2 + x1) / 2., (y2 + y1) / 2.
    cx, cy = cx.int().item(), cy.int().item()
    h, w = h.item(), w.item()
    radius = gaussian_radius((h, w))
    # radius = math.sqrt(h*w)# 不推荐
    radius = max(1, int(radius))

    hm = draw_gaussian2(hm, (cy, cx), radius)

    return hm


"""使用fcos 方式标记 heatmap，即框内所有点都做标记"""


def drawHeatMapV3(hm: torch.tensor, box: torch.tensor, device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上 (注： box 已经按照从大到小排序好)
    :return:
    """
    # hm = hm.cpu().numpy()
    x1, y1, x2, y2 = box
    # int_x1 = x1.ceil().int().item()
    # int_y1 = y1.ceil().int().item()
    # int_x2 = x2.floor().int().item()
    # int_y2 = y2.floor().int().item()
    # h,w = y2-y1,x2-x1
    # h,w = h.floor().int().item(),w.floor().int().item()
    int_x1 = x1.floor().int().item()  # +w//4
    int_y1 = y1.floor().int().item()  # +h//4
    int_x2 = x2.ceil().int().item()  # -w//4
    int_y2 = y2.ceil().int().item()  # -h//4

    # 中心点
    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.
    cx, cy = cx.int().item(), cy.int().item()

    fh, fw = hm.shape
    for y, x in product(range(int_y1, int_y2), range(int_x1, int_x2)):
        if x <= 0 or y <= 0 or x >= fw or y >= fh: continue
        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y

        if l <= 0 or t <= 0 or r <= 0 or b <= 0: continue

        if x == cx and y == cy:
            centerness = 1
        else:
            # centerness = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
            centerness = (min(l, r) / max(l, r)) * (min(t, b) / max(t, b))
            # centerness *= np.exp(centerness-1)
        hm[y, x] = centerness

    return hm


def drawHeatMapV0(hm: torch.tensor, box: torch.tensor, device="cpu", thred_radius=1):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    hm = hm.cpu().numpy()
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    cx, cy = (x2 + x1) / 2., (y2 + y1) / 2.
    cx, cy = cx.int().item(), cy.int().item()
    h, w = h.item(), w.item()
    radius = int(gaussian_radius((h, w)))
    # radius = math.sqrt(h*w) # 不推荐
    # radius = max(1, radius)
    if radius > thred_radius:
        hm = torch.from_numpy(draw_umich_gaussian(hm, (cx, cy), radius)).to(device)
    else:
        radius = thred_radius
        hm = torch.from_numpy(draw_msra_gaussian(hm, (cx, cy), radius)).to(device)  # 等价于 drawHeatMapV2
    return hm


"""取综合结果
# 推荐使用
"""


def drawHeatMap(hm: torch.tensor, box: torch.tensor, device="cpu", thred_radius=1, dosegm=False):
    """
   :param hm: torch.tensor  shape [128,128]
   :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上 (注： box 已经按照从大到小排序好)
   :return:
   """
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    h, w = h.item(), w.item()
    radius = int(gaussian_radius((h, w)))

    if dosegm:
        if radius <= thred_radius:
            return drawHeatMapV3(hm, box, device)
        else:
            return drawHeatMapV1(hm, box, device)
    else:
        return drawHeatMapV1(hm, box, device)


if __name__ == "__main__":
    torch.manual_seed(100)
    conv1 = nn.Conv2d(1,1,3,2,1,bias=True)
    # print(conv1.bias is None)
    # _initParmasV2(conv1.modules(),mode="kaiming_normal")
    _initParmas(conv1.modules(),mode="kaiming")
    print(conv1.weight)

    """
    tensor([[[[ 0.0195, -0.0071, -0.0022],
          [ 0.0065,  0.0099,  0.0034],
          [-0.0155, -0.0099, -0.0192]]]], requires_grad=True)
    
    tensor([[[[-0.4411, -0.4368,  0.2720],
          [ 0.2445,  0.3321, -0.0943],
          [ 0.4635,  0.5738,  0.2962]]]], requires_grad=True)
    
    tensor([[[[ 0.9206, -0.3360, -0.1016],
          [ 0.3085,  0.4650,  0.1579],
          [-0.7304, -0.4684, -0.9054]]]], requires_grad=True)
    """