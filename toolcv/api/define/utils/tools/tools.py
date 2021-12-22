import torch
from torchvision.ops import batched_nms
import numpy as np

from toolcv.tools.utils import xywh2x1y1x2y2, _nms


def top_k(center_heatmap_preds, k=100, dim=-1):
    """
    center_heatmap_preds: 已经使用过 sigmoid
    """
    # 选取前100 top_100
    bs, c, h, w = center_heatmap_preds.shape
    center_heatmap_preds = center_heatmap_preds.contiguous().view(bs, -1)
    batch_score, index = torch.topk(center_heatmap_preds, k=min(k, center_heatmap_preds.size(-1)), dim=dim)
    batch_label = index // (h * w)
    batch_index = index % (h * w)
    batch_ys = batch_index // w
    batch_xs = batch_index % w

    return batch_index, batch_label, batch_score, batch_ys, batch_xs


def transpose_and_gather_feat(feat, batch_index):
    """
    feat: [bs,c,h,w]
    """
    bs, c, h, w = feat.shape
    feat = feat.contiguous().view(bs, c, -1).permute(0, 2, 1)

    return torch.stack([feat[i][batch_index[i]] for i in range(bs)], 0)


# shape [bs,c,h,w]
def get_bboxes(center_heatmap_preds, wh_preds, offset_preds, with_nms,
               iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    center_heatmap_preds = _nms(center_heatmap_preds, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(center_heatmap_preds, k=k)
    wh_preds = transpose_and_gather_feat(wh_preds, batch_index)
    offset_preds = transpose_and_gather_feat(offset_preds, batch_index)

    img_h, img_w = in_shape
    bs, c, h, w = center_heatmap_preds.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score, batch_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


# shape [bs,c,h,w]
def get_bboxesv2(centerness_pred, cls_pred, wh_preds, offset_preds, with_nms,
                 iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    centerness_pred = _nms(centerness_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(centerness_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    wh_preds = transpose_and_gather_feat(wh_preds, batch_index)
    offset_preds = transpose_and_gather_feat(offset_preds, batch_index)

    img_h, img_w = in_shape
    bs, c, h, w = centerness_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score, cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


# shape [bs,c,h,w]
def get_bboxesv3(anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                 iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    """动态选择 anchor"""

    fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    wh_cls = transpose_and_gather_feat(wh_cls, batch_index)
    bs, h, c = wh_cls.shape
    wh_cls = wh_cls.contiguous().view(bs, h, c // 2, 2).softmax(2)
    wh_cls_score, wh_cls_index = wh_cls.max(2)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            iw, ih = wh_cls_index[i, j]
            _anchors.append([anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    offset_preds = bbox_offset[..., :2]
    wh_preds = bbox_offset[..., 2:]
    wh_preds = wh_preds.exp() * anchors
    # 恢复到heatmap上
    wh_preds[..., 0] *= fw
    wh_preds[..., 1] *= fh

    img_h, img_w = in_shape
    bs, c, h, w = object_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score * wh_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


def get_bboxesv3_2(anchors, object_pred, cls_pred, bbox_offset, wh_cls, keypoints_pred, with_nms,
                   iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    """动态选择 anchor"""

    fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    wh_cls = transpose_and_gather_feat(wh_cls, batch_index)
    bs, h, c = wh_cls.shape
    wh_cls = wh_cls.contiguous().view(bs, h, c // 2, 2).softmax(2)
    wh_cls_score, wh_cls_index = wh_cls.max(2)
    keypoints_pred = transpose_and_gather_feat(keypoints_pred, batch_index)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            iw, ih = wh_cls_index[i, j]
            _anchors.append([anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    offset_preds = bbox_offset[..., :2]
    wh_preds = bbox_offset[..., 2:]
    wh_preds = wh_preds.exp() * anchors
    # 恢复到heatmap上
    wh_preds[..., 0] *= fw
    wh_preds[..., 1] *= fh

    img_h, img_w = in_shape
    bs, c, h, w = object_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score * wh_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keypoints = keypoints_pred[keep]
    tmp = torch.zeros([*keypoints.shape, 3], device=keypoints.device)
    tmp[..., -1] = (keypoints > 0).float()
    tmp[..., 1] = keypoints
    tmp[..., 0] = keypoints % 56

    heatmap_size = 56
    offset_x = boxes[:, 0]
    offset_y = boxes[:, 1]
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]
    scale_x = heatmap_size / boxes_w
    scale_y = heatmap_size / boxes_h
    tmp[..., 0] = tmp[..., 0] / scale_x + offset_x
    tmp[..., 1] = tmp[..., 1] / scale_y + offset_y
    for i in range(len(boxes_w)):
        tmp[i, :, 0] = tmp[i, :, 0].clamp(0, boxes_w[i])
        tmp[i, :, 1] = tmp[i, :, 1].clamp(0, boxes_h[i])

    keypoints = tmp

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keypoints = keypoints[keep]
    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]
        keypoints[..., :2] -= torch.tensor(padding[:2], device=boxes.device)[None]
        keypoints[..., :2] /= torch.tensor(scale_factors[:2], device=boxes.device)[None]

    return boxes, scores, labels, keypoints


def get_bboxes_keypoints(anchors, object_pred, cls_pred, bbox_offset, wh_cls, keypoints_logit, keypoints_offset,
                         with_nms, iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    """动态选择 anchor"""

    fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    wh_cls = transpose_and_gather_feat(wh_cls, batch_index)
    bs, h, c = wh_cls.shape
    wh_cls = wh_cls.contiguous().view(bs, h, c // 2, 2).softmax(2)
    wh_cls_score, wh_cls_index = wh_cls.max(2)
    keypoints_logit = transpose_and_gather_feat(keypoints_logit, batch_index)
    keypoints_offset = transpose_and_gather_feat(keypoints_offset, batch_index)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            iw, ih = wh_cls_index[i, j]
            _anchors.append([anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    offset_preds = bbox_offset[..., :2]
    wh_preds = bbox_offset[..., 2:]
    wh_preds = wh_preds.exp() * anchors
    # 恢复到heatmap上
    wh_preds[..., 0] *= fw
    wh_preds[..., 1] *= fh

    img_h, img_w = in_shape
    bs, c, h, w = object_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)
    bs, n, c = keypoints_offset.shape
    # keypoints
    keypoints_offset = keypoints_offset.contiguous().view(-1, n, 17, 2)
    keypoints_offset[..., 0] += batch_xs[..., None]
    keypoints_offset[..., 1] += batch_ys[..., None]
    keypoints_offset[..., 0] *= img_w / w
    keypoints_offset[..., 1] *= img_h / h
    keypoints_v = (keypoints_logit > 0.5).float()
    keypoints_offset[..., 0] *= keypoints_v
    keypoints_offset[..., 1] *= keypoints_v
    # keypoints = torch.stack((keypoints_offset[..., 0], keypoints_offset[..., 1], keypoints_v, keypoints_logit), -1)
    keypoints = torch.stack((keypoints_offset[..., 0], keypoints_offset[..., 1], keypoints_v), -1)

    # fillter
    scores, labels = batch_score * cls_score * wh_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keypoints = keypoints[keep]

    # 裁剪到对应的boxes内
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        keypoints[i, :, 0] = keypoints[i, :, 0].clamp(x1, x2)
        keypoints[i, :, 1] = keypoints[i, :, 1].clamp(y1, y2)

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keypoints = keypoints[keep]
    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]
        keypoints[..., :2] -= torch.tensor(padding[:2], device=boxes.device)[None]
        keypoints[..., :2] /= torch.tensor(scale_factors[:2], device=boxes.device)[None]

    return boxes, scores, labels, keypoints


def get_bboxesv4(anchors, object_pred, cls_pred, bbox_offset, wh_cls, with_nms,
                 iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    """动态选择 anchor"""

    fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    wh_cls = transpose_and_gather_feat(wh_cls, batch_index)
    bs, h, c = wh_cls.shape
    wh_cls = wh_cls.contiguous().view(bs, h, c // 2, 2).softmax(2)
    wh_cls_score, wh_cls_index = wh_cls.max(2)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            iw, ih = wh_cls_index[i, j]
            _anchors.append([anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    offset_preds = bbox_offset[..., :2]
    offset_preds *= anchors

    wh_preds = bbox_offset[..., 2:]
    wh_preds = wh_preds.exp() * anchors
    # 恢复到heatmap上
    wh_preds[..., 0] *= fw
    wh_preds[..., 1] *= fh

    img_h, img_w = in_shape
    bs, c, h, w = object_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score * wh_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


def get_bboxesv5(anchors, object_pred, cls_pred, bbox_offset, bbox_cls, with_nms,
                 iou_threshold, conf_threshold, scale_factors, padding, in_shape,
                 to_img=True, k=100, do_bbox_offset=True):
    """动态选择 anchor 和 中心点
        do_bbox_offset = True   (1、bbox粗略分类 2、对bbox进一步做回归)
        do_bbox_offset = False  (1、bbox精细分类 只使用分类方式代替回归)
    """

    # fh, fw = object_pred.shape[-2:]
    object_pred = _nms(object_pred, 3)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_k(object_pred, k=k)
    cls_pred = transpose_and_gather_feat(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    if do_bbox_offset: bbox_offset = transpose_and_gather_feat(bbox_offset, batch_index)
    bbox_cls = transpose_and_gather_feat(bbox_cls, batch_index)
    bs, h, c = bbox_cls.shape
    bbox_cls = bbox_cls.contiguous().view(bs, h, c // 4, 4).softmax(2)
    bbox_cls_score, bbox_cls_index = bbox_cls.max(2)

    # 找到对应的anchor
    anchors_batch = []
    for i in range(bs):
        _anchors = []
        for j in range(h):
            ix, iy, iw, ih = bbox_cls_index[i, j]
            _anchors.append([anchors[ix], anchors[iy], anchors[iw], anchors[ih]])
        anchors_batch.append(_anchors)

    anchors = torch.tensor(anchors_batch).to(cls_pred.device)

    # decode
    if do_bbox_offset:
        bbox_offset[..., :2] *= anchors[..., 2:]
        bbox_offset[..., :2] += anchors[..., :2]
        bbox_offset[..., 2:] = bbox_offset[..., 2:].exp()
        bbox_offset[..., 2:] *= anchors[..., 2:]
    else:
        bbox_offset = anchors
    # 恢复到输入图像上
    img_h, img_w = in_shape
    bbox_offset *= torch.tensor([img_w, img_h, img_w, img_h], device=bbox_offset.device)[None, None]

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(bbox_offset)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score * bbox_cls_score.prod(-1), cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


# -----------------------------------------------------------
def top_kv3(center_heatmap_preds, k=100, dim=-1):
    """
    center_heatmap_preds: 已经使用过 sigmoid
    """
    # 选取前100 top_100
    bs, h, w, a, c = center_heatmap_preds.shape
    center_heatmap_preds = center_heatmap_preds.contiguous().view(bs, -1)
    batch_score, index = torch.topk(center_heatmap_preds, k=min(k, center_heatmap_preds.size(-1)), dim=dim)
    batch_ys = index // (w * a * c)
    batch_xs = index % (w * a * c) // (a * c)
    batch_a = index % (w * a * c) % (a * c) // c
    batch_label = index % (w * a * c) % (a * c) % c
    batch_index = index

    return batch_index, batch_label, batch_score, batch_ys, batch_xs


def transpose_and_gather_featv3(feat, batch_index):
    """
    feat: [bs, h, w, a, c]
    """
    bs, h, w, a, c = feat.shape
    feat = feat.contiguous().view(bs, -1, c)

    return torch.stack([feat[i][batch_index[i]] for i in range(bs)], 0)


# for yolov3 shape [bs,h,w,a,c]
def get_bboxes_yolov3(centerness_pred, cls_pred, wh_preds, offset_preds, with_nms,
                      iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    bs, h, w, a, c = centerness_pred.shape
    centerness_pred = centerness_pred.contiguous().view(bs, h, w, -1).permute(0, 3, 1, 2)
    centerness_pred = _nms(centerness_pred, 3)
    centerness_pred = centerness_pred.permute(0, 2, 3, 1).contiguous().view(bs, h, w, a, c)
    # 选取前100 top_100
    batch_index, batch_label, batch_score, batch_ys, batch_xs = top_kv3(centerness_pred, k=k)
    cls_pred = transpose_and_gather_featv3(cls_pred, batch_index)
    cls_score, cls_label = cls_pred.max(-1)
    wh_preds = transpose_and_gather_featv3(wh_preds, batch_index)
    offset_preds = transpose_and_gather_featv3(offset_preds, batch_index)

    img_h, img_w = in_shape
    bs, h, w, a, c = centerness_pred.shape
    stride = torch.tensor([img_w / w, img_h / h], device=wh_preds.device)[None, None]
    wh = wh_preds * stride
    xy = (offset_preds + torch.stack((batch_xs, batch_ys), -1)) * stride

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(torch.cat((xy, wh), -1))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    scores, labels = batch_score * cls_score, cls_label

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


# for ssd shape [bs,h*w*a,c]
def get_bboxes_ssd(centerness_preds, cls_preds, reg_preds, with_nms,
                   iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    if centerness_preds is None:
        bs, h, c = cls_preds.shape
        batch_score, index = torch.topk(cls_preds.view(bs, -1), k=k, dim=-1)
        batch_index = index // c
        batch_label = index % c

        scores, labels = batch_score, batch_label
    else:
        bs, h = centerness_preds.shape
        batch_score, index = torch.topk(centerness_preds.view(bs, -1), k=k, dim=-1)
        batch_index = index

        cls_preds = torch.stack([cls_preds[i][batch_index[i]] for i in range(bs)], 0)
        cls_scores, cls_labels = cls_preds.max(-1)

        scores, labels = batch_score * cls_scores, cls_labels

    reg_preds = torch.stack([reg_preds[i][batch_index[i]] for i in range(bs)], 0)

    # 恢复到输入图像上
    img_h, img_w = in_shape
    reg_preds[..., [0, 2]] *= img_w
    reg_preds[..., [1, 3]] *= img_h

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(reg_preds)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


def get_bboxes_ssdv2(featmap_sizes, num_anchors, centerness_preds, cls_preds, reg_preds, with_nms,
                     iou_threshold, conf_threshold, scale_factors, padding, in_shape, to_img=True, k=100):
    if centerness_preds is None:
        bs, h, c = cls_preds.shape
        start = 0
        end = 0
        cls_pred_list = []
        for i in range(len(num_anchors)):
            num_anchor = num_anchors[i]
            fh, fw = featmap_sizes[i]
            end = start + fh * fw * num_anchor
            cls_pred = cls_preds[:, start:end, :]
            cls_pred = cls_pred.contiguous().view(bs, fh, fw, -1).permute(0, 3, 1, 2)
            cls_pred = _nms(cls_pred, 3)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, c)
            cls_pred_list.append(cls_pred)
            start = end

        cls_preds = torch.cat(cls_pred_list, 1)

        batch_score, index = torch.topk(cls_preds.view(bs, -1), k=k, dim=-1)
        batch_index = index // c
        batch_label = index % c

        scores, labels = batch_score, batch_label
    else:
        bs, h = centerness_preds.shape
        start = 0
        end = 0
        centerness_preds_list = []
        for i in range(len(num_anchors)):
            num_anchor = num_anchors[i]
            fh, fw = featmap_sizes[i]
            end = start + fh * fw * num_anchor
            centerness_pred = centerness_preds[:, start:end]
            centerness_pred = centerness_pred.contiguous().view(bs, fh, fw, -1).permute(0, 3, 1, 2)
            centerness_pred = _nms(centerness_pred, 3)
            centerness_pred = centerness_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1)
            centerness_preds_list.append(centerness_pred)
            start = end

        centerness_preds = torch.cat(centerness_preds_list, 1)

        batch_score, index = torch.topk(centerness_preds.view(bs, -1), k=k, dim=-1)
        batch_index = index

        cls_preds = torch.stack([cls_preds[i][batch_index[i]] for i in range(bs)], 0)
        cls_scores, cls_labels = cls_preds.max(-1)

        scores, labels = batch_score * cls_scores, cls_labels

    reg_preds = torch.stack([reg_preds[i][batch_index[i]] for i in range(bs)], 0)

    # 恢复到输入图像上
    img_h, img_w = in_shape
    reg_preds[..., [0, 2]] *= img_w
    reg_preds[..., [1, 3]] *= img_h

    # xywh 2 x1y1x2y2
    boxes = xywh2x1y1x2y2(reg_preds)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_h - 1)

    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if with_nms:
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if to_img:
        # 缩放到原始图像上
        boxes -= torch.tensor(padding, device=boxes.device)[None]
        boxes /= torch.tensor(scale_factors, device=boxes.device)[None]

    return boxes, scores, labels


def grid(rows, cols):
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    grid_x, grid_y = np.meshgrid(x, y)
    xy = np.stack((grid_x, grid_y), -1)
    return xy


def grid_torch(rows, cols):
    x = torch.arange(0, cols)
    y = torch.arange(0, rows)
    grid_y, grid_x = torch.meshgrid(y, x)
    xy = torch.stack((grid_x, grid_y), -1)
    return xy


# from torchvision.models.detection.roi_heads import keypoints_to_heatmap,heatmaps_to_keypoints,keypointrcnn_loss
# from toolsmall.tools.other import keypoints_to_heatmap

def keypoints_to_heatmap(keypoints, rois, heatmap_size, to_htmap=False):
    """
    keypoints:[-1,17,3]
    rois:[-1,4]
    heatmap_size: int 56
    """
    # type: # (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    if not to_htmap:
        lin_ind = y * heatmap_size + x
        heatmaps = lin_ind * valid

        # heatmaps 是 heat_map 的稀疏表示（按行展开）
        # heatmaps [17,] ;valid [17,]  heatmaps//heatmap_size=y; heatmaps%heatmap_size=x;heatmaps>0 为 v
        return heatmaps, valid
    else:
        heat_map = torch.zeros([17, heatmap_size, heatmap_size])
        for i, _y, _x, _v in enumerate(zip(y, x, valid)):
            heat_map[i, _y, _x] = _v

        # heat_map:[17,56,56]
        # heat_map->[17,-1]  heat_map.topk(1,-1) 返回 值v及位置p  p//heatmap_size=y ;p%heatmap_size=x
        return heat_map, valid
