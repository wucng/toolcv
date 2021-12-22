"""
https://github.com/open-mmlab/mmdetection
"""
# import mmcv
# import torch.nn as nn
# from mmdet.models.losses import VarifocalLoss

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import warnings

__all__ = ["labelsmooth", "labelsmooth_focal", "selecterrLoss", "oneshotLoss", "tripletLoss", "distilLoss",
           "LabelSmooth", "LabelSmoothFocal", "OneShotLoss", "TripletLoss", "DistilLoss"]


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class FocalLoss2(nn.Module):
    """
    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    """

    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def softmax(pred):
    tp = pred.exp()
    return tp / tp.sum(-1, keepdims=True)


def softmaxv2(pred):
    """防止数据溢出"""
    pred = pred - pred.max(-1, keepdims=True)[0]
    tp = pred.exp()
    return tp / tp.sum(-1, keepdims=True)


def disti_softmax(pred, T=2):
    """
    # pred = torch.randn([2, 5])
    pred = torch.tensor([[-0.1, 3.2, 0.3], [0.2, -0.1, 4.9]])

    print(softmax(pred))
    print(softmaxv2(pred))
    print(disti_softmax(pred, T=3))
    """
    pred = pred / T

    return softmaxv2(pred)


def disti_softmaxv2(pred, T=2, log=False):
    return F.log_softmax(pred / T, -1) if log else F.softmax(pred / T, -1)


def labelsmooth(pred, target, weight=None, esp=0.1, reduction="mean"):
    """
    Example:
        >>> from torchvision.models import AlexNet

        >>> device = "cuda:0"
        >>> num_classes = 3
        >>> model = AlexNet(num_classes).to(device)

        >>> img = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred = model(img)

        >>> target = torch.tensor([[0, 1], [0, 1]]).to(device)
        >>> weight = torch.tensor([[0.86, 0.14], [0.14, 0.86]]).to(device)
        >>> # target+weight 转成 one-hot 标签为 torch.tensor([[0.86,0.14,0],[0.14,0.86,0]])
        >>> loss = labelsmooth(pred, target, weight=weight)
        >>> # 等价于
        >>> target = torch.tensor([[0.86,0.14,0],[0.14,0.86,0]]).to(device)
        >>> loss = labelsmooth(pred, target)

        >>> # label smooth
        >>> target = torch.tensor([0, 1]).to(device)
        >>> loss = labelsmooth(pred, target,esp=0.1)
        >>> 对应的one-hot标签为 [[0.9,0.05,0.05],[0.05,0.9,0.05]]

    """
    if target.ndim == pred.ndim:
        if weight is None:
            # loss = F.kl_div(disti_softmaxv2(pred,log=True), target, reduction='none').sum(-1)
            # loss = F.binary_cross_entropy(pred.sigmoid(), target, reduction='none').sum(-1)

            # 使用 label smooth
            log_pred = F.log_softmax(pred, -1)
            loss = -(log_pred * target).sum(-1)

        else:
            loss = torch.zeros([1], device=pred.device, requires_grad=True)
            for i in range(weight.size(-1)):
                loss = loss + F.cross_entropy(pred, target[:, i], reduction="none") * weight[:, i]
            loss = loss.sum(-1)

    else:
        # label smooth
        num_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, -1)
        loss = -(log_pred).sum(-1) * esp / num_classes + (1 - esp) * F.nll_loss(log_pred, target,
                                                                                reduction="none")
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class LabelSmooth:
    def __init__(self, esp=0.1, reduction="mean"):
        self.esp = esp
        self.reduction = reduction

    def __call__(self, pred, target, weight=None):
        return labelsmooth(pred, target, weight, self.esp, self.reduction)


def labelsmooth_focal(pred, target, cls_weight=None, esp=0.1, gamma=2, reduction="mean"):
    """
    :param pred: [b,num_classes]
    :param target:[b,] or [b,num_classes]
    :param cls_weight: list ,len(cls_weight) = num_classes  对应每个类别的权重
    :param gamma: 参考 focal loss 越容易训练的 则对应的 分数越大 ，而(1-score)**gamma 则会降低其权重
    :return:
    """
    num_classes = pred.size(-1)
    if target.ndim == 1:
        target = F.one_hot(target, num_classes)
        if esp > 0:  # label smooth
            target = (1 - target) * esp / (num_classes - 1) + target * (1 - esp)
    if cls_weight is None:
        cls_weight = torch.ones([1, num_classes], device=pred.device)
    else:
        assert len(cls_weight) == num_classes
        cls_weight = torch.tensor([cls_weight], device=pred.device)

    log_pred = -pred.log_softmax(-1)

    if gamma > 0:
        loss = target * log_pred * cls_weight * (1 - pred.detach().softmax(-1)) ** gamma
    else:
        loss = target * log_pred * cls_weight

    loss = loss.sum(-1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class LabelSmoothFocal:
    def __init__(self, esp=0.1, gamma=2, reduction="mean"):
        self.esp = esp
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, target, cls_weight=None):
        return labelsmooth_focal(pred, target, cls_weight, self.esp, self.gamma, self.reduction)


def selecterrLoss(pred, label2, label, margin=2, reduction="mean", use_focal=False):
    """90%选择正确标签 10%选择错误标签"""
    same_label = (label2 == label).float()
    if use_focal:
        loss_ = labelsmooth_focal(pred, label, reduction="none")
    else:
        loss_ = labelsmooth(pred, label, reduction='none')
    loss = same_label * loss_ + ((1 - same_label) * torch.clamp(margin - loss_, 0.0))

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def oneshot_loss(outputs1, outputs2, same_classes, margin=2, reduction='sum'):
    # 结合one-shot-learning loss : 类内小 内间大
    # # same_classes 不同为0 相同为1
    euclidean_distance = torch.nn.functional.pairwise_distance(outputs1, outputs2)
    loss_contrastive = (same_classes) * torch.pow(euclidean_distance, 2) + \
                       (1 - same_classes) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    if reduction == 'sum':
        loss_contrastive = loss_contrastive.sum()
    elif reduction == 'mean':
        loss_contrastive = loss_contrastive.mean()

    return loss_contrastive


def oneshotLoss(pred1, pred2, target1, target2, alpha=0.5, mode="kl_div", margin=2, reduction='mean', use_focal=False):
    """
    Example:
        >>> from torchvision.models import AlexNet

        >>> device = "cuda:0"
        >>> num_classes = 3
        >>> model = AlexNet(num_classes).to(device)

        >>> img = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred1 = model(img)
        >>> img = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred2 = model(img)

        >>> target1 = torch.tensor([0, 1]).to(device)
        >>> target2 = torch.tensor([0, 1]).to(device)

        >>> loss = oneshotloss(pred1, pred2, target1, target2)
    """
    if use_focal:
        loss = labelsmooth_focal(pred1, target1, reduction=reduction) + \
               labelsmooth_focal(pred2, target2, reduction=reduction)
    else:
        loss = labelsmooth(pred1, target1, reduction=reduction) + labelsmooth(pred2, target2, reduction=reduction)
    same_classes = (target1 == target2).float()
    if mode == "kl_div":
        kl_div = F.kl_div(disti_softmaxv2(pred1, log=True), disti_softmaxv2(pred2), reduction='none').sum(-1)
        loss_kl = (same_classes * kl_div + (1 - same_classes) * torch.clamp(margin - kl_div, min=0.0))
        if reduction == "mean":
            loss_kl = loss_kl.mean()
        elif reduction == "sum":
            loss_kl = loss_kl.sum()

        loss = loss * alpha + (1 - alpha) * loss_kl

    else:
        # dist2 = F.pairwise_distance(c, d, p=2)#pytorch求欧氏距离
        # loss = loss * alpha + (1 - alpha) * (
        #         F.pairwise_distance(pred1, pred2, p=2) * (2 * (target1 == target2) - 1)).mean()

        euclidean_distance = F.pairwise_distance(pred1, pred2)
        loss_contrastive = (same_classes) * euclidean_distance + \
                           (1 - same_classes) * torch.clamp(margin - euclidean_distance, min=0.0)

        if reduction == "mean":
            loss_contrastive = loss_contrastive.mean()
        elif reduction == "sum":
            loss_contrastive = loss_contrastive.sum()

        loss = loss * alpha + (1 - alpha) * loss_contrastive

    return loss


class OneShotLoss:
    def __init__(self, alpha=0.5, mode="kl_div", margin=2, reduction='mean'):
        self.alpha = alpha
        self.mode = mode
        self.margin = margin
        self.reduction = reduction

    def __call__(self, pred1, pred2, target1, target2):
        return oneshotLoss(pred1, pred2, target1, target2, self.alpha, self.mode, self.margin, self.reduction)


def tripletLoss(pred1, pred2, pred3, target1, target2, target3, alpha=0.5, mode="kl_div",
                margin=2, reduction='mean', use_focal=False):
    """
    Example:
        >>> from torchvision.models import AlexNet

        >>> device = "cuda:0"
        >>> num_classes = 3
        >>> model = AlexNet(num_classes).to(device)

        >>> img1 = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred1 = model(img1)

        >>> img2 = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred2 = model(img2)

        >>> img3 = torch.ones([2, 3, 224, 224]).to(device)
        >>> pred3 = model(img3)

        >>> target1 = torch.tensor([0, 1]).to(device)
        >>> target2 = torch.tensor([0, 1]).to(device)
        >>> target3 = torch.tensor([1, 0]).to(device)

        >>> loss = tripletLoss(pred1, pred2,pred3, target1, target2,target3)

        >>> loss.backward()

        >>> print(loss.item())
    """
    assert (target1 == target2).all()
    assert (target1 != target3).all()
    if use_focal:
        loss = labelsmooth_focal(pred1, target1, reduction=reduction) + \
               labelsmooth_focal(pred2, target2, reduction=reduction) + \
               labelsmooth_focal(pred3, target3, reduction=reduction)
    else:
        loss = labelsmooth(pred1, target1, reduction=reduction) + \
               labelsmooth(pred2, target2, reduction=reduction) + \
               labelsmooth(pred3, target3, reduction=reduction)

    if mode == "kl_div":

        loss_kl = F.kl_div(disti_softmaxv2(pred1, log=True), disti_softmaxv2(pred2), reduction='none').sum(-1) + \
                  torch.clamp(margin - F.kl_div(disti_softmaxv2(pred1, log=True),
                                                disti_softmaxv2(pred3), reduction='none').sum(-1), min=0.0) + \
                  torch.clamp(margin - F.kl_div(disti_softmaxv2(pred2, log=True),
                                                disti_softmaxv2(pred3), reduction='none').sum(-1), min=0.0)

        if reduction == "mean":
            loss_kl = loss_kl.mean()
        elif reduction == "sum":
            loss_kl = loss_kl.sum()

        loss = loss * alpha + (1 - alpha) * loss_kl

    else:
        loss_contrastive = F.pairwise_distance(pred1, pred2) + \
                           torch.clamp(margin - F.pairwise_distance(pred1, pred3), min=0.0)

        if reduction == "mean":
            loss_contrastive = loss_contrastive.mean()
        elif reduction == "sum":
            loss_contrastive = loss_contrastive.sum()

        loss = loss * alpha + (1 - alpha) * loss_contrastive

    return loss


class TripletLoss:
    def __init__(self, alpha=0.5, mode="kl_div", margin=2, reduction='mean'):
        self.alpha = alpha
        self.mode = mode
        self.margin = margin
        self.reduction = reduction

    def __call__(self, pred1, pred2, pred3, target1, target2, target3):
        return tripletLoss(pred1, pred2, pred3, target1, target2, target3, self.alpha, self.mode, self.margin,
                           self.reduction)


def distilLoss(tpred, spred, target=None, train_teacher=False, alpha=0.5, mode="kl_div", reduction='mean',
               use_focal=False):
    """
    Example:
        >>> from torch import nn
        >>> from torchvision.models import resnet18
        >>> from torchvision.models import AlexNet

        >>> device = "cuda:0"
        >>> num_classes = 3
        >>> tmodel = resnet18(True)
        >>> tmodel.fc = nn.Linear(tmodel.inplanes, num_classes)
        >>> tmodel.to(device)

        >>> smodel = AlexNet(num_classes).to(device)

        >>> img = torch.ones([2, 3, 224, 224]).to(device)
        >>> target = torch.tensor([0, 1]).to(device)

        >>> tpred = tmodel(img)
        >>> spred = smodel(img)

        >>> loss = distilLoss(tpred, spred, target)

        >>> loss.backward()

        >>> print(loss.item())

    """
    if mode == "kl_div":
        loss = F.kl_div(disti_softmaxv2(spred, log=True), disti_softmaxv2(tpred.detach()), reduction='none').sum(-1)
    else:
        loss = F.pairwise_distance(spred, tpred.detach(), p=2)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    if target is not None:
        if use_focal:
            if train_teacher:
                cls_loss = labelsmooth_focal(tpred, target, reduction=reduction) + \
                           labelsmooth_focal(spred, target, reduction=reduction)
            else:
                cls_loss = labelsmooth_focal(spred, target, reduction=reduction)
        else:
            if train_teacher:
                cls_loss = labelsmooth(tpred, target, reduction=reduction) + \
                           labelsmooth(spred, target, reduction=reduction)
            else:
                cls_loss = labelsmooth(spred, target, reduction=reduction)

        loss = loss * (1 - alpha) + alpha * cls_loss

    return loss


class DistilLoss:
    def __init__(self, train_teacher=False, alpha=0.5, mode="kl_div", reduction='mean'):
        self.train_teacher = train_teacher
        self.alpha = alpha
        self.mode = mode
        self.reduction = reduction

    def __call__(self, tpred, spred, target=None):
        return distilLoss(tpred, spred, target, self.train_teacher, self.alpha, self.mode, self.reduction)


def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)


def distillation(preds, labels, teacher_scores, temp=5.0, alpha=0.7):
    return nn.KLDivLoss()(F.log_softmax(preds / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(preds, labels) * (1. - alpha)


def distillation2(preds, labels, teacher_scores, temp=5.0, alpha=0.7):
    return nn.MSELoss()(preds, teacher_scores) * alpha + F.cross_entropy(preds, labels) * (1. - alpha)


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # loss_reg = self.loss_weight * gaussian_focal_loss(
        #     pred,
        #     target,
        #     weight,
        #     alpha=self.alpha,
        #     gamma=self.gamma,
        #     reduction=reduction,
        #     avg_factor=avg_factor)

        loss_reg = self.loss_weight * gaussian_focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma)
        loss_reg = loss_reg.sum() / avg_factor

        return loss_reg



def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    """Calculate balanced L1 loss.

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, 4).
        target (torch.Tensor): The learning target of the prediction with
            shape (N, 4).
        beta (float): The loss is a piecewise function of prediction and target
            and ``beta`` serves as a threshold for the difference between the
            prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss.
            Defaults to 1.5.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".

    Returns:
        torch.Tensor: The calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    return loss



def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss

def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss

def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss
    
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss

def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    loss = 1 - cious
    return loss


def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


# @mmcv.jit(derivate=True, coderize=True)
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

if __name__ == "__main__":
    from torch import nn
    from torchvision.models import AlexNet

    device = "cuda:0"
    num_classes = 3

    model = AlexNet(num_classes).to(device)

    img = torch.ones([2, 3, 224, 224]).to(device)
    target = torch.tensor([0, 1]).to(device)

    pred = model(img)

    loss = labelsmooth(pred, target)
    # loss = selecterrLoss(pred, target, target)

    loss.backward()

    print(loss.item())
