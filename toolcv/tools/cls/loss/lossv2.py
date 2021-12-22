import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["labelsmooth", "labelsmooth_focal", "selecterrLoss", "oneshotLoss", "tripletLoss", "distilLoss",
           "LabelSmooth", "LabelSmoothFocal", "OneShotLoss", "TripletLoss", "DistilLoss"]


class FocalLoss(nn.Module):
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
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
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
