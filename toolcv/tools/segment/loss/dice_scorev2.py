"""
dice_loss = 1-mIOU
"""
import torch.nn.functional as F
import torch
import numpy as np


def dice_coeff(pred, target, epsilon=1e-6):
    """
    pred:[bs,h,w]
    target:[bs,h,w]
    """
    bs = pred.size(0)
    dice = 0
    for i in range(bs):
        _pred = pred[i].reshape(-1)
        _target = target[i].reshape(-1)
        inter = torch.dot(_pred, _target)
        sets_sum = torch.sum(_pred) + torch.sum(_target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        dice += (2 * inter + epsilon) / (sets_sum + epsilon)

    return dice / bs


def multiclass_dice_coeff(pred, target):
    """
    pred:[bs,c,h,w]
    target:[bs,h,w]
    """
    n_classes = pred.size(1)
    if target.ndim == 3:
        target = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    if target.ndim == 1:
        target = F.one_hot(target, n_classes).float()

    dice = 0
    for c in range(n_classes):
        dice += dice_coeff(pred[:, c], target[:, c])

    return dice / n_classes


def dice_loss(pred, target):
    return 1 - multiclass_dice_coeff(pred, target)


if __name__ == "__main__":
    n_classes = 21
    masks_pred = torch.randn([2, n_classes, 32, 32], requires_grad=True)
    true_masks = torch.from_numpy(np.random.choice(n_classes, [2, 32, 32])).long()
    loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                     F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float())
    # loss = dice_loss(F.one_hot(masks_pred.max(dim=1)[1],n_classes).permute(0, 3, 1, 2).float(),
    #                  F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float())
    #loss = dice_loss(F.one_hot(masks_pred.max(dim=1)[1], n_classes).permute(0, 3, 1, 2).float().detach() * \
    #                 F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float())
    print(loss)

    # from dice_score import dice_loss
    # loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
    #                  F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),True)
    # loss = dice_loss(F.one_hot(masks_pred.max(dim=1)[1], n_classes).permute(0, 3, 1, 2).float(),
    #                  F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float())
    # print(loss)

    loss.backward()
