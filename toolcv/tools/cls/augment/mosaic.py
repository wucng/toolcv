import numpy as np
import torch
from torch.nn import functional as F
import random


def mosaictwo_cls(imgs, targets, use_random=True, mode="horizontal", num_classes=None):
    """两张拼接"""
    assert len(imgs) == len(targets) == 2
    assert targets[0] != targets[1]
    alpha = random.choice(np.linspace(0.55, 0.9, 10))

    img = torch.ones_like(imgs[0])

    _, h, w = img.shape
    cx, cy = w // 2, h // 2
    if use_random:
        if mode:
            tmp_w = int(w * alpha)
            s1 = random.choice(range(cx // 2 if (w - tmp_w > cx // 2) else (w - tmp_w) // 2, w - tmp_w))
            s2 = random.choice(range(cx // 2 if (tmp_w > cx // 2) else (tmp_w) // 2, tmp_w))

            img[..., :tmp_w] = imgs[0][..., s1:s1 + tmp_w]
            img[..., tmp_w:] = imgs[1][..., s2:s2 + w - tmp_w]
        else:
            tmp_h = int(h * alpha)
            s1 = random.choice(range(cy // 2 if (h - tmp_h > cy // 2) else (h - tmp_h) // 2, h - tmp_h))
            s2 = random.choice(range(cy // 2 if (tmp_h > cy // 2) else (tmp_h) // 2, tmp_h))
            img[:, :tmp_h, :] = imgs[0][:, s1:s1 + tmp_h, :]
            img[:, tmp_h:, :] = imgs[1][:, s2:s2 + h - tmp_h, :]
    else:
        if mode:
            tmp_w = int(w * alpha)
            img[..., :tmp_w] = imgs[0][..., :tmp_w]
            img[..., tmp_w:] = imgs[1][..., tmp_w:]
        else:
            tmp_h = int(h * alpha)
            img[:, :tmp_h, :] = imgs[0][:, :tmp_h, :]
            img[:, tmp_h:, :] = imgs[1][:, tmp_h:, :]

    if num_classes is not None:
        target = F.one_hot(targets[0], num_classes) * alpha + F.one_hot(targets[1], num_classes) * (1 - alpha)
    else:
        target = {"label": targets, "weight": [alpha, 1 - alpha]}

    return img, target


def mosaicfour_cls(imgs, targets, use_random=True, num_classes=None):
    """四张拼接"""
    assert len(imgs) == len(targets) == 4
    assert targets[0] != targets[1] != targets[2] != targets[3]

    img = torch.ones_like(imgs[0])

    _, h, w = img.shape
    cx, cy = w // 2, h // 2

    while True:
        x = random.choice(range(cx // 4, w - cx // 4))
        y = random.choice(range(cy // 4, h - cy // 4))

        weight1 = (x * y) / (w * h)
        weight2 = ((w - x) * y) / (w * h)
        weight3 = (x * (h - y)) / (w * h)
        weight4 = ((w - x) * (h - y)) / (w * h)

        if max(weight1, weight2, weight3, weight4) > 0.55: break

    if use_random:
        s1 = random.choice(range(cy // 2 if (h - y > cy // 2) else (h - y) // 2, h - y))
        s2 = random.choice(range(cx // 2 if (w - x > cx // 2) else (w - x) // 2, w - x))
        img[:, :y, :x] = imgs[0][:, s1:s1 + y, s2:s2 + x]

        s1 = random.choice(range(cy // 2 if (h - y > cy // 2) else (h - y) // 2, h - y))
        s2 = random.choice(range(cx // 2 if (x > cx // 2) else (x) // 2, x))
        img[:, :y, x:] = imgs[1][:, s1:s1 + y, s2:w - x + s2]

        s1 = random.choice(range(cy // 2 if (y > cy // 2) else (y) // 2, y))
        s2 = random.choice(range(cx // 2 if (w - x > cx // 2) else (w - x) // 2, w - x))
        img[:, y:, :x] = imgs[2][:, s1:h - y + s1, s2:s2 + x]

        s1 = random.choice(range(cy // 2 if (y > cy // 2) else (y) // 2, y))
        s2 = random.choice(range(cx // 2 if (x > cx // 2) else (x) // 2, x))
        img[:, y:, x:] = imgs[3][:, s1:h - y + s1, s2:w - x + s2]
    else:
        img[:, :y, :x] = imgs[0][:, :y, :x]
        img[:, :y, x:] = imgs[1][:, :y, x:]
        img[:, y:, :x] = imgs[2][:, y:, :x]
        img[:, y:, x:] = imgs[3][:, y:, x:]

    if num_classes is not None:
        target = F.one_hot(targets[0], num_classes) * weight1 + F.one_hot(targets[1], num_classes) * weight2 + \
                 F.one_hot(targets[2], num_classes) * weight3 + F.one_hot(targets[3], num_classes) * weight4
    else:
        target = {"label": targets, "weight": [weight1, weight2, weight3, weight4]}

    return img, target


if __name__ == "__main__":
    from torchvision.models import AlexNet

    num_classes = 5
    model = AlexNet(num_classes)
    imgs = [torch.randn([3, 224, 224]), torch.randn([3, 224, 224]), torch.randn([3, 224, 224]),
            torch.randn([3, 224, 224])]
    targets = torch.tensor([0, 1, 2, 3])

    # """
    img, target = mosaicfour_cls(imgs, targets, use_random=True, num_classes=num_classes)
    pred = model(img[None])
    # loss = F.kl_div(pred.sigmoid(), target[None])
    # loss = F.binary_cross_entropy(pred.sigmoid(), target[None])

    # 使用 label smooth
    log_pred = F.log_softmax(pred, -1)
    loss = -(log_pred*target[None]).sum(-1).mean()

    loss.backward()
    """
    # img, target = mosaicfour_cls(imgs, targets, use_random=True, num_classes=None)
    # pred = model(img[None])
    #
    # label = target['label']
    # weight = target['weight']
    # loss = torch.tensor([0])
    # for i in range(len(weight)):
    #     loss = loss + F.cross_entropy(pred, label[i][None]) * weight[i]
    #
    # loss.backward()
    # """

    # label smooth
    # img = torch.randn([3, 224, 224])
    # target = torch.tensor([1])
    # pred = model(img[None])
    # esp = 0.1
    # log_pred = F.log_softmax(pred, -1)
    # loss = -(log_pred).sum(-1).mean()* esp / num_classes + (1 - esp) * F.nll_loss(log_pred, target)
    # loss.backward()

    print(loss.item())
