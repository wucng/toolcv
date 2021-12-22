import numpy as np
import torch
from torch.nn import functional as F
import random


def mixup_cls(imgs, targets, num_classes=None):
    assert len(imgs) == len(targets) == 2
    assert targets[0] != targets[1]
    alpha = random.choice(np.linspace(0.55, 0.9, 10))
    img = imgs[0] * alpha + imgs[1] * (1 - alpha)
    if num_classes is not None:
        target = F.one_hot(targets[0], num_classes) * alpha + F.one_hot(targets[1], num_classes) * (1 - alpha)
    else:
        target = {"label": targets, "weight": [alpha, 1 - alpha]}

    return img, target


if __name__ == "__main__":
    imgs = [torch.randn([3, 224, 224]), torch.randn([3, 224, 224])]
    targets = torch.tensor([0, 1])

    for _ in range(1000):
        img, target = mixup_cls(imgs[:2],targets[:2],num_classes=5)

    print("over")