from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CyclicLR, OneCycleLR
from torch.optim.lr_scheduler import _LRScheduler, StepLR
import math


# from timm.scheduler import TanhLRScheduler

def ManualModifyLR(optimizer, lr):
    """手动更新学习率"""
    # update learning rate
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def CosineAnnealingLROnece(optimizer, T_max=20, lrf=0.01):
    """
    等价于 CosineAnnealingLR(optimizer, T_max=20,eta_min=lr*lrf)

    每个 batch or 每个epoch 执行 scheduler.step()
    """
    lf = lambda x: ((1 + math.cos(x * math.pi / T_max)) / 2) * (
            1 - lrf) + lrf  # cosine  last lr=lr*lrf
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    return scheduler


def SineAnnealingLROnece(optimizer, T_max=20, lrf=0.01, ratio=1 / 4):
    """
    推荐每个 batch 执行 scheduler.step()
    """
    lf = lambda x: ((1 + math.cos(math.pi + x * math.pi / int(T_max * ratio))) / 2) * (
            1 - lrf) + lrf if x < int(T_max * ratio) else ((1 + math.cos(x * math.pi / T_max)) / 2) * (
            1 - lrf) + lrf  # cosine  last lr=lr*lrf
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    return scheduler


def SineAnnealingLROnecev2(optimizer, T_max=20, lrf=0.01, ratio=1 / 4, gamma=0.9, min_g=4):
    """
    推荐每个 batch 执行 scheduler.step()
    """
    lf = lambda x: ((1 + math.cos(math.pi + x * math.pi / int(T_max * ratio))) / 2) * (
            1 - lrf) + lrf if x < int(T_max * ratio) else gamma ** min(x // T_max, min_g) * (
            (1 + math.cos(x * math.pi / T_max)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    return scheduler


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    # return initial_lr * (1 - epoch / max_epochs) ** exponent
    return initial_lr * exponent ** (epoch if epoch < max_epochs else max_epochs)


def PolyLR(optimizer, max_epochs, initial_lr, exponent=0.9):
    lf = lambda x: poly_lr(x, max_epochs, initial_lr, exponent)

    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    return scheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def test_scheduler(init_lr=0.1, min_lr=1e-5):
    import torch
    from torchvision.models import AlexNet
    import matplotlib.pyplot as plt

    model = AlexNet(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=20,eta_min=min_lr)
    # scheduler = CyclicLR(optimizer, 1e-3, 0.05, mode='triangular', step_size_up=300)
    # scheduler = OneCycleLR(optimizer,max_lr= init_lr, total_steps= 100, anneal_strategy= 'cos')
    # scheduler = CosineAnnealingLROnece(optimizer, 20)
    # scheduler = TanhLRScheduler(optimizer,20*2)
    scheduler = PolyLR(optimizer, 40, 0.1, 0.95)

    plt.figure()
    x = list(range(100))
    y = []
    for epoch in range(1, 101):
        optimizer.zero_grad()
        optimizer.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step(epoch)
        # y.append(scheduler.get_lr()[0])
        y.append(scheduler.get_last_lr()[0])
        # y.append(scheduler._get_lr(epoch)[0])

    # 画出lr的变化
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve changes as epoch goes on!")
    plt.show()


def test_scheduler_step(init_lr=0.1, min_lr=1e-5):
    import torch
    from torchvision.models import AlexNet
    import matplotlib.pyplot as plt

    model = AlexNet(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    # scheduler = CosineAnnealingLROnece(optimizer, 20*2)
    # scheduler = SineAnnealingLROnece(optimizer, 20 * 2)
    scheduler = SineAnnealingLROnecev2(optimizer, 20 * 2, min_g=8)

    plt.figure()
    x = []
    y = []
    i = 0
    for epoch in range(0, 20):
        for step in range(20):
            steps = step + epoch * 20
            optimizer.zero_grad()
            optimizer.step()
            print("epoch:%d steps:%d lr:%.5f" % (epoch, steps, optimizer.param_groups[0]['lr']))
            scheduler.step()
            # y.append(scheduler.get_lr()[0])
            y.append(scheduler.get_last_lr()[0])
            x.append(i)
            i += 1

    # 画出lr的变化
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve changes as epoch goes on!")
    plt.show()


if __name__ == "__main__":
    test_scheduler()
    # test_scheduler_step()
