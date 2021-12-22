# camera-ready
# https://hub.fastgit.org/fregu856/deeplabv3/blob/master/utils/utils.py
"""
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        # save current model
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
"""
import torch
import torch.nn as nn

import numpy as np

def add_weight_decay(net, l2_value, skip_list=()):
    """
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
    params = add_weight_decay(network, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color

def weight_category():
    """
    - 0
        1、计算整个训练集中各个类别出现的频率
        f_c=\frac{训练集中被标记为c的像素数}{训练集中所有图片的总像素数},c=1,...,K
        2、f_{median} = median(f1,f2,...fk)
        3、为每个类别的loss分配权重
            w_c=\frac{f_{median}}{f_c},c=1,...,K
        这意味着训练集中数量较大的类的权值小于1，数量最小的类的权值最大
    - 1
        # compute the class weights according to the ENet paper:
        class_weights = []
        total_count = sum(trainId_to_count.values())
        for trainId, count in trainId_to_count.items():
            trainId_prob = float(count) / float(total_count)
            trainId_weight = 1 / np.log(1.02 + trainId_prob)
            class_weights.append(trainId_weight)

        print(class_weights)
    - 2
        # 其他设置权重
        1、统计所有数据集 每个类别的像素数量 包括背景
        2、1/nums 作为权重 像素数量越多 权重越小 数量越少权重越大
    - 3
        # define a weighted loss (0 weight for 0 label)
        weights_list = [0]+[1 for i in range(17)]
        weights = np.asarray(weights_list)
        weigthtorch = torch.Tensor(weights_list)
    """
    pass