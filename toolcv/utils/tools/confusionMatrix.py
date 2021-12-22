# https://zhuanlan.zhihu.com/p/443499860

from toolcv.tools.utils import box_iou
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


class ConfusionMatrix:
    """
    https://zhuanlan.zhihu.com/p/443499860

    用于目标检测的混淆矩阵；
    混淆矩阵的最右边一列，就能够看出每个类别漏检的概率
    单看最下面一行，也能够看出不同类别的误报率
    """

    def __init__(self, iou_thred=0.5, classes=[]):
        self.iou_thred = iou_thred
        assert len(classes) > 0
        self.classes = classes + ["_bg_"]  # 加上背景
        self.num_classes = len(self.classes)

        self.matrix = np.zeros([self.num_classes, self.num_classes])

    def add_batch(self, pred, target):
        pboxes, plabels = pred["boxes"].cpu(), pred["labels"].cpu().numpy()
        tboxes, tlabels = target["boxes"].cpu(), target["labels"].cpu().numpy()

        if pboxes is None or len(pboxes) == 0 or pboxes.sum().item() == 0:
            for label in tlabels:
                self.matrix[label, -1] += 1  # 漏检（将目标当成背景）
        else:
            pindex = []
            ious = box_iou(tboxes, pboxes)
            # gt-boxes 对应的最大 预测框
            values, indices = ious.max(1)
            for i, (value, ind) in enumerate(zip(values, indices)):
                if value >= self.iou_thred:
                    pindex.append(ind)
                    self.matrix[tlabels[i], plabels[ind]] += 1
                else:
                    self.matrix[tlabels[i], -1] += 1  # 漏检（将目标当成背景）
            # 误检(将背景误检为目标)
            if len(plabels) > len(pindex):
                for i in range(len(plabels)):
                    if i in pindex: continue
                    self.matrix[-1, plabels[i]] += 1

    def get_confusion_matrix(self):
        return pd.DataFrame(self.matrix.astype('int'), index=self.classes, columns=self.classes)

    def save_confusion_matrix(self, save_path="./confusion_matrix.csv"):
        self.get_confusion_matrix().to_csv(save_path.replace(".csv", "_%s.csv" % self.iou_thred))

    def plot_confusion_matrix(self, figsize=(20, 8), dpi=None, percent=False):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
        plt.title('confusion_matrix iou=%s' % self.iou_thred)
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=-45)
        plt.yticks(tick_marks, self.classes)
        # ij配对，遍历矩阵迭代器
        # iters = np.reshape([[[i, j] for j in range(self.num_classes)] for i in range(self.num_classes)],
        #                    (self.matrix.size, 2))
        # for i, j in iters:
        #     plt.text(j, i, format(self.matrix[i, j]))  # 显示对应的数字

        # for i in range(self.num_classes):
        #     for j in range(self.num_classes):

        for i, j in product(range(self.num_classes), range(self.num_classes)):
            if percent:
                plt.text(j, i, "%.3f" % (self.matrix[i, j] / np.sum(self.matrix[i])))
            else:
                plt.text(j, i, "%d" % int(self.matrix[i, j]))

        plt.ylabel('Real label')
        plt.xlabel('Prediction')
        plt.tight_layout()
        plt.show()
