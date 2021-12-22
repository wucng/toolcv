import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import sys, os
from glob import glob
from tqdm import tqdm
import torch
from collections import Counter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import manifold
import math
import numpy as np
import PIL.Image
import itertools
import cv2
from prettytable import PrettyTable
import torch
from torch.nn.utils import clip_grad_norm_
from fvcore.nn.focal_loss import sigmoid_focal_loss

from toolsmall.data import glob_format


class History():
    epoch = []
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # 打印训练结果信息
    def show_final_history(self, history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
        ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        ax[0].grid()
        ax[1].grid()
        plt.show()


def confusion_matrix2(y_true, y_pred, cate_list=[], save_path=None):  # "./test.csv"
    matrix = metrics.confusion_matrix(y_true, y_pred)
    h, w = matrix.shape
    new_matrix = np.zeros([h, w + 1], np.float32)  # 增加一列score
    new_matrix[:h, :w] = matrix
    for row, item in enumerate(matrix):
        new_matrix[row, -1] = item[row] / np.sum(item)

    # df=pd.DataFrame(new_matrix,index=cate_list,columns=[*cate_list,"score"])
    df = pd.DataFrame(matrix, index=cate_list, columns=cate_list)
    # df["score"]=None # 增加一列score
    df["recall"] = new_matrix[:, -1]
    if save_path == None:
        # print(df)
        sys.stdout.write("%s\n" % (str(df)))
    else:
        # df.to_csv("./test.csv",index=False)
        df.to_csv(save_path)


def confusion_matrix(y_true, y_pred, cate_list=[], save_path=None):  # "./test.csv"
    matrix = metrics.confusion_matrix(y_true, y_pred)
    h, w = matrix.shape
    new_matrix = np.zeros([h + 1, w + 1], np.float32)  # 增加一列score
    new_matrix[:h, :w] = matrix
    for row, item in enumerate(matrix):
        new_matrix[row, -1] = item[row] / np.sum(item)
    for col in range(w):
        new_matrix[-1, col] = matrix[col, col] / np.sum(matrix[:, col])

    # df=pd.DataFrame(new_matrix,index=cate_list,columns=[*cate_list,"score"])
    df = pd.DataFrame(new_matrix, index=cate_list + ["precision"], columns=cate_list + ["recall"], dtype=np.float16)
    # df["recall"]=new_matrix[:,-1]
    # df["precision"]=new_matrix[-1,:]
    if save_path == None:
        # print(df)
        sys.stdout.write("%s\n" % (str(df)))
    else:
        # df.to_csv("./test.csv",index=False)
        df.to_csv(save_path)


# 打印混淆矩阵
def show_confusion_matrix(y_true, y_pred, classes=[]):
    mat = metrics.confusion_matrix(y_true, y_pred)
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))  # 根据需要自行设置大小（也可省略）
    plt.title('confusion matrix', y=1.05, size=15)  # 加标题
    sns.heatmap(mat, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.show()


def show_confusion_matrix2(y_true, y_pred, classes=None, title='Confusion Matrix', cmap=None, normalize=False,
                           figure_size=(8, 6)):
    cm = metrics.confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figure_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
        return self

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)


def static_data(dataPaths=[], classnames=None):
    label_names = []
    for path in dataPaths:
        if classnames is not None:
            label_names.append(classnames.index(os.path.basename(os.path.dirname(path))))
        else:
            label_names.append(os.path.basename(os.path.dirname(path)))

    # 先调一下背景和载入一下数据
    sns.set(style="darkgrid")
    f, ax = plt.subplots(1, 2, figsize=(12, 8))
    sns.countplot(x=label_names, ax=ax[0])
    data = dict(Counter(label_names))
    ax[1].pie(list(data.values()), labels=list(data.keys()), autopct='%1.2f%%')
    plt.show()


def get_img_label(dataPaths=[], classnames=[], counts=25):
    np.random.shuffle(dataPaths)
    labels = []
    images = []
    for i, path in enumerate(dataPaths):
        if i >= counts: break
        if classnames:
            labels.append(classnames.index(os.path.basename(os.path.dirname(path))))
        images.append(np.asarray(PIL.Image.open(path).convert("RGB"), np.uint8))

    return images, labels


def split_data(data_path, classes, train_ratio=0.8):
    train_data = {}
    val_data = {}
    for name in classes:
        paths = glob_format(os.path.join(data_path, name))
        num_train = int(len(paths) * train_ratio)
        np.random.shuffle(paths)
        train_data[name] = paths[:num_train]
        val_data[name] = paths[num_train:]

    return train_data, val_data


def get_img_labelV2(train_data={}, classes=[], counts=None):
    paths = []
    for name in classes:
        _paths = train_data[name]
        paths.extend(_paths[:counts] if isinstance(counts, int) else _paths)

    labels = []
    images = []
    for path in paths:
        # labels.append(classnames.index(os.path.basename(os.path.dirname(path))))
        labels.append(os.path.basename(os.path.dirname(path)))
        images.append(np.asarray(PIL.Image.open(path).convert("RGB"), np.uint8))

    return images, labels


# 可视化每个类别的图片
def show_images(images, labels, rows=5, cols=5):
    f, axes = plt.subplots(rows, cols, figsize=(12, 8))
    # f, axes=plt.subplots(rows,cols)
    # f.subplots_adjust(0, 0, 3, 3)
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(images[idx])
        ax.axis("off")
        ax.set_title(labels[idx])

    plt.show()


# RGB格式
img_mean = np.array([[0.485, 0.456, 0.406]])
img_std = np.array([[0.229, 0.224, 0.225]])


def resizeAndNormal(images, labels, classes=[]):
    # images = [cv2.resize(image,(28,28)).reshape(-1)/255.0 for image in images] # 0~1
    # images = [cv2.resize(image,(28,28)).reshape(-1)/255.0 -0.5 for image in images] # -0.5~0.5
    images = [(cv2.resize(image, (28, 28)).reshape(-1) / 255.0 - 0.5) * 2 for image in images]  # -1~1
    # images = [((cv2.resize(image,(28,28))/255.0-img_mean)/img_std).reshape(-1)  for image in images]

    labels = [classes.index(label) for label in labels]

    return images, labels


def plot_images(images, titles=None, figure_size=(10, 10)):
    """
    Source
    ---------
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    cols = int(math.sqrt(len(images)))
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=figure_size)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(title)
    plt.axis('off')
    plt.show()


# 查看特征空间空间上分布
# 1、使用t-SNE将这些特征简化为二维
# 2、使用PCA将这些特征简化为二维
def visual_feature_PCA(feature, y_pred):
    """
    :param feature: [bs.m]
    :param y_pred: [bs,]
    :return:
    """
    # pca
    pca = PCA(n_components=2)
    pca.fit(feature)
    X_new = pca.transform(feature)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y_pred)
    for i in range(len(y_pred)):
        plt.text(X_new[i, 0] * 1.01, X_new[i, 1] * 1.01, y_pred[i],
                 fontsize=10, color="r", style="italic", weight="light",
                 verticalalignment='center', horizontalalignment='right', rotation=0)  # 给散点加标签

    plt.show()


def visual_feature_TSNE(feature, y_pred):
    """
    :param feature: [bs.m]
    :param y_pred: [bs,]
    :return:
    """
    # "t-SNE"
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feature)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y_pred[i]), color=plt.cm.Set1(y_pred[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


# visual_feature_TSNE(feature,y_pred)
# visual_feature_PCA(feature,y_pred)


# -----------------tensorboard------------------------
from torch.utils.tensorboard import SummaryWriter
import torchvision


def createWriter(summary_path):
    writer = SummaryWriter(summary_path)
    # $ tensorboard --logdir='summary_path'
    return writer


def draw_img(writer, images):
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # write to tensorboard
    writer.add_image('four_images', img_grid)


def draw_loss(writer, train_loss, val_loss, epoch):
    writer.add_scalars("loss", {"val": val_loss, "train": train_loss}, epoch)


def draw_acc(writer, train_acc, val_acc, epoch):
    writer.add_scalars("acc", {"val": val_acc, "train": train_acc}, epoch)


def draw_model(writer, model, images):
    """输出模型结构"""
    writer.add_graph(model, images)


def draw_feature(writer, features, class_labels, label_img=None):
    """特征可视化
    features: [100,10]
    class_labels:[100]
    label_img:[100,3,28,28] or [100,28,28]

    example:
        # 可视化图像特征
        features = images.view(-1, 28 * 28)
        writer.add_embedding(features,
                            metadata=class_labels,
                            label_img=images.unsqueeze(1))

        # 可视化某一层输出特征
        features = net((images.unsqueeze(1)/255.-0.5)/0.5)
        writer.add_embedding(features,metadata=class_labels)

    """
    writer.add_embedding(features, metadata=class_labels, label_img=label_img)


# --------------matplotlib 可视化--------------------
def feature_visual(out_put):
    """
    https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
    # forward
    out_put = model(img)
    """
    for idx, feature_map in enumerate(out_put):
        # [N, C, H, W] -> [C, H, W]
        im = np.squeeze(feature_map.detach().cpu().numpy())
        # [C, H, W] -> [H, W, C]
        im = np.transpose(im, [1, 2, 0])

        # show top 12 feature maps
        plt.figure()
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            # [H, W, C]
            plt.imshow(im[:, :, i], cmap='gray')
            if i == 0: plt.title(str(idx))
        plt.show()


def weight_visual(model):
    """
    https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
    """
    weights_keys = model.state_dict().keys()
    for key in weights_keys:
        # remove num_batches_tracked para(in bn)
        if "num_batches_tracked" in key:
            continue
        # [kernel_number, kernel_channel, kernel_height, kernel_width]
        weight_t = model.state_dict()[key].cpu().numpy()

        # read a kernel information
        # k = weight_t[0, :, :, :]

        # calculate mean, std, min, max
        weight_mean = weight_t.mean()
        weight_std = weight_t.std(ddof=1)
        weight_min = weight_t.min()
        weight_max = weight_t.max()
        print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                                   weight_std,
                                                                   weight_max,
                                                                   weight_min))

        # plot hist image
        plt.close()
        weight_vec = np.reshape(weight_t, [-1])
        plt.hist(weight_vec, bins=50)
        plt.title(key)
        plt.show()


# -----------------------------------------------------
def cross_entropy(preds,targets,reduction="mean"):
    num_classes = preds.size(-1)
    onehotLabel = torch.nn.functional.one_hot(targets, num_classes).float().to(preds.device)
    loss = -preds.log_softmax(1)* (onehotLabel*(num_classes+2)-1) # 同时考虑到正负样本的loss 并且 分类正确的权重大于所有错误权重之和
    if reduction=="mean":
        return loss.mean()
    else:
        return loss.sum()

def cross_entropyV2(preds,targets,reduction="mean",alpha=0.5):
    num_classes = preds.size(-1)
    onehotLabel = torch.nn.functional.one_hot(targets, num_classes).float().to(preds.device)
    loss = -preds.log_softmax(1)*onehotLabel
    # 分错的loss
    loss2 = (preds.softmax(1)*(1-onehotLabel)).sum(1)
    if reduction=="mean":
        return loss.mean()+alpha*loss2.mean()
    else:
        return loss.sum()+alpha*loss2.mean()


def cls_loss(loss_func,outputs,targets,use_focal_loss=False, smooth_label=False):
    if use_focal_loss:
        num_classes = outputs.size(-1)
        onehotLabel = torch.nn.functional.one_hot(targets, num_classes).float().to(outputs.device)
        if smooth_label:  # 标签平滑
            value = np.random.uniform(0.7, 1.0)
            onehotLabel = (1.0 - onehotLabel) * ((1.0 - value) / (num_classes - 1)) + onehotLabel * value
        losses = sigmoid_focal_loss(outputs, onehotLabel, reduction="mean")
    else:
        if smooth_label:  # 标签平滑
            num_classes = outputs.size(-1)
            onehotLabel = torch.nn.functional.one_hot(targets, num_classes).float().to(outputs.device)
            value = np.random.uniform(0.7, 1.0)
            onehotLabel = (1.0 - onehotLabel) * ((1.0 - value) / (num_classes - 1)) + onehotLabel * value
            losses = -(onehotLabel * outputs.log_softmax(1)).mean()
        else:
            losses = loss_func(outputs, targets)
    losses = losses * losses.detach()  # ** 2  # loss 大的 平方后更大 梯度大 更新力度大；loss小的 平方后更小 梯度小 更新力度小

    return losses

def oneshot_loss(outputs1,outputs2,label, margin=2):
    # 结合one-shot-learning loss : 类内小 内间大
    # # label 不同为0 相同为1
    euclidean_distance = torch.nn.functional.pairwise_distance(outputs1,outputs2)
    loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                  (1-label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

    return loss_contrastive * loss_contrastive.detach()

def train_one_epoch_oneshot(model, optimizer, loss_func, data_loader, device, epoch, print_freq=50, use_oneshot_loss=False,
                    gamma=0.5, margin=2,use_focal_loss=False, smooth_label=False,use_cls_loss=True,writer=None):
    model.train()
    num_datas = len(data_loader.dataset)
    losses1 = 0
    losses2 = 0
    losses_oneshot = 0
    iter_per_epoch = len(data_loader)
    for batch_idx, (images, targets,_images,_targets,same_class) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)

        _images = _images.to(device)
        _targets = _targets.to(device)
        _outputs = model(_images)

        if use_cls_loss:
            losses1 = cls_loss(loss_func, outputs, targets, use_focal_loss, smooth_label)
            losses2 = cls_loss(loss_func, _outputs, _targets, use_focal_loss, smooth_label)

        if use_oneshot_loss:
            # outputs1,outputs2 = outputs[:-1], outputs[1:]
            # label = (targets[:-1] == targets[1:]).float()  # 不同为0 相同为1
            label = same_class.float().to(device)
            losses_oneshot = oneshot_loss(outputs,_outputs,label,margin)

        losses = losses_oneshot+(losses1 + losses2) * gamma

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), num_datas,
                       100. * batch_idx * len(images) / num_datas, losses.item()))

        if writer is not None:
            iters = iter_per_epoch*epoch+batch_idx
            writer.add_scalar("loss/train",losses.item(),iters)



def train_one_epoch(model, optimizer, loss_func, data_loader, device, epoch, print_freq=50, use_oneshot_loss=False,
                    gamma=0.5, margin=2,
                    use_focal_loss=False, smooth_label=False
                    ):
    model.train()
    num_datas = len(data_loader.dataset)
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        losses = cls_loss(loss_func, outputs, targets, use_focal_loss, smooth_label)

        if use_oneshot_loss:
            # 结合one-shot-learning loss : 类内小 内间大
            outputs1,outputs2 = outputs[:-1], outputs[1:]
            label = (targets[:-1] == targets[1:]).float()  # 不同为0 相同为1
            losses_oneshot = oneshot_loss(outputs1, outputs2, label, margin)
            losses = losses + losses_oneshot * gamma

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), num_datas,
                       100. * batch_idx * len(images) / num_datas, losses.item()))


"""advanced"""
def train_one_epoch_V2(model, optimizer, loss_func, data_loader, device, epoch,
                       print_freq=50, use_oneshot_loss=False, gamma=0.5, margin=2,
                       mixup_alpha=1.0, ricap_beta=0.3, seed=100):
    """
    # 设置一个随机数，来选择增强方式
    # 1.普通方式
    # 2.ricap
    # 3.mixup
    """
    np.random.seed(seed)
    model.train()
    num_datas = len(data_loader.dataset)
    for batch_idx, (images, targets) in enumerate(data_loader):
        state = np.random.choice(["general", "ricap", "mixup"], 1)[0]

        if state == "general":
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            losses = cls_loss(loss_func, outputs, targets, use_focal_loss, smooth_label)

            if use_oneshot_loss:
                # 结合one-shot-learning loss : 类内小 内间大
                outputs1, outputs2 = outputs[:-1], outputs[1:]
                label = (targets[:-1] == targets[1:]).float()  # 不同为0 相同为1
                losses_oneshot = oneshot_loss(outputs1, outputs2, label, margin)
                losses = losses + losses_oneshot * gamma

        elif state == "ricap":
            # ricap 数据随机裁剪组合增强
            I_x, I_y = images.size()[2:]

            w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
            h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(images.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = images[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = targets[idx].to(device)
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                 torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
            patched_images = patched_images.to(device)
            output = model(patched_images)

            losses = sum([W_[k] * loss_func(output, c_[k]) for k in range(4)])

        else:  # mixup
            l = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(images.size(0))
            input_a, input_b = images, images[idx]
            target_a, target_b = targets, targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b

            target_a = target_a.to(device)
            target_b = target_b.to(device)
            mixed_input = mixed_input.to(device)
            output = model(mixed_input)
            losses = l * loss_func(output, target_a) + (1 - l) * loss_func(output, target_b)

        optimizer.zero_grad()
        losses.backward()
        # clip_grad_norm_(model.parameters(), 1.0 - 1e-10)  # 梯度裁剪
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), num_datas,
                       100. * batch_idx * len(images) / num_datas, losses.item()))


@torch.no_grad()
def evaluate(model, loss_func, data_loader, device):
    model.eval()
    num_datas = len(data_loader.dataset)
    total_loss = 0.0
    _preds = []
    _trues = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        losses = loss_func(outputs, targets)
        total_loss += losses.item()
        preds = outputs.argmax(1)
        _preds.extend(preds)
        _trues.extend(targets)

    valid_loss = total_loss / num_datas
    valid_acc = (torch.eq(torch.tensor(_preds), torch.tensor(_trues)).sum().float() / num_datas).item()

    return valid_loss, valid_acc


@torch.no_grad()
def evaluateV2(model, loss_func, data_loader, device, topk=5):
    model.eval()
    num_datas = len(data_loader.dataset)
    total_loss = 0.0
    _preds = []
    _preds_topk = []
    _trues = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        losses = loss_func(outputs, targets)
        total_loss += losses.item()
        preds = outputs.argmax(1)
        preds_topk = torch.topk(outputs, topk)[1]
        _preds.extend(preds)
        _preds_topk.extend(preds_topk)
        _trues.extend(targets)

    valid_loss = total_loss / num_datas
    valid_acc = (torch.eq(torch.tensor(_preds), torch.tensor(_trues)).sum().float() / num_datas).item()

    topk_acc = 1.0 * sum([1 if t in p else 0 for p, t in zip(_preds_topk, _trues)]) / num_datas

    return valid_loss, valid_acc, topk_acc


@torch.no_grad()
def predict(self, visual_feature=False):
    _preds = []
    _trues = []
    _features = []
    self.network.eval()
    for i, (images, targets) in enumerate(tqdm(self.data.test_loader)):
        # if i==0:
        #     draw_img(self.writer,images)

        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.network(images)
        preds = outputs.argmax(1)
        _preds.extend(preds.cpu().numpy())
        _trues.extend(targets.cpu().numpy())

        _features.extend(outputs.cpu().numpy())

        # if i==0:
        #     draw_model(self.writer,self.network,images)

        if visual_feature:
            print("\ntrues:", _trues)
            print("preds:", _preds)
            feature_visual(self.network.feature(images))

    # ConfusionMatrix(len(classes),classes).update(_preds,_trues).summary()
    # confusion_matrix(_trues,_preds,classes)
    show_confusion_matrix2(_trues, _preds, classes)

    _features = np.array(_features)
    visual_feature_TSNE(_features, _preds)


@torch.no_grad()
def cls_evaluate_File(model, data_loader, device, save_path=None, topk=5):
    """统计最终预测结果 将结果存到.csv文件
    file_name | true | pred | score

    """
    if save_path is None:
        save_path = "cls_result.csv"
    model.eval()
    _preds = []
    _trues = []
    _paths = []
    _scores = []
    paths = data_loader.dataset.paths
    batch_size = data_loader.batch_size
    for i, (images, targets) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        # preds = outputs.argmax(1)
        # scores,preds = outputs.softmax(1).max(1)
        scores, preds = torch.topk(outputs.softmax(1), topk)  # top5
        _preds.extend(preds.cpu().numpy())
        _trues.extend(targets.cpu().numpy())
        _paths.extend(paths[i * batch_size:(i + 1) * batch_size])
        _scores.extend(scores.cpu().numpy())

    # result = [[os.path.basename(_path),_true,_pred,_score] for _path,_true,_pred,_score in zip(_paths,_trues,_preds,_scores)]
    result = [[os.path.basename(_path), _true, ",".join([str(v) for v in _pred]), ",".join([str(v) for v in _score])]
              for _path, _true, _pred, _score in zip(_paths, _trues, _preds, _scores)]

    df = pd.DataFrame(np.array(result), columns=["file_name", "true", "pred", "score"])
    df.to_csv(save_path, index=False)