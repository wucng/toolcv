import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import sys,os
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

class History():
    epoch = []
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # 打印训练结果信息
    def show_final_history(self,history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
        ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        plt.show()


def confusion_matrix(y_true,y_pred,cate_list=[],save_path=None): # "./test.csv"
    matrix=metrics.confusion_matrix(y_true,y_pred)
    h,w=matrix.shape
    new_matrix=np.zeros([h,w+1],np.float32) # 增加一列score
    new_matrix[:h,:w]=matrix
    for row,item in enumerate(matrix):
        new_matrix[row,-1]=item[row]/np.sum(item)

    # df=pd.DataFrame(new_matrix,index=cate_list,columns=[*cate_list,"score"])
    df=pd.DataFrame(matrix,index=cate_list,columns=cate_list)
    # df["score"]=None # 增加一列score
    df["score"]=new_matrix[:,-1]
    if save_path==None:
        # print(df)
        sys.stdout.write("%s\n"%(str(df)))
    else:
        # df.to_csv("./test.csv",index=False)
        df.to_csv(save_path)

# confusion_matrix(y_true,y_pred,cate_list)


# 打印混淆矩阵
def show_confusion_matrix(y_true, y_pred,classes=[]):
    mat=metrics.confusion_matrix(y_true,y_pred)
    colormap = plt.cm.viridis
    plt.figure(figsize=(12,12)) # 根据需要自行设置大小（也可省略）
    plt.title('confusion matrix', y=1.05, size=15) #加标题
    sns.heatmap(mat,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True,
                xticklabels=classes,yticklabels=classes)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.show()


def show_confusion_matrix2(y_true, y_pred,classes=None,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):
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

def static_data(dataPaths=[],classnames=None):
    label_names=[]
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


def get_img_label(dataPaths=[],classnames=[],counts=25):
    np.random.shuffle(dataPaths)
    labels = []
    images = []
    for i,path in enumerate(dataPaths):
        if i >= counts:break
        if classnames:
            labels.append(classnames.index(os.path.basename(os.path.dirname(path))))
        images.append(np.asarray(PIL.Image.open(path).convert("RGB"),np.uint8))

    return images,labels


# 可视化每个类别的图片
def show_images(images,labels,rows=5,cols=5):
    # f, axes=plt.subplots(rows,cols,figsize=(12,8))
    f, axes=plt.subplots(rows,cols)
    f.subplots_adjust(0, 0, 3, 3)
    for idx,ax in enumerate(axes.flatten()):
        ax.imshow(images[idx])
        ax.axis("off")
        ax.set_title(labels[idx])

    plt.show()



def plot_images(images, titles=None, figure_size=(10,10)):
    """
    Source
    ---------
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    cols = int(math.sqrt(len(images)))
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=figure_size)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
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
def visual_feature_PCA(feature,y_pred):
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
    plt.show()

def visual_feature_TSNE(feature,y_pred):
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