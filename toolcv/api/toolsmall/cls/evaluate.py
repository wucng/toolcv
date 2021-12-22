"""
分类中的混淆矩阵：(假设总共有2个类别分别为cat，dog)
         |   cat(pred)  |    dog(pred)  |    recall      | f1_score
---------|--------------|---------------|----------------|-----------
cat(true)|      3       |       1       | 3/(3+1)=0.75   | 0.666
---------|--------------|---------------|----------------------------
dog(true)|      2       |       5       | 5/(5+2)=0.714  | 0.769
---------|--------------|---------------|----------------|-----------
precision| 3/(3+2)=0.6  | 5/(5+1)=0.833 |

accuracy=(3+5)/(3+1+2+5)=0.727

f1_score=2*P*R/(P+R)

说明：以第一行说明，
真实数据中有4只cat，经过分类器后，其中3只被正确分类,1只被错误分类（一行个数相加 等于真实目标数）


----------------------------------------------------------------------------------
目标检测的混淆矩阵：(假设有2个类别分别为cat,dog，注还有一个默认的类别为背景background)
说明：
检测正确的：
    1、pred_box与 gt_box 的IOU值大于某个阈值(0.5) （boxes）
    2、pred_box对应的pred_label与 gt_box的label相同 （labels）
检测不正确：
    1、每个gt_box最多只有一个pred_box 多出的算误检 （可以算成为检测到 背景）
    2、gt_box没有对应的pred_box 算漏检
    3、检测不正确的 统一 看成为 检测成背景



          | cat(pred)    |      dog      |  recall         |  f1_score
----------|--------------|---------------|-----------------|---------------
cat(true) |      3       |       0       | 3/(3+1+0)=0.75  |    0.666
----------|--------------|---------------|-----------------|----------------
dog       |      0       |       4       | 4/(4+2+1)=0.714 |    0.769
----------|--------------|---------------|-----------------|----------------
background|      2       |       1       |       0         |      0
----------|--------------|---------------|-----------------|----------------
precision | 3/(3+2)=0.6  | 4/(4+1)=0.8   |       0         |      0

说明：以第一行说明，（注：一行个数相加 不等于真实目标数，需要单独计算真实目标数）
真实数据中有4只cat，经过检测器后，其中3只被正确分类,2个pred_box被错误检测为背景(产生5个pred_box)
真实数据中有7只dog，经过检测器后，其中4只被正确分类,1个pred_box被错误检测为背景(产生5个pred_box)
"""

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os
import pandas as pd
import sys
# from toolsmall.tools.utils.tools import box_iou
from toolcv.tools.utils import box_iou

# 计算precision，recall，f1_score
def cal_confusion_matrix(matrix,cate_list=[],save_path=None): # "./test.csv"
    # matrix=metrics.confusion_matrix(y_true,y_pred)
    h,w=matrix.shape
    new_matrix=np.zeros([h+1,w+2],np.float32)
    new_matrix[:h,:w]=matrix
    for row,item in enumerate(matrix):
        new_matrix[row,-2]=item[row]/item[-1] if item[-1]>0 else 0 # 召回率 recall

    for i in range(min(h,w)):
        new_matrix[-1, i] = matrix[i,i]/sum(matrix[:,i]) if sum(matrix[:,i])>0 else 0 # 准确率 precision

    for i in range(min(h,w)):
        R = new_matrix[i,-2]
        P = new_matrix[-1,i]
        new_matrix[i, -1] = 2*P*R/(P+R) if P+R>0 else 0 # 计算 F1


    df=pd.DataFrame(new_matrix,index=[*cate_list,"__background__","precision"],columns=[*cate_list,"__background__","true_nums","recall","F1"])

    if save_path==None:
        # print(df)
        sys.stdout.write("%s\n"%(str(df)))
    else:
        # df.to_csv(save_path, index=False)
        df.to_csv(save_path, index=True)

def cal_confusion_matrix2(matrix,cate_list=[],save_path=None): # "./test.csv"
    # matrix=metrics.confusion_matrix(y_true,y_pred)
    h,w=matrix.shape
    new_matrix=np.zeros([h+1,w+2],np.float32)
    new_matrix[:h,:w]=matrix
    for row,item in enumerate(matrix):
        new_matrix[row,-2]=item[row]/np.sum(item) # 召回率 recall

    for row,item in enumerate(matrix.T):
        new_matrix[-1,row]=item[row]/np.sum(item) # 准确率 precision

    for i,(P,R) in enumerate(zip(new_matrix[-1,:],new_matrix[:,-2])):
        new_matrix[i,-1]=0 if P+R==0 else 2*P*R/(P+R) # 计算 F1

    # df=pd.DataFrame(new_matrix,index=[*cate_list,"precision"],columns=[*cate_list,"recall","F1"])
    cate_list.append("backgrund")
    df=pd.DataFrame(matrix,index=cate_list,columns=cate_list)
    # df["score"]=None # 增加一列score
    df["precision"] = new_matrix[-1,:-2]
    df["recall"]=new_matrix[:-1,-2]
    df["F1"]=new_matrix[:-1,-1]

    if save_path==None:
        # print(df)
        sys.stdout.write("%s\n"%(str(df)))
    else:
        # df.to_csv("./test.csv",index=False)
        df.to_csv(os.path.join(save_path,"evaluate.csv"))

def obj_detecte_confusion_matrix(preds=[],targets=[],num_classes=21,classes=[],save_path=None,iou_thred=[0.5,0.7]):
    """ 针对目标检测
    preds:[{"boxes": boxes, "scores": scores, "labels": labels},{},{}]
    targets:[{"boxes": boxes,"labels": labels,"path":path},{},{}]
    # 创建一个混淆矩阵
    # 行数=num_classes （不包括背景）
    # 列数=num_classes+1 （包括背景，最后一列为背景）

    统计 iou>0.5 iou>0.7的
    """
    if save_path is None:
        save_path = "./obj_detecte_confusion_matrix_result.csv"
    num_classes = num_classes + 1
    # confusion_matrix_05 = np.zeros([num_classes, num_classes+1])# 最后一列标记总的类别数 对应ＩＯＵ大于０．５
    # confusion_matrix_07 = np.zeros([num_classes, num_classes+1])# 最后一列标记总的类别数 对应ＩＯＵ大于０．７
    confusion_matrix = {}
    for thred in iou_thred:
        confusion_matrix[str(thred)] = np.zeros([num_classes, num_classes+1])

    for pred,target in tqdm(zip(preds,targets)):
        t_boxes = target["boxes"]
        p_boxes = pred["boxes"]
        ious = box_iou(p_boxes, t_boxes.to(p_boxes.device))
        values, indexs = ious.max(0)
        # scores = pred["scores"].cpu().numpy()
        t_labels = target["labels"].cpu().numpy()
        p_labels = pred["labels"].cpu().numpy()
        # path = target['path']

        t_boxes = t_boxes.cpu().numpy()
        p_boxes = p_boxes.cpu().numpy()

        for i,(v,ind) in enumerate(zip(values,indexs)):
            ｔ = t_labels[i]
            p = p_labels[ind]
            for thred in iou_thred:
                if v >= thred:
                    confusion_matrix[str(thred)][t,p] += 1 # 正确

                confusion_matrix[str(thred)][t,-1] += 1 # 真实类别个数

        if len(p_boxes)>len(t_boxes):
            # 误检的 多个重复框也算误检
            _values, _indexs = ious.max(1)
            for i, (v, ind) in enumerate(zip(_values, _indexs)):
                if i not in indexs: # 误检
                    p = p_labels[i]
                    for thred in iou_thred:
                        confusion_matrix[str(thred)][-1, p] += 1

    for thred in iou_thred:
        cal_confusion_matrix(confusion_matrix[str(thred)],classes,save_path+"_%s.csv"%(str(thred)))


def obj_detecte_evaluate(preds=[],targets=[],save_path=None):
    """统计最终预测结果 将结果存到.csv文件
    file_name | true | x1| y1 | x2 | y2 | pred | p_x1 | p_y1 | p_x2 | p_y2 | score |iou

    正确预测到 pred | p_x1 | p_y1 | p_x2 | p_y2 | iou 有值
    预测错误  pred | p_x1 | p_y1 | p_x2 | p_y2 | iou  为-1
    如果预测数多于 实际 则 true | x1| y1 | x2 | y2
    """
    result = []
    if save_path is None:
        save_path = "./obj_detecte_result.csv"
    for pred,target in tqdm(zip(preds,targets)):
        t_boxes = target["boxes"]
        p_boxes = pred["boxes"]
        ious = box_iou(p_boxes, t_boxes.to(p_boxes.device))
        values, indexs = ious.max(0)
        scores = pred["scores"].cpu().numpy()
        t_labels = target["labels"].cpu().numpy()
        p_labels = pred["labels"].cpu().numpy()
        path = target['path']

        t_boxes = t_boxes.cpu().numpy()
        p_boxes = p_boxes.cpu().numpy()

        for i,(v,ind) in enumerate(zip(values,indexs)):
            result.append([os.path.basename(path),t_labels[i],*t_boxes[i],
                           p_labels[ind],*p_boxes[ind],scores[ind],v.item()])

        if len(p_boxes)>len(t_boxes):
            # 误检的 多个重复框也算误检
            _values, _indexs = ious.max(1)
            for i, (v, ind) in enumerate(zip(_values, _indexs)):
                if i not in indexs: # 误检
                    result.append(["", -1, *[0,0,0,0],
                                   p_labels[i], *p_boxes[i], scores[i], 1.0])


    # save
    df = pd.DataFrame(np.array(result),columns=["file_name","tlabel","x1","y1","x2","y2",
                                                "plabel","px1","py1","px2","py2","score","iou"])
    df.to_csv(save_path,index=False)

@torch.no_grad()
def obj_eval_to_file(self):
    self.network.eval()
    _preds = []
    _targets = []
    for idx, (data, target) in enumerate(self.data.pred_loader):
        _targets.extend(target)

        if self.use_cuda:
            data = batch(data, stride=32).to(self.device)
            new_target = [
                {k: v.to(self.device) for k, v in targ.items() if k not in ["path", "boxes", "labels", "masks"]}
                for targ in target]
        else:
            new_target = target

        preds = self.network(data, new_target, False, True)

        _preds.extend(preds)

    obj_detecte_evaluate(_preds,_targets)
    # obj_detecte_confusion_matrix(_preds,_targets,num_classes,classes,iou_thred=[0.3,0.5,0.7,0.9])

if __name__=="__main__":
    pass