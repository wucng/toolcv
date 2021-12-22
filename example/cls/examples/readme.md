# 模型迁移
```py
1、模型迁移
    - model:efficientnet_b0
    - epochs:15
    - batch_size:64
    - weight_decay:5e-5
    - lr:5e-4
    - 80%训练 20%测试
    - size:(224,224)
    - transforms: mode = "v1"
    - loss: CrossEntropyLoss
    - optim:RAdam
    - lr_scheduler:CosineAnnealingLR

mean time:23.893
max test acc:0.93052
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/274b9fdfcff44124b02051d35d16c075.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)
# 模型微调01
```py
2、模型微调
    - weight_decay:1e-4  # 正则变大
    - lr:5e-5 # 学习率降低
    - transforms: mode = "v2" # 使用数据增强
    - loss: labelsmooth
    - create_dataset:mode="oneshot"

    通过 #1 的训练发现：
        * 1、训练acc 高 而 测试的acc 低 出现过拟合 - 增加 weight_decay （减少有效参数）
        * 2、出现过拟合 增大 transforms
        * 3、使用模型微调  需增加 weight_decay 并 学习率降低

mean time:52.906
max test acc:0.99455
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/23716d0f84454bb5b084ee676da75b67.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)
# 模型微调02
```py
2.1、模型微调
    - weight_decay:5e-5
    - lr:5e-4
    - transforms: mode = "v2" # 使用数据增强
    - loss: labelsmooth
    - create_dataset:mode="oneshot"

    通过 #2 的训练发现：
    *  1、训练acc 低 出现欠拟合 - 降低 weight_decay （增加有效参数）
    *  2、学习能力略有不足 - 适当增加 lr
    *  3、未出现过拟合 不修改 transforms
    *  4、如果 降低 weight_decay 还是出现欠拟合 需要换一个更大的网络

mean time:52.325
max test acc:1.00000
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/f448e3bcba7e4c07a8141d0c3f49ab7c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)
# 其他策略
```py
如果出现过拟合 修改
    1、增大 weight_decay
    2、使用 dropout、dropblock、droppath
    3、数据增强 get_transforms(mode) mode=v2,v3,v4, get_transformsv2() （推荐 v2）
    3.1、数据增强 create_dataset(mode) mode = mixup", "mosaictwo", "mosaicfour" （推荐 "mosaictwo"）

如果出现欠拟合
	增加网络训练参数：
		1、降低 weight_decay
		2、解冻网络参数、或 换个更大的网络
	正则化：
		0、降低 weight_decay（L2正则）
		1、不使用 dropout、dropblock、droppath
		2、使用简单数据增强（resize+水平镜像） 或 不使用数据增强
	

3、训练策略：
    数据选择：create_dataset(mode) mode = "selecterr","oneshot","triplet" （推荐 "oneshot"）
    模型蒸馏 + 逐步放大 size + 正则
4、 修改 criterion、optimizer, lr_scheduler
```