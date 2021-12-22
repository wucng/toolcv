- [各种 Scheduler 学习率曲线可视化](https://zhuanlan.zhihu.com/p/352821601)



- ```python
  scheduler = CosineAnnealingLR(optimizer, T_max=20,eta_min=min_lr)
  ```

让学习率随epoch的变化图类似于cos，更新策略：

![[公式]](https://www.zhihu.com/equation?tex=new_%7Blr%7D%3Deta_%7Bmin%7D%2B0.5%2A%28initial_%7Blr%7D-eta_%7Bmin%7D%29%5Ctimes%281%2Bcos%28%5Cfrac%7Bepoch%7D%7BT_%7Bmax%7D%7D%5Cpi%29%29)

其中，![[公式]](https://www.zhihu.com/equation?tex=new_%7Blr%7D) 表示新学习率，![[公式]](https://www.zhihu.com/equation?tex=initial_%7Blr%7D)表示初始学习率，![[公式]](https://www.zhihu.com/equation?tex=eta_%7Bmin%7D) 表示最小学习率，![[公式]](https://www.zhihu.com/equation?tex=T_%7Bmax%7D)表示cos周期的1/2。

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max,eta_min=0,last_epoch=-1)
```

其中，余弦退火学习率中LR的变化是周期性的，T_max是周期的1/2；eta_min(float)表示学习率的最小值，默认为0；last_epoch(int)代表上一个epoch数，该变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整。当为-1时，学习率设为初始值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0db09868dab24c47a28c5b6027dd695a.png)





- ```python
  scheduler = OneCycleLR(optimizer,max_lr= init_lr, total_steps= 100, anneal_strategy= 'cos')
  ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/b5e6caa9bb74481ea38419949cfc90ee.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)



- ```python
  lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (
          1 - 0.1) + 0.1  # cosine  last lr=lr*lrf
  scheduler = LambdaLR(optimizer,lr_lambda=lf)
  ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/41f7c06f12fe465aa79bfb52ddc9a7c5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)



- ```python
  def SineAnnealingLROnecev2(optimizer, T_max=20, lrf=0.01, ratio=1 / 4, gamma=0.9, min_g=4):
      lf = lambda x: ((1 + math.cos(math.pi + x * math.pi / int(T_max * ratio))) / 2) * (
              1 - lrf) + lrf if x < int(T_max * ratio) else gamma ** min(x // T_max, min_g) * (
              (1 + math.cos(x * math.pi / T_max)) / 2) * (1 - lrf) + lrf  # cosine  last lr=lr*lrf
      scheduler = LambdaLR(optimizer, lr_lambda=lf)
  
      return scheduler
  
  scheduler = SineAnnealingLROnecev2(optimizer, 20 * 2,min_g=8)
  ```



![在这里插入图片描述](https://img-blog.csdnimg.cn/bbddb2a1eb4d43bb983b356ae0a3f6b7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)



- ```python
  def SineAnnealingLROnece(optimizer, T_max=20, lrf=0.01, ratio=1 / 4):
      lf = lambda x: ((1 + math.cos(math.pi + x * math.pi / int(T_max * ratio))) / 2) * (
              1 - lrf) + lrf if x < int(T_max * ratio) else ((1 + math.cos(x * math.pi / T_max)) / 2) * (
              1 - lrf) + lrf  # cosine  last lr=lr*lrf
      scheduler = LambdaLR(optimizer, lr_lambda=lf)
  
      return scheduler
  
  scheduler = SineAnnealingLROnece(optimizer, 20)
  ```



![在这里插入图片描述](https://img-blog.csdnimg.cn/ee2118e95afe410ea865b497ccd02dab.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)