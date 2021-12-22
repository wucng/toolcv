# 分类 模型（backbone）
- https://github.com/rwightman/pytorch-image-models # timm

# 目标检测+分割
- https://github.com/open-mmlab/mmdetection # mmdet
- https://github.com/open-mmlab/mmcv
- https://github.com/facebookresearch/detectron2
- https://github.com/facebookresearch/fvcore
- https://github.com/ultralytics/yolov5

# pytorch轻便库（类似于 tensorflow的keras）
- https://github.com/PyTorchLightning/pytorch-lightning

# pytorch 训练加速库
- https://github.com/NVIDIA/apex

# 其他
- https://github.com/fastai/fastai
- https://github.com/yitu-opensource/MobileNeXt
- https://github.com/pytorch/pytorch
- https://github.com/pytorch/vision
- https://github.com/torchgan/torchgan
- https://github.com/Lyken17/pytorch-OpCounter # thop（打印网络 flops与参数量）

# 数据增强库
- https://github.com/albumentations-team/albumentations
- https://github.com/aleju/imgaug
- https://github.com/bethgelab/imagecorruptions

# pycocotools
```python
# 计算 mAP
pip install pycocotools
```

# toolcv

tool for CV

# requirements
```python
!pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html 
!pip install openmim  > /dev/null
!mim install mmdet
!pip install fvcore>=0.1
!pip install pytorch-lightning>=1.3.8
!pip install timm>=0.4.12
!pip install pycocotools
!pip install pycuda
!pip install thop # 计算网络的flops

# apex 安装
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# python setup.py install --cpp_ext --cuda_ext

```


```py
!pip install https://codechina.csdn.net/wc781708249/toolcv/archive/master.zip
# or
!pip install 'git+https://codechina.csdn.net/wc781708249/toolcv.git'
!pip install 'git+https://codechina.csdn.net/wc781708249/toolcv.git@master' # @切换分支
# or
!python setup.py install
```

```py
!pip install 'git+https://codechina.csdn.net/wc781708249/toolcv.git' > /dev/null
!pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html  > /dev/null
!pip install openmim  > /dev/null
!mim install mmdet  > /dev/null
```



# wanb(类似tensorboard)

- https://blog.csdn.net/qq_40507857/article/details/112791111

```python
# pip install wandb

# Flexible integration for any Python script
import wandb

# 1. Start a W&B run
wandb.init(project='gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training here
‍
# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})
```
