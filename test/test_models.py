# @Time: 2021/4/13 13:29 
# @Author: wucong
# @File: test.py 
# @Software: PyCharm
# @Version: win10 Python 3.7.6
# @Version: torch 1.8.1+cu111 torchvision 0.9.1+cu111
# @Version: tensorflow 2.4.1+cu111  keras 2.4.3
# @Describe:
import torch
import torch.nn as nn
from toolcv.models.pytorch import CONV_REGISTRY,SEBLOCK_REGISTRY
print(CONV_REGISTRY.__dict__['_obj_map'])

m = nn.Sequential(
    CONV_REGISTRY.get("CBA")(3,32),
    CONV_REGISTRY.get('depthwise_conv')(32,32),
    SEBLOCK_REGISTRY.get("SELayerV2")(32)
)
print(m)