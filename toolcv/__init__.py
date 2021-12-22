# from .models import *
# from .data import *
# from .network import *
# from .tools import *
# from .cls import *
# from .api import *

"""
example:

from fvcore.common.registry import Registry
from torch import nn
import torch
BACKBONE_REGISTRY = Registry('BACKBONE')
# @BACKBONE_REGISTRY.register()
class Mybackbone(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.m = nn.Sequential(nn.Conv2d(in_c,out_c,3,2,1))
    def forward(self,x):
        return self.m(x)
BACKBONE_REGISTRY.register(Mybackbone)
print(BACKBONE_REGISTRY.get('Mybackbone')(3,32)(torch.rand([1,3,5,5])))

"""