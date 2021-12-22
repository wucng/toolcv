from fvcore.common.registry import Registry

ACTIVATE_REGISTRY = Registry('ACTIVATE')
SEBLOCK_REGISTRY = Registry('SEBLOCK')
CONV_REGISTRY = Registry('CONV')
BOTTLEBLOCK_REGISTRY = Registry('BOTTLEBLOCK')
BASICBLOCK_REGISTRY = Registry('BASICBLOCK')

BACKBONE_REGISTRY = Registry('BACKBONE')


from .common import *
from .ghostNet import GhostNet
from .mobilenetv3 import MobileNetV3
from .octconv import OctResNet,OctaveConv
from .oct_mobilenet import OctMobileNet
from .regnet import RegNet
from .res2net import Res2Net
# from .res2net_v1b import Res2Net
from .res2next import Res2NeXt

from .backbone import *