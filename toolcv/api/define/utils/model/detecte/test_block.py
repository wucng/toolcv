from toolcv.api.define.utils.model.detecte.headfile import *
import torch
from torch import nn

x = torch.rand([1, 128, 32, 32])
# flops:  2.425M params:  9.472K memory(about): 37.888K
# m = Reduction(128,16,'relu')
# flops:  4.784M params:  18.688K memory(about): 74.752K
# m = Reduction(128,8,'relu')
# flops:  2.425M params:  9.472K memory(about): 37.888K
# m = Reduction(128,16,'prelu')
# flops:  18.153M params:  17.728K memory(about): 70.912K
# m = Bottle2neckX(128,32,4,8)
# flops:  75.366M params:  73.600K memory(about): 294.4K
# m = Bottle2neckX(128,32,8,16)
# flops:  36.045M params:  35.200K memory(about): 140.8K
# m = Bottle2neckX(128,32,4,16)
# flops:  18.753M params:  18.313K memory(about): 73.252K
# m = Bottle2neck(128, 32)
# flops:  18.219M params:  17.792K memory(about): 71.168K
# m = Bottle2neck(128, 32, baseWidth=64, scale=1)
# flops:  18.219M params:  17.792K memory(about): 71.168K
# m = Bottleneck(128,32)
# flops:  18.481M params:  18.048K memory(about): 72.192K
# m = Bottleneck(128,32,groups=32,base_width=4) # Bottlenecxk
# flops:  80.740M params:  78.720K memory(about): 314.88K
# m = BottleneckCSP(128,128)
# flops:  20.120M params:  19.584K memory(about): 78.336K
# m = BottleneckCSP(128,128,g=16,e=1/4)
# flops:  45.351M params:  44.160K memory(about): 176.64K
# m = BottleneckCSP(128,128,g=16)
# flops:  18.351M params:  17.795K memory(about): 71.18K
# m = ECABottleneck(128,32)
# flops:  18.352M params:  19.840K memory(about): 79.36K
# m = SEBottleneck(128,32)
# flops:  18.352M params:  19.840K memory(about): 79.36K
# m = SEBottleneck(128,32,groups=32,base_width=4)
# flops:  11.239M params:  10.976K memory(about): 43.904K
# m = GhostBottleneck(128,128)
# flops:  18.455M params:  19.938K memory(about): 79.752K
# m = CbamBottleneck(128,32)
# flops:  5.293M params:  5.688K memory(about): 22.752K
# m = GhostBottleneck2(128,32,128,3,1,True)
# flops:  5.259M params:  5.136K memory(about): 20.544K
# m = GhostBottleneck2(128,32,128,3,1,False)
# flops:  18.219M params:  17.792K memory(about): 71.168K
# m = gluonBottleneck(128,32,cardinality=1,use_se=False)
# flops:  18.615M params:  20.232K memory(about): 80.928K
# m = gluonBottleneck(128,32,cardinality=32,base_width=4,use_se=True)
# flops:  18.352M params:  19.976K memory(about): 79.904K
# m = hrBottleneck(128,32,use_se=True)
# flops:  18.615M params:  20.232K memory(about): 80.928K
# m = hrBottleneck(128,32,cardinality=32,base_width=4,use_se=True)
# flops:  18.219M params:  17.792K memory(about): 71.168K
# m = hrBottleneck(128,32,use_se=False)
# flops:  4.202M params:  4.104K memory(about): 16.416K
# m = I2RBlock(128,128,1,32)
# flops:  1.319M params:  1.288K memory(about): 5.152K
# m = I2RBlock(128,128,1,32,True)
# flops:  11.600M params:  11.328K memory(about): 45.312K
# m = I2RBlockv3_fbn(128,128,1,32,True)
# flops:  9.077M params:  8.864K memory(about): 35.456K
# m = MBv3Block(3,128,32,128,nn.ReLU(inplace=True),None, 1)
# flops:  9.079M params:  11.048K memory(about): 44.192K
# m = MBv3Block(3,128,32,128,nn.ReLU(inplace=True),bnet.SEBlock(128), 1)
# flops:  18.481M params:  18.048K memory(about): 72.192K
# m = MixNetBlock(128,128,[3],[1])
# flops:  246.809M params:  241.058K memory(about): 964.232K
# x = torch.rand([1,512,32,32])
# m = EPSABlock(512,128)
# flops:  214.225M params:  263.752K memory(about): 1055.008K
# m = InvertedResidual(128,128,1,6)
# flops:  9.113M params:  9.720K memory(about): 38.88K
# m = InvertedResidual(128,128,1,1/4)
# flops:  35.723M params:  41.184K memory(about): 164.736K
# m = SKUnit(128, 128, 32, 2, 8, 2)
# flops:  11.374M params:  14.272K memory(about): 57.088K
# m = SKUnit(128, 128, 32, 2, 16, 2,mid_features=32)
# flops:  53.903M params:  185.600K memory(about): 742.4K
# m = Stage3(128)
# m.conv2.stride = 1
# flops:  100.663M params:  98.432K memory(about): 393.728K
# m = TransformerBlock(128,128,4,1)
# flops:  18.407M params:  17.976K memory(about): 71.904K
# m = C3Ghost(128,128,1,g=16,e=1/4)
# flops:  37.601M params:  36.720K memory(about): 146.88K
# m = C3Ghost(128,128,1,g=1)
# flops:  19.890M params:  19.424K memory(about): 77.696K
# m = C3SPP(128, 128, n=1, g=16, e=1 / 4)
# flops:  23.462M params:  22.944K memory(about): 91.776K
# m = C3TR(128, 128, 1, g=16, e=1 / 4)
# flops:  42.336M params:  41.344K memory(about): 165.376K
# m = SPPF(128,128)

# -----------------------------------------------------------------------------------------
# flops:  13.107M params:  12.800K memory(about): 51.2K
# m = AttBottleneckBlock(128,128,bottle_ratio=1/4,group_size=16)

layers = LayerFn()
layers.attn = get_attn('ecam')
layers.self_attn = get_attn('involution')
# flops:  13.122M params:  12.913K memory(about): 51.652K
# m = AttBottleneckBlock(128, 128, bottle_ratio=1 / 4, group_size=16,attn_last=True, layers=layers)
# flops:  13.122M params:  12.913K memory(about): 51.652K
# m = AttBottleneckBlock(128, 128, bottle_ratio=1 / 4, group_size=16, attn_last=True, layers=layers,
#                        drop_block=DropBlock2d,drop_path_rate=0.3)
# flops:  23.083M params:  22.641K memory(about): 90.564K
# m = DarkBlock(128, 128, bottle_ratio=1 / 4, group_size=16, attn_last=True, layers=layers, drop_block=DropPath,
#               drop_path_rate=0.3)
# flops:  23.083M params:  22.641K memory(about): 90.564K
# m = EdgeBlock(128, 128, bottle_ratio=1 / 4, group_size=16, attn_last=True, layers=layers, drop_block=DropBlock2d,
#               drop_path_rate=0.3)
# flops:  20.986M params:  20.593K memory(about): 82.372K
# m = RepVggBlock(128, 128, bottle_ratio=1 / 4, group_size=16, layers=layers, drop_block=DropBlock2d, drop_path_rate=0.3)
# flops:  8.798M params:  8.592K memory(about): 34.368K
# m = SelfAttnBlock(128, 128, bottle_ratio=1 / 4, group_size=16, layers=layers, drop_block=DropPath, drop_path_rate=0.3)

# --------------------3D(rnn格式)-----------------------------------
# x = torch.rand([2, 64, 128])
# flops:  9.535M params:  74.032K memory(about): 296.128K
# m = convitBlock(128, 4, mlp_ratio=1 / 4)
# flops:  9.699M params:  74.056K memory(about): 296.224K
# m = LayerScaleBlock(128, 4, mlp_ratio=1 / 4)
# m = LayerScaleBlockClassAttn(128, 4, mlp_ratio=1 / 4)
# m = SerialBlock(128, 4, mlp_ratio=1 / 4)
# m = ParallelBlock(128, 4, mlp_ratios=[1 / 4])
# flops:  18.874M params:  140.032K memory(about): 560.128K
# m = MixerBlock(128,64,mlp_ratio=[0.5,4])
# flops:  917.504K params:  10.464K memory(about): 41.856K
# m = SpatialGatingBlock(128,64,1/4)
# flops:  2.097M params:  12.512K memory(about): 50.048K
# m = ResBlock(128,64,1/4)
# -------------------------------------------------------

# flops:  8.993M params:  8.881K memory(about): 35.524K
# m = ResBottleneck(128, 128, bottle_ratio=1 / 4, groups=16, attn_last=True, attn_layer=get_attn('ecam'), aa_layer=None,
#                   drop_block=DropBlock2d)
# flops:  23.083M params:  22.641K memory(about): 90.564K
# m = cspDarkBlock(128, 128, bottle_ratio=1 / 4, group_size=16, attn_last=True, layers=layers,
#               drop_block=DropBlock2d,drop_path_rate=0.3)
# flops:  9.372M params:  9.152K memory(about): 36.608K
# m = DlaBottleneck(128,128,1,cardinality=16,base_width=2)
# flops:  9.372M params:  9.152K memory(about): 36.608K
# m = DlaBottle2neck(128,128,1,scale=1,cardinality=16,base_width=2)
# flops:  30.179M params:  29.472K memory(about): 117.888K
# m = DualPathBlock(128, 32, 32, 128, 16, 32, 'proj', False)
# flops:  9.077M params:  9.416K memory(about): 37.664K
# m = InvertedResidual(128,128,exp_ratio=1/4,se_layer=get_attn('se'),drop_path_rate=0.3)
# flops:  42.271M params:  41.832K memory(about): 167.328K
# m = EdgeResidual(128, 128, exp_ratio=1 / 4,se_layer=get_attn('se'), drop_path_rate=0.3)
# m = CondConvResidual(128, 128, exp_ratio=1 / 4,se_layer=get_attn('se'), drop_path_rate=0.3)
# flops:  5.260M params:  5.688K memory(about): 22.752K
# m = GhostBottleneck(128,32,128,se_ratio=1/4)
# flops:  12.911M params:  12.608K memory(about): 50.432K
m = gluonBlock(128, 32, norm_layer=nn.BatchNorm2d)

print(m)
flops(m, x)
print(m(x).shape)
