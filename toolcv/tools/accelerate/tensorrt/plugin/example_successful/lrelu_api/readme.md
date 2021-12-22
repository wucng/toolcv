- [使用tensorrt api 自带的 plugin](https://hub.fastgit.org/NVIDIA/TensorRT/tree/master/plugin)
- 注意： cpp 版本 能编译 但 调用失败
```py
# 查看已存在的plugin 
# (Tensorrt 7.2.3.4 对应的版本 https://hub.fastgit.org/NVIDIA/TensorRT/tree/release/7.2/plugin)

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
	print(plugin_creator.name)

"""
CustomQKVToContextPluginDynamic
CustomEmbLayerNormPluginDynamic
RnRes2Br1Br2c_TRT
RnRes2Br1Br2c_TRT
CustomSkipLayerNormPluginDynamic
CustomSkipLayerNormPluginDynamic
CustomSkipLayerNormPluginDynamic
GroupNormalizationPlugin
CustomGeluPluginDynamic
CustomEmbLayerNormPluginDynamic
CgPersistentLSTMPlugin_TRT
CustomQKVToContextPluginDynamic
CustomQKVToContextPluginDynamic
CustomFCPluginDynamic
SingleStepLSTMPlugin
RnRes2Br2bBr2c_TRT
RnRes2Br2bBr2c_TRT
RnRes2FullFusion_TRT
GridAnchor_TRT
NMS_TRT
Reorg_TRT
Region_TRT
Clip_TRT
LReLU_TRT
PriorBox_TRT
Normalize_TRT
RPROI_TRT
BatchedNMS_TRT
BatchedNMSDynamic_TRT
FlattenConcat_TRT
CropAndResize
DetectionLayer_TRT
Proposal
ProposalLayer_TRT
PyramidROIAlign_TRT
ResizeNearest_TRT
Split
SpecialSlice_TRT
InstanceNormalization_TRT

"""

```
