- https://codechina.csdn.net/wc781708249/toolsmall/-/tree/master/toolsmall/tools/speed/layers
- https://codechina.csdn.net/wc781708249/toolsmall/-/blob/master/toolsmall/tools/speed/layers/onnxAPI.py
- https://codechina.csdn.net/wc781708249/toolsmall/-/blob/master/toolsmall/tools/speed/layers/onnxGraphAPI.py

- https://hub.fastgit.org/onnx/onnx/blob/master/docs/Operators.md
- https://github.com/onnx/onnx
- https://hub.fastgit.org/onnx/onnx
- https://pypi.org/project/onnx-simplifier/
- https://hub.fastgit.org/daquexian/onnx-simplifier
- https://github.com/saurabh-shandilya/onnx-utils
- [onnx的基本操作](https://blog.csdn.net/cfh1021/article/details/108732114)
- https://github.com/DaDaLynn/LearnONNX
- [onnx模型的量化处理](https://blog.csdn.net/znsoft/article/details/114637468)
- [基于onnx的网络裁剪](https://zhuanlan.zhihu.com/p/212893519)
- [onnx修改教程](https://blog.csdn.net/weixin_41521681/article/details/112724867)
---

@[toc]
![](https://img-blog.csdnimg.cn/e74374399ab14e5498dcbf6ee178125b.png)


```py
pip install onnxruntime
pip install onnx
```
# 环境
```py
# 环境
CentOS Linux release 7.8.2003 (Core)
Python 3.6.5 
pytorch 							1.7.1
torchvision                       0.8.2
onnx 								1.10.1
onnx-graphsurgeon                 0.3.11
onnxruntime                       1.8.1
cuda 10.2
cudnn 8.1.1
pycuda                            2021.1
tensorrt                          7.2.3.4
tensorflow-gpu                    1.15.0
```

# 上采样scale问题
- `INVALID_GRAPH: Assertion failed: ctx->tensors().count(inputName)`  
- https://blog.csdn.net/qq_26751117/article/details/111352947

参考https://github.com/Tianxiaomo/pytorch-YOLOv4这个实现，在inference时不使用torch自己的插值函数，而是自己重写，成功导出TensorRT

```py
class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
 
    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size
 
        if inference:
 
            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)
 
            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')
```

# pytorch导出onnx模型
```py
def export_to_onnx(model, inputs=(torch.rand([1, 3, 224, 224]),),
                   export_file="model.onnx", opset_version=11, dynamic_batch=True):
    """
    Exp:
        - https://zhuanlan.zhihu.com/p/338791726
        支持动态batch_size

        使用 netron 查看输出的 onnx 文件 模型结构
        - https://github.com/lutzroeder/netron/releases

    Parameters:
        inputs: 也可以有多个输入 如 (torch.rand([1, 3, 224, 224]),torch.rand([1, 3, 224, 224]))
        dynamic_batch: True 动态 batch；False 非动态
    Returns:
        None：得到输出的onnx文件
    Raises:

    Examples
    --------
    >>> from torchvision.models.vgg import vgg16
    >>> model = vgg16()
    >>> export_to_onnx(model)

    """
    model.eval()
    if len(inputs) == 1:
        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
    elif len(inputs) == 2:
        input_names = ["input1", "input2"]
        output_names = ["output"]
        dynamic_axes = {'input1': {0: 'batch_size'},
                        'input2': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}

    torch.onnx.export(model, inputs, export_file, verbose=True, opset_version=opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes if dynamic_batch else None)

```

-  动态batch
![在这里插入图片描述](https://img-blog.csdnimg.cn/a40f607546d34143ab16877ed1876887.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)
-  固定batch
![在这里插入图片描述](https://img-blog.csdnimg.cn/5cb46ac1368e4a5696495c9b9f93eb24.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_20,color_FFFFFF,t_70,g_se,x_16)
# model
```py
import torch
from torch import nn

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.Upsample((8, 8))
        )

        self.fc = nn.Linear(4096, 10)
        self.logit = nn.Softmax(-1)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.logit(x)

        return x


x = torch.rand([1, 3, 32, 32])
model = Mymodel()
print(model(x).shape)
# export_to_onnx(model, inputs=(x,), dynamic_batch=False)
torch.onnx.export(model, x, 'model.onnx', opset_version=11, verbose=True) # 默認輸出固定尺寸
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/50d738bc5e4d48ba91499fd9d71509da.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_18,color_FFFFFF,t_70,g_se,x_16)

# 修改pytorch源代码简化 onnx 结构
- 1、对于view、reshape操作，-1指定放在batch维度（禁止指定大于-1的明确数字），其他维度通过计算得到（必须转成 int类型）
```py
# -1 只放在 batch位置上；x.size() 加上 int
x = x.view(x.size(0), -1) -> x = x.view(-1, int(x.size(1) * x.size(2) * x.size(3)))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/d2ea0ca0e0654a83ae5b70f453134b53.png)

- 2、对于 `F.interpolate` or `nn.Upsample` ，使用scale_factor指定倍率，而不是使用size参数指定大小
```py
nn.Upsample((8, 8)) -> nn.Upsample(None,scale_factor=2.0)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/ccf0446e2c93428d93d591f995183ddb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5ZC055eV,size_13,color_FFFFFF,t_70,g_se,x_16)

最终简化后的模型结构为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/d32f36f59b6748a3bb66deb090d0e4fd.png)

# onnx api (onnx修改成功 但是 转trt 报错）
- 注意 修改完成的onnx模型 使用 onnxruntime 执行会报错，但转成tensorrt 执行没问题

较难使用 推荐使用`onnx_graphsurgeon`

```py
def modify_onnx(onnx_file="model.onnx"):
    """https://github.com/onnx/onnx/blob/master/docs/Operators.md
        https://hub.fastgit.org/onnx/onnx/blob/master/docs/Operators.md

        能修改成功 但是 onnxruntime 无法运行
    """
    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    node = graph.node
    for i in range(len(node)):
        # print(node[i].name)
        # 修改 upsample
        if node[i].name == 'Shape_5':
            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=['17'],
                value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1], np.zeros(1, np.float32))
            )  # 新建新节点
            graph.node.remove(old_scale_node)  # 删除旧节点
            graph.node.insert(i, new_scale_node)  # 插入新节点

        if node[i].name == 'Resize_12':
            scales = np.array([1.0, 1.0, 2.0, 2.0]).astype(np.float32)
            scale_init = onnx.helper.make_tensor('scale_factor', onnx.TensorProto.FLOAT, scales.shape, scales)
            # scale_input = onnx.helper.make_tensor_value_info('scale_factor', onnx.TensorProto.FLOAT, scales.shape)
            rank = 4
            # roi_input = onnx.helper.make_tensor_value_info('roi', onnx.TensorProto.FLOAT, [rank])
            roi_init = onnx.helper.make_tensor('roi', onnx.TensorProto.FLOAT, [rank], [0, 0, 0, 0])

            graph.initializer.append(scale_init)
            graph.initializer.append(roi_init)

            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Resize",
                inputs=['14', 'roi', 'scale_factor'],
                outputs=['25'],
                mode='nearest',
                name='Resize_12'
            )  # 新建新节点
            graph.node.remove(old_scale_node)  # 删除旧节点
            graph.node.insert(i, new_scale_node)  # 插入新节点

        # 修改 x.view (reshape)
        if node[i].name in ['Shape_13']:
            # graph.node.remove(node[i])  # 删除节点
            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=['26'],
                value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1], np.zeros(1, np.float32))
            )  # 新建新节点
            graph.node.remove(old_scale_node)  # 删除旧节点
            graph.node.insert(i, new_scale_node)  # 插入新节点

        if node[i].name == 'Reshape_18':
            dtype = onnx.TensorProto.INT64
            value = [-1, 4096]
            layer_name = 'reshape_constant'
            # constant_input = onnx.helper.make_tensor_value_info(layer_name, dtype, [len(value)])
            constant_init = onnx.helper.make_tensor(
                layer_name,
                dtype,
                [len(value)],
                vals=value
            )
            graph.initializer.append(constant_init)

            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Reshape",
                inputs=['25', layer_name],
                outputs=['33'],
                name='Reshape_18'
            )  # 新建新节点
            graph.node.remove(old_scale_node)  # 删除旧节点
            graph.node.insert(i, new_scale_node)  # 插入新节点

    # 删除多余的节点
    nodes = []
    for i in range(len(node)):
        if node[i].name in ['Conv_0', 'Relu_1', 'MaxPool_2', 'MaxPool_3', 'Resize_12', 'Reshape_18', 'Gemm_19',
                            'Softmax_20']:
            nodes.append(node[i])

    graph = onnx.helper.make_graph(nodes, graph.name, graph.input, graph.output, graph.initializer)
    info_model = onnx.helper.make_model(graph)
    # onnx_model = onnx.shape_inference.infer_shapes(info_model)
    onnx_model = info_model

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_file.replace('.onnx', '_new.onnx'))
    # converted_model = version_converter.convert_version(onnx_model, 11)
    # onnx.save(converted_model, onnx_file.replace('.onnx', '_new.onnx'))

```

# onnx_graphsurgeon(onnx修改成功 但是 转trt 报错）
- 注意 修改完成的onnx模型 使用 onnxruntime 执行会报错，但转成tensorrt 执行没问题
- https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon
```py
# python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

def layer_variable(name="", dtype=np.float32):
    return gs.Variable(name, dtype)


def layer_constant(name="", values=np.ones(shape=(3, 3), dtype=np.float32)):
    return gs.Constant(name, values)


def layer_upsample(layer_name, input_node, graph, resize_scale_factors=2):
    attrs = {
        "coordinate_transformation_mode": 'asymmetric',
        "mode": 'nearest',
        "nearest_mode": 'floor',
    }
    inputs = [input_node]
    scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
    scale_name = layer_name + ".scale"
    roi_name = layer_name + ".roi"
    scale = layer_constant(scale_name, scales)
    roi = layer_constant(roi_name, np.asarray([0, 0, 0, 0], np.float32))
    inputs.append(roi)
    inputs.append(scale)
    output_node = layer_variable(layer_name)
    node = gs.Node(op="Resize", inputs=inputs, outputs=[output_node], attrs=attrs)
    graph.nodes.append(node)

    return output_node


def onnx_reshape(graph, layer_name, input_node, value):
    inputs = [input_node]
    inputs.append(gs.Constant(layer_name + '_constant', np.asarray(value, np.int64)))
    output_node = gs.Variable(layer_name, np.float32)
    node = gs.Node(op="Reshape", inputs=inputs, outputs=[output_node])

    graph.nodes.append(node)

    return output_node


def modify_onnx_gs(onnx_file="model.onnx"):
    """
    graphsurgeon (安装tensorrt 时 cd xxx/tensorrt/graphsurgeon 进行安装)
    https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon
    https://hub.fastgit.org/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon
    https://hub.fastgit.org/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization

    python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

    """
    import onnx_graphsurgeon as gs

    model = onnx.load(onnx_file)
    graph = gs.import_onnx(model)
    nodes = graph.nodes
    for i in range(len(nodes)):
        # nodes[i].op == 'Shape'
        if nodes[i].name in ['Shape_5', 'Concat_10', 'Shape_13', 'Concat_17']:
            nodes[i].outputs.clear()  # 删除某个节点

        # 修改 upsample
        if nodes[i].name == 'Resize_12':
            nodes[i].outputs.clear()  # 删除
            upsample_node = layer_upsample('Resize_12', nodes[i].inputs[0], graph)
            next_node = [node for node in nodes if node.name == "Reshape_18"][-1]
            next_node.inputs.clear()
            next_node.inputs = [upsample_node]  # 重新设置输入

        # 修改 x.view (reshape)
        if nodes[i].name == 'Reshape_18':
            nodes[i].outputs.clear()  # 删除
            reshape_node = onnx_reshape(graph,'Reshape_18',nodes[i].inputs[0],[-1,4096])
            next_node = [node for node in nodes if node.name == "Gemm_19"][-1]
            next_node.inputs.clear()
            next_node.inputs = [reshape_node]  # 重新设置输入

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnx_file.replace('.onnx', '_new.onnx'))
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/7f1e022804344af4bec66dbb3767f21e.png)
# onnx_graphsurgeon v2 (onnx修改成功且转trt 成功）
```py
def modify_onnx_gsv2(onnx_file="model.onnx", out_file="model_m.onnx"):
    """---修改onnx 成功 但是 转 trt 成功
    graphsurgeon (安装tensorrt 时 cd xxx/tensorrt/graphsurgeon 进行安装)
    https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon
    https://hub.fastgit.org/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon
    https://hub.fastgit.org/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization

    python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

    """
    model = onnx.load(onnx_file)
    graph = gs.import_onnx(model)
    nodes = graph.nodes
    for i in range(len(nodes)):
        # nodes[i].op == 'Shape'
        if nodes[i].name in ['Shape_5', 'Concat_10', 'Shape_13', 'Concat_17']:
            nodes[i].outputs.clear()  # 删除某个节点

        # 修改 upsample
        if nodes[i].name == 'Resize_12':
            # print(nodes[i].inputs)
            nodes[i].inputs[-1] = layer_constant('23', np.array([1, 64, 8, 8]))  # 修改size
            # nodes[i].inputs[-2] = layer_constant('24', np.array([1, 1, 2, 2], np.float32)) # 修改scale会失败

        # 修改 x.view (reshape)
        if nodes[i].name == 'Reshape_18':
            # print(nodes[i].inputs)
            nodes[i].inputs[-1] = layer_constant('32', np.array([1, 4096]))

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_file)

```

# 使用onnx-simplifier  (onnx修改成功且转trt 成功）
```py
pip install onnx-simplifier

python3 -m onnxsim input_onnx_model output_onnx_model
```
