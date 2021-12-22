import torch
from torch import nn
import onnxruntime as rt
import numpy as np
import onnx
from onnx import version_converter, helper
import onnx_graphsurgeon as gs
import struct

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


def get_onnx_graph(onnx_file="model.onnx"):
    """
    对生成的 onnx 进行查看
    - https://www.jianshu.com/p/476478c17b8e
    """

    # Load the ONNX model
    model = onnx.load(onnx_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def modify_onnx_node(onnx_file="model.onnx"):
    """
    可以选择删除一些节点，修改一下节点，增加一些节点
    https://blog.csdn.net/github_28260175/article/details/105736654
    https://github.com/onnx/onnx/blob/master/docs/Operators.md
    https://zhuanlan.zhihu.com/p/394395167
    """
    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    node = graph.node
    for i in range(len(node)):
        # node[i].op_type == 'Conv'
        if node[i].name == 'Conv_0':
            print(node[i].input)
            for item in node[i].attribute:
                if item.name == 'kernel_shape':
                    pass

        # 将relu替换成 sigmoid
        if node[i].name == 'Relu_1':
            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Sigmoid",
                inputs=['33'],
                outputs=['34'],
                # value=onnx.helper.make_tensor('Sigmoid_1', onnx.TensorProto.FLOAT, [4], [1, 1, 1.81, 1.81])
            )  # 新建新节点
            graph.node.remove(old_scale_node)  # 删除旧节点
            graph.node.insert(i, new_scale_node)  # 插入新节点

    # # 修改输入名
    for inp in graph.input:
        if inp.name == 'input':
            inp.name = 'data'

    node[0].input[0] = 'data'

    # # 修改输出名
    # for outp in graph.output:
    #     if outp.name == 'output':
    #         outp.name = 'logit'

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_file.replace('.onnx', '_new.onnx'))


def onnx_inference(onnx_file="model.onnx", data=None):
    '''
    onnx前向InferenceSession的使用
    '''
    if data is None: data = np.array(np.random.randn(1, 3, 224, 224))
    sess = rt.InferenceSession(onnx_file)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
    print(pred_onx)  # 1x1000
    print(np.argmax(pred_onx))


# ----------------------------example----------------------------------
def example(onnx_file="model.onnx", mode=0, dynamic_batch=False):
    class Mymodel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
                nn.MaxPool2d(3, 2, 1),
                nn.Upsample(None, scale_factor=2.0) if mode else nn.Upsample((8, 8))
            )

            self.fc = nn.Linear(4096, 10)
            self.logit = nn.Softmax(-1)

            self.mode = mode

        def forward(self, x):
            x = self.layer1(x)
            if self.mode:
                x = x.view(-1, int(x.size(1) * x.size(2) * x.size(3)))
            else:
                x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.logit(x)

            return x

    x = torch.rand([1, 3, 32, 32])
    model = Mymodel()
    model.eval()
    # torch.save(model.state_dict(),'model.pth')
    torch2npz(model)
    print(model(x).shape)
    print(list(model.state_dict().keys()))
    export_to_onnx(model, inputs=(x,), dynamic_batch=dynamic_batch)
    # torch.onnx.export(model, x, onnx_file, opset_version=11, verbose=True,
    #                   input_names=['input'], output_names=['output'])


def modify_onnx(onnx_file="model.onnx", out_file="model_m.onnx"):
    """---修改onnx 成功 , 转 trt 成功
    https://github.com/onnx/onnx/blob/master/docs/Operators.md
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
            attrs = {
                "coordinate_transformation_mode": 'asymmetric',
                "mode": 'nearest',
                "nearest_mode": 'floor',
            }

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
                # mode="nearest",
                **attrs,
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
    onnx.save(onnx_model, out_file)
    # converted_model = version_converter.convert_version(onnx_model, 11)
    # onnx.save(converted_model, out_file)


def modify_onnxv2(onnx_file="model.onnx", out_file="model_m.onnx"):
    """---修改onnx 成功 但是 转 trt 失败
    https://github.com/onnx/onnx/blob/master/docs/Operators.md
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
            attrs = {
                "coordinate_transformation_mode": 'asymmetric',
                "mode": 'nearest',
                "nearest_mode": 'floor',
            }
            scales = np.array([]).astype(np.float32)
            scale_init = onnx.helper.make_tensor('scale_factor', onnx.TensorProto.FLOAT, scales.shape, scales)
            # scale_input = onnx.helper.make_tensor_value_info('scale_factor', onnx.TensorProto.FLOAT, scales.shape)
            # roi_input = onnx.helper.make_tensor_value_info('roi', onnx.TensorProto.FLOAT, [rank])
            roi = np.array([]).astype(np.float32)
            roi_init = onnx.helper.make_tensor('roi', onnx.TensorProto.FLOAT, roi.shape, roi)
            size = np.array([1, 64, 8, 8]).astype(np.float32)
            size_init = onnx.helper.make_tensor('size', onnx.TensorProto.FLOAT, size.shape, size)

            graph.initializer.append(scale_init)
            graph.initializer.append(roi_init)
            graph.initializer.append(size_init)

            old_scale_node = node[i]
            new_scale_node = onnx.helper.make_node(
                "Resize",
                inputs=['14', 'roi', 'scale_factor', 'size'],
                outputs=['25'],
                # mode="nearest",
                **attrs,
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
    onnx.save(onnx_model, out_file)
    # converted_model = version_converter.convert_version(onnx_model, 11)
    # onnx.save(converted_model, out_file)


def layer_variable(name="", shape=(3, 3), dtype=np.float32):
    return gs.Variable(name, dtype, shape)


def layer_constant(name="", values=np.ones(shape=(3, 3), dtype=np.float32)):
    return gs.Constant(name, values)


def layer_upsample(layer_name, input_node, output_shape, graph, resize_scale_factors=2):
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
    output_node = layer_variable(layer_name, output_shape)
    node = gs.Node(op="Resize", inputs=inputs, outputs=[output_node], attrs=attrs)
    graph.nodes.append(node)

    return output_node


def onnx_reshape(graph, layer_name, input_node, output_shape, value):
    inputs = [input_node]
    inputs.append(gs.Constant(layer_name + '_constant', np.asarray(value, np.int64)))
    output_node = gs.Variable(layer_name, np.float32, output_shape)
    node = gs.Node(op="Reshape", inputs=inputs, outputs=[output_node])

    graph.nodes.append(node)

    return output_node


def modify_onnx_gs(onnx_file="model.onnx", out_file="model_m.onnx"):
    """---修改onnx 成功 但是 转 trt 失败
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
            nodes[i].outputs.clear()  # 删除
            upsample_node = layer_upsample('Resize_12', nodes[i].inputs[0], [1, 64, 8, 8], graph)
            next_node = [node for node in nodes if node.name == "Reshape_18"][-1]
            next_node.inputs.clear()
            next_node.inputs = [upsample_node]  # 重新设置输入

        # 修改 x.view (reshape)
        if nodes[i].name == 'Reshape_18':
            nodes[i].outputs.clear()  # 删除
            reshape_node = onnx_reshape(graph, 'Reshape_18', nodes[i].inputs[0], [1, 4096], [-1, 4096])
            next_node = [node for node in nodes if node.name == "Gemm_19"][-1]
            next_node.inputs.clear()
            next_node.inputs = [reshape_node]  # 重新设置输入

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_file)


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
            nodes[i].inputs[-1] = layer_constant('32', np.array([-1, 4096]))

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_file)


def modify_onnx_api(onnx_file="model.onnx", out_file="model_m.onnx"):
    """---修改onnx 成功 但是 转 trt 成功
    onnx-simplifier
    pip install onnx-simplifier
    python3 -m onnxsim input_onnx_model output_onnx_model
    """
    # 动态batch
    cmd = 'python -m onnxsim %s %s --input-shape "1,3,32,32"' % (onnx_file, out_file)
    # cmd = 'python -m onnxsim %s %s' % (onnx_file, out_file)
    # import commands
    import os
    os.system(cmd)


# ------------------用于C++ tensorrt API---------------------------------
def torch2wts(model:torch.nn.Module,save_path:str="./model.wts"):
    """用于C++ tensorrt API"""
    model.eval()
    with open(save_path,'w') as fp:
        fp.write("{}\n".format(len(model.state_dict().keys())))
        for k,v in model.state_dict().items():
            print("key:",k)
            print("value:",v.shape)
            vr = v.reshape(-1).cpu().numpy()
            fp.write("{} {}".format(k, len(vr)))
            for vv in vr:
                fp.write(" ")
                fp.write(struct.pack(">f", float(vv)).hex())
            fp.write("\n")

# --------------------用于python tensorrt API---------------------------------
def torch2npz(model:torch.nn.Module,save_path:str="./model.npz"):
    model.eval()
    dict_weights = model.state_dict()
    # save to npz
    weights_arg = {}
    for key, value in dict_weights.items():
        weights_arg[key] = value.cpu().numpy()

    np.savez(save_path, **weights_arg)

    print("save to %s success!"%(save_path))
    # weights = np.load(save_path)

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('-mode', default=0, type=int, help='0,1')
    args.add_argument('-file', default='model.onnx', type=str, help='output onnx file')
    args.add_argument('-outfile', default='model_m.onnx', type=str, help='modify onnx file')
    arg = args.parse_args()

    example(arg.file, mode=arg.mode, dynamic_batch=False)

    # modify_onnx(arg.file, 'model_m.onnx')  # 修改onnx 成功  转 trt 成功
    # modify_onnxv2(arg.file, arg.outfile)  # 修改onnx 成功  转 trt 失败
    # modify_onnx_gs(arg.file, arg.outfile) # 修改onnx 成功  转 trt 失败
    # modify_onnx_gsv2(arg.file, 'model_gs.onnx')  # 修改onnx 成功  转 trt 成功
    # modify_onnx_api(arg.file, arg.outfile) # 修改onnx 成功  转 trt 成功
