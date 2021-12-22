"""
# 打开 netron 查看生成的模型结构

https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon

# install
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/onnx-graphsurgeon
make install  (make install_venv  if installing inside a virtual environment:)
make build
python3 -m pip install onnx_graphsurgeon/dist/onnx_graphsurgeon-X.Y.Z-py2.py3-none-any.whl --user


# import graphsurgeon # 这个是安装tensorrt 是安装的 是用于解析 tensorflow文件

import onnx_graphsurgeon as gs # 用于解析tensorflow文件 onnx文件
"""

import onnx_graphsurgeon as gs
import numpy as np
import onnx

import torch
from torch import nn
import tensorrt as trt

# from toolsmall.tools.speed.modelTansform import onnx2engine, \
#     torch2onnx, torch2npz,to_numpy,loadEngine,common
from toolcv.tools.accelerate.tensorrt.api.modelTansform import onnx2engine, \
    torch2onnx, torch2npz,to_numpy,loadEngine,common

class OnnxGraphAPI:
    def __init__(self,model_path="model.npz",output_file_path='model.onnx',
                 alpha_lrelu = 0.1,epsilon_bn = 1e-5,momentum_bn = 0.99):
        self.weights = np.load(model_path)
        self.node=list()
        self.inputs=list()
        self.outputs=list()

        self.alpha_lrelu = alpha_lrelu
        self.epsilon_bn = epsilon_bn
        self.momentum_bn = momentum_bn
        self.graph_name = "model_test"
        self.output_file_path = output_file_path

    def layer_variable(self,name="",shape=(3,3),dtype=np.float32):
        return gs.Variable(name,dtype,shape)

    def layer_constant(self,name="",values=np.ones(shape=(3, 3), dtype=np.float32)):
        return gs.Constant(name,values)

    def layer_input(self,name="",shape=(3,3),dtype=np.float32):
        x = self.layer_variable(name,shape,dtype)
        self.inputs.append(x)
        return x

    def layer_conv(self,layer_name,input_node,output_shape,kernel_shape=(3,3),strides=(1,1),pads=(1,1,1,1)):
        attrs ={
            'group':1,
            'dilations':[1,1],
            'kernel_shape':kernel_shape,
            'strides':strides,
            'pads':pads,
            # "auto_pad": 'SAME_LOWER',
        }
        inputs=[input_node]
        weights_name = layer_name + ".weight"
        W = self.layer_constant(weights_name,self.weights[weights_name])
        inputs.append(W)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            b = self.layer_constant(bias_name,self.weights[bias_name])
            inputs.append(b)

        output_node = self.layer_variable(layer_name, output_shape)

        node = gs.Node(op="Conv",inputs=inputs,outputs=[output_node],attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_deconv(self,layer_name,input_node,output_shape,kernel_shape=(3,3),strides=(2,2),pads=(1,1,1,1),output_padding=(1, 1)):
        attrs ={
            'group':1,
            'dilations':[1,1],
            'kernel_shape':kernel_shape,
            'strides':strides,
            'pads':pads,
            'output_padding':output_padding,
            # "auto_pad": 'SAME_LOWER',
        }
        inputs=[input_node]
        weights_name = layer_name + ".weight"
        W = self.layer_constant(weights_name,self.weights[weights_name])
        inputs.append(W)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            b = self.layer_constant(bias_name,self.weights[bias_name])
            inputs.append(b)

        output_node = self.layer_variable(layer_name, output_shape)

        node = gs.Node(op="ConvTranspose",inputs=inputs,outputs=[output_node],attrs=attrs)

        self.node.append(node)

        return output_node


    def layer_bn(self,layer_name,input_node,output_shape):
        attrs = {
            "epsilon":self.epsilon_bn,
            "momentum":self.momentum_bn
        }
        inputs = [input_node]
        param_names = [layer_name + ".weight", layer_name + ".bias", layer_name + ".running_mean",layer_name + ".running_var"]
        for param in param_names:
            inputs.append(self.layer_constant(param,self.weights[param]))

        output_node = self.layer_variable(layer_name, output_shape)

        node = gs.Node(op="BatchNormalization", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_lrule(self,layer_name,input_node,output_shape,alpha_lrelu=0.1):
        attrs = {
            "alpha":alpha_lrelu
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="LeakyRelu", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_rule(self,layer_name,input_node,output_shape):
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Relu", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_tanh(self, layer_name, input_node, output_shape):
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Tanh", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node


    def layer_maxpool(self,layer_name,input_node,output_shape,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        attrs = {
            "kernel_shape":kernel_shape,
            "strides":strides,
            "pads":pads,
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="MaxPool", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_avgpool(self,layer_name,input_node,output_shape,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        attrs = {
            "kernel_shape":kernel_shape,
            "strides":strides,
            "pads":pads,
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="AveragePool", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_flatten(self,layer_name,input_node,output_shape,axis=1):
        attrs = {
            "axis":axis,
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Flatten", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_fc(self,layer_name,input_node,output_shape):
        attrs = {
            "alpha":1.,
            "beta":1.,
            "transB":1,
        }
        inputs = [input_node]
        weights_name = layer_name + ".weight"
        W = self.layer_constant(weights_name, self.weights[weights_name])
        inputs.append(W)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            b = self.layer_constant(bias_name, self.weights[bias_name])
            inputs.append(b)

        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Gemm", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_sigmoid(self,layer_name,input_node,output_shape):
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Sigmoid", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_softmax(self,layer_name,input_node,output_shape,axis=1):
        attrs = {
            "axis":axis
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Softmax", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_add(self,layer_name,input_node1,input_node2,output_shape):
        inputs = [input_node1,input_node2]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Add", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_mul(self,layer_name,input_node1,input_node2,output_shape):
        inputs = [input_node1,input_node2]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Mul", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_exp(self,layer_name,input_node,output_shape):
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Exp", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_concat(self,layer_name,input_node=[],output_shape=(),axis=1):
        attrs = {
            "axis":axis
        }
        inputs = input_node
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Concat", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node


    def layer_upsample(self,layer_name,input_node,output_shape,resize_scale_factors=2):
        attrs = {
            "coordinate_transformation_mode":'asymmetric',
            "mode":'nearest',
            "nearest_mode":'floor',
        }
        inputs = [input_node]
        scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
        scale_name = layer_name + ".scale"
        roi_name = layer_name + ".roi"
        scale = self.layer_constant(scale_name,scales)
        roi = self.layer_constant(roi_name,np.asarray([0,0,0,0],np.float32))
        inputs.append(roi)
        inputs.append(scale)
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Resize", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_transpose(self, layer_name, input_node, output_shape, perm=[0,1,2,3]):
        attrs = {
            "perm":perm
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Transpose", inputs=inputs, outputs=[output_node], attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_reshape(self, layer_name, input_node, output_shape, value):
        inputs = [input_node]
        inputs.append(self.layer_constant(layer_name+'_constant',np.asarray(value,np.int64)))
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Reshape", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_clamp(self, layer_name, input_node, output_shape, min_value=0.0,max_value=1.0):
        inputs = [input_node]
        inputs.append(self.layer_constant(layer_name+"_min",np.array([min_value],np.float32)))
        inputs.append(self.layer_constant(layer_name+"_max",np.array([max_value],np.float32)))

        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Clip", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node

    def layer_squeeze(self, layer_name, input_node, output_shape,axes=0):
        attrs = {
            "axes": [axes]
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Squeeze", inputs=inputs, outputs=[output_node],attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_unsqueeze(self, layer_name, input_node, output_shape,axes=0):
        attrs = {
            "axes": [axes]
        }
        inputs = [input_node]
        output_node = self.layer_variable(layer_name, output_shape)
        node = gs.Node(op="Unsqueeze", inputs=inputs, outputs=[output_node],attrs=attrs)

        self.node.append(node)

        return output_node

    def layer_slice(self, layer_name, input_node, output_shape, start=(0, 0, 0, 0), shape=(2, 2, 3, 3),
                    stride=(1, 1, 1, 1)):
        """
        x = torch.randn([8,8])
        x[:,2:4]
        layer_slice("slice",x,(8,2),(0,2),(8,4),(1,1))
        """
        inputs = [input_node]

        inputs.extend([self.layer_constant(layer_name + '_constant_start', np.asarray(start, np.int32)),
                       self.layer_constant(layer_name + '_constant_shape', np.asarray(shape, np.int32)),
                       self.layer_constant(layer_name + '_constant_axis', np.arange(0,len(start)).astype(np.int32)),
                       self.layer_constant(layer_name + '_constant_stride', np.asarray(stride, np.int32)),
                       ])
        name = layer_name
        output_node = self.layer_variable(name, output_shape)

        node = gs.Node(op="Slice", inputs=inputs, outputs=[output_node])

        self.node.append(node)

        return output_node


    def save_model(self):
        # Note that initializers do not necessarily have to be graph inputs
        graph = gs.Graph(nodes=self.node, inputs=self.inputs, outputs=self.outputs)
        # print(onnx.helper.printable_graph(graph))
        onnx.save(gs.export_onnx(graph), self.output_file_path)

        """验证保存的.onnx格式是否正确"""
        onnx_model = onnx.load(self.output_file_path)
        onnx.checker.check_model(onnx_model)

    @classmethod
    def resize_model(cls,input_file="model.onnx",output_file="subgraph.onnx"):
        """修改模型的输入与输出(截断输入输出)"""
        model = onnx.load(input_file)
        graph = gs.import_onnx(model)
        # tensors = graph.tensors()
        # 重新设置模型的输入与输出
        # graph.inputs = [tensors['x'].to_variable(np.float32)]
        # graph.outputs = [tensors['sigmoid'].to_variable(np.float32,shape=(1,8))] # 原本输出节点名为"softmax"
        first_add = [node for node in graph.nodes if node.op == "Sigmoid"][-1]
        graph.outputs = [first_add.outputs[0].to_variable(np.float32,shape=(1,8))]

        graph.cleanup()
        onnx.save(gs.export_onnx(graph), output_file)

    @classmethod
    def modeify_model(cls, input_file="model.onnx", output_file="modified.onnx"):
        """修改某个节点
        Sigmoid 替换成 LeakyRelu
        """
        graph = gs.import_onnx(onnx.load(input_file))

        first_add = [node for node in graph.nodes if node.op == "Sigmoid"][-1]  # 找到最后一个名为 Sigmoid 的节点
        # first_add.inputs = [inp for inp in first_add.inputs if inp.name == "fc"]  # 找到其对应的输入
        # first_add.inputs = [inp for inp in first_add.inputs]  # 找到其对应的输入
        # first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"] # 找到其对应的输入(删除为‘b’的输入节点)
        first_add.outputs.clear()
        # 2. Change the Add to a LeakyRelu
        lrelu = gs.Variable('new_lrelu', dtype=np.float32)
        graph.nodes.append(gs.Node(op="LeakyRelu", inputs=first_add.inputs, outputs=[lrelu], attrs={"alpha": 0.02}))

        # 找到下一个节点 重新设置输入
        first_add = [node for node in graph.nodes if node.op == "Softmax"][-1]  # 找到最后一个名为 Sigmoid 的节点
        first_add.inputs.clear()
        first_add.inputs = [lrelu]  # 重新设置输入

        # 5. Remove unused nodes/tensors, and topologically sort the graph
        graph.cleanup().toposort()

        onnx.save(gs.export_onnx(graph), output_file)

    @classmethod
    def add_model(cls, input_file="model.onnx", output_file="add.onnx"):
        """增加节点
        在Sigmoid 前增加 LeakyRelu 节点()
        """
        graph = gs.import_onnx(onnx.load(input_file))

        first_add = [node for node in graph.nodes if node.op == "Sigmoid"][-1]  # 找到最后一个名为 Sigmoid 的节点
        # first_add.inputs = [inp for inp in first_add.inputs if inp.name == "fc"]  # 找到其对应的输入
        # first_add.inputs = [inp for inp in first_add.inputs]  # 找到其对应的输入
        # first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"] # 找到其对应的输入(删除为‘b’的输入节点)

        # 2. Change the Add to a LeakyRelu
        lrelu = gs.Variable('new_lrelu', dtype=np.float32)
        graph.nodes.append(gs.Node(op="LeakyRelu", inputs=first_add.inputs, outputs=[lrelu], attrs={"alpha": 0.02}))

        # 此时 sigmoid输入变成了lrelu（输出）
        first_add.inputs.clear()
        first_add.inputs = [lrelu]

        # 5. Remove unused nodes/tensors, and topologically sort the graph
        graph.cleanup().toposort()

        onnx.save(gs.export_onnx(graph), output_file)

    @classmethod
    def remove_model(cls, input_file="model.onnx", output_file="removed.onnx"):
        """删除某个节点
        删除sigmoid节点
        """
        graph = gs.import_onnx(onnx.load(input_file))
        first_add = [node for node in graph.nodes if node.op == "Sigmoid"][-1]
        # first_add.inputs = [inp for inp in first_add.inputs if inp.name == "fc"]  # 找到其对应的输入
        # first_add.inputs = [inp for inp in first_add.inputs]  # 找到其对应的输入
        first_add.outputs.clear()

        # 找到下一个节点 重新设置输入
        next_add = [node for node in graph.nodes if node.op == "Softmax"][-1]  # 找到最后一个名为 Sigmoid 的节点
        next_add.inputs.clear() # 先清除，再重新指定
        next_add.inputs = first_add.inputs  # 重新设置输入

        # Remove the fake node from the graph completely
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), output_file)

    @classmethod
    def remove_model2(cls, input_file="model.onnx", output_file="removed.onnx"):
        """删除某个节点
        删除sigmoid节点
        """
        graph = gs.import_onnx(onnx.load(input_file))
        first_add = [node for node in graph.nodes if node.op == "Sigmoid"][-1]

        # 找到下一个节点 重新设置输入
        next_add = [node for node in graph.nodes if node.op == "Softmax"][-1]  # 找到最后一个名为 Sigmoid 的节点
        next_add.inputs.clear()  # 先清除，再重新指定
        next_add.inputs = first_add.inputs  # 重新设置输入

        # 删除sigmoid节点
        graph.nodes.remove(first_add)

        # Remove the fake node from the graph completely
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), output_file)

    @classmethod
    def modeify_model2(cls, input_file="model.onnx", output_file="add.onnx"):
        """重新修改resize的实现
        """
        graph = gs.import_onnx(onnx.load(input_file))

        first_add = [node for node in graph.nodes if node.op == "LeakyRelu"][0]  # 找到 LeakyRelu 的节点
        # first_add = [node for node in graph.nodes if node.name == "LeakyRelu_2"][0]  # 找到 LeakyRelu 的节点
        # first_add.inputs = [inp for inp in first_add.inputs]  # 找到其对应的输入
        # first_add.outputs = [inp for inp in first_add.outputs]  # 找到其对应的输出
        first_add.outputs.clear() # 必须执行,clear 删除掉输出的相关链接 ,但也导致 LeakyRelu 没有了输出，因此必须重新实现生成新的输出
        # graph.nodes.remove(first_add) # 删除整个节点

        second_add = [node for node in graph.nodes if node.op == "MaxPool"][0]
        # second_add = [node for node in graph.nodes if node.name == "MaxPool_32"][0]
        second_add.inputs.clear() # 必须执行,clear 删除掉输入的相关链接，后面得重新指定其输入

        # 重新定义LeakyRelu层
        attrs = {
            "alpha": 0.1
        }
        lrelu = gs.Variable("new_lrelu",np.float32)
        node = gs.Node(op="LeakyRelu", inputs=first_add.inputs, outputs=[lrelu], attrs=attrs)
        graph.nodes.append(node)

        # 重新定义resize层(实现upsample)
        attrs = {
            "coordinate_transformation_mode": 'asymmetric',
            "mode": 'nearest',
            "nearest_mode": 'floor',
        }
        layer_name = "new_resize" # 不要和原来 的resize节点名重复
        scales = np.array([1.0, 1.0, 2, 2]).astype(np.float32)
        scale_name = layer_name + ".scale"
        roi_name = layer_name + ".roi"
        scale = gs.Constant(scale_name,scales)
        roi = gs.Constant(roi_name,np.asarray([0, 0, 0, 0], np.float32))
        # inputs =first_add.outputs
        inputs = [lrelu]
        inputs.append(roi)
        inputs.append(scale)
        resize = gs.Variable(layer_name, dtype=np.float32)
        node = gs.Node(op="Resize", inputs=inputs, outputs=[resize], attrs=attrs)
        graph.nodes.append(node)

        # 重新设置下一层的输入节点
        second_add.inputs = [resize]

        # 5. Remove unused nodes/tensors, and topologically sort the graph
        graph.cleanup().toposort()

        onnx.save(gs.export_onnx(graph), output_file)


# ----------------------api---------------------------------------------
def onnx_slice(nodes,layer_name,input_node,output_shape,start=(0, 0, 0, 0), shape=(2, 2, 3, 3),
                    stride=(1, 1, 1, 1)):
    """
    x = torch.randn([8,8])
    x[:,2:4]

    onnx_slice(nodes,"slice",x,(0,2),(8,4),(1,1))
    """
    inputs = [input_node]

    inputs.extend([gs.Constant(layer_name + '_constant_start', np.asarray(start, np.int32)),
                   gs.Constant(layer_name + '_constant_shape', np.asarray(shape, np.int32)),
                   gs.Constant(layer_name + '_constant_axis', np.arange(0, len(start)).astype(np.int32)),
                   gs.Constant(layer_name + '_constant_stride', np.asarray(stride, np.int32)),
                   ])
    name = layer_name
    output_node = gs.Variable(name, np.float32,output_shape)

    node = gs.Node(op="Slice", inputs=inputs, outputs=[output_node])

    nodes.append(node)

    return output_node

def onnx_upsample(nodes,layer_name,input_node,output_shape=None,resize_scale_factors=2):
    attrs = {
        "coordinate_transformation_mode": 'asymmetric',
        "mode": 'nearest',
        "nearest_mode": 'floor',
    }
    layer_name = layer_name  # 不要和原来 的resize节点名重复
    scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
    scale_name = layer_name + ".scale"
    roi_name = layer_name + ".roi"
    scale = gs.Constant(scale_name, scales)
    roi = gs.Constant(roi_name, np.asarray([0, 0, 0, 0], np.float32))
    inputs = [input_node, roi, scale]
    output_node = gs.Variable(layer_name, dtype=np.float32,shape=output_shape)
    node = gs.Node(op="Resize", inputs=inputs, outputs=[output_node], attrs=attrs)

    nodes.append(node)

    return output_node


def onnx_sigmoid(nodes,layer_name,input_node,output_shape):
    inputs = [input_node]
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Sigmoid", inputs=inputs, outputs=[output_node])

    nodes.append(node)

    return output_node

def onnx_exp(nodes,layer_name,input_node,output_shape):
    inputs = [input_node]
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Exp", inputs=inputs, outputs=[output_node])

    nodes.append(node)

    return output_node


def onnx_add(nodes,layer_name,input_node1,input_node2,output_shape):
    inputs = [input_node1,input_node2]
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Add", inputs=inputs, outputs=[output_node])

    nodes.append(node)

    return output_node

def onnx_mul(nodes,layer_name,input_node1,input_node2,output_shape):
    inputs = [input_node1,input_node2]
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Mul", inputs=inputs, outputs=[output_node])

    nodes.append(node)
    return output_node

def onnx_reshape(nodes,layer_name, input_node, output_shape, value):
    inputs = [input_node]
    inputs.append(gs.Constant(layer_name+'_constant',np.asarray(value,np.int64)))
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Reshape", inputs=inputs, outputs=[output_node])

    nodes.append(node)

    return output_node


def onnx_transpose(nodes,layer_name, input_node, output_shape, perm=[0,1,2,3]):
    attrs = {
        "perm":perm
    }
    inputs = [input_node]
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Transpose", inputs=inputs, outputs=[output_node], attrs=attrs)

    nodes.append(node)

    return output_node

def onnx_concat(nodes,layer_name,input_node=[],output_shape=(),axis=1):
    attrs = {
        "axis":axis
    }
    inputs = input_node
    output_node = gs.Variable(layer_name,np.float32, output_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[output_node], attrs=attrs)

    nodes.append(node)

    return output_node

# ----------------------------------------------------------

def torch2onnx_with_onnxGraphAPI():
    gsapi = OnnxGraphAPI()
    bs = 8
    # input
    x = gsapi.layer_input("x", (bs, 3, 224, 224))
    x = gsapi.layer_conv('conv', x, (bs, 32, 112, 112),strides=(2,2))
    x = gsapi.layer_bn('bn',x,(bs,32, 112, 112))
    x = gsapi.layer_lrule('lrule',x,(bs,32, 112, 112))
    x = gsapi.layer_upsample('upsample',x,(bs,32,224,224))
    x = gsapi.layer_maxpool('maxpool',x,(bs,32,112,112))
    x = gsapi.layer_deconv('deconv',x,(bs,32,224,224),strides=(2,2))
    x = gsapi.layer_tanh('tanh',x,(bs,32,224,224))
    x = gsapi.layer_maxpool('maxpool2', x, (bs, 32, 112, 112))
    x = gsapi.layer_avgpool('avgpool',x,(bs,32,1,1),(112,112),(1,1))
    x = gsapi.layer_flatten('flatten',x,(bs,32),1)
    x = gsapi.layer_fc("fc",x,(bs,8))
    x = gsapi.layer_sigmoid('sigmoid',x,(bs,8))
    x = gsapi.layer_softmax('softmax',x,(bs,8),1)

    x = gsapi.layer_slice("slice",x,(bs,2),(0,2),(bs,4),(1,1))
    # x1 = gsapi.layer_slice("slice1",x,(bs,2),(0,0),(bs,2),(1,1))
    # x2 = gsapi.layer_slice("slice2",x,(bs,2),(0,2),(bs,4),(1,1))
    # x1 = gsapi.layer_mul("mul1",x1,gs.Constant("x11",np.ones([1,2],np.float32)*5),(bs,2))
    #
    # x = gsapi.layer_add("add1",x1,x2,(bs,2))

    # x = gsapi.layer_exp("exp",x,(bs,8))
    # x =gsapi.layer_slice('slice',x,(1,4),(0,0),(1,4),(1,1))

    # x1 = gsapi.layer_constant('c',np.ones([1,8],np.float32))
    # x = gsapi.layer_add("add",x,x1,(bs,8))
    # x = gsapi.layer_mul("mul",x,x1,(bs,8))

    # x = gsapi.layer_squeeze("squeeze",x,(8,),0)
    # x = gsapi.layer_unsqueeze('unsqueeze',x,(1,8),0)
    # x = gsapi.layer_transpose('transpose',x,(8,1),(1,0))
    # x = gsapi.layer_reshape("reshape",x,(1,2,4),(1,2,4))
    # x = gsapi.layer_clamp('clip',x,(1,8))
    gsapi.outputs.append(x)
    gsapi.save_model()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.lrelu = nn.LeakyReLU(0.1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.maxpool = nn.MaxPool2d(2, 2, 0)
        self.deconv = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        # self.gn = nn.GroupNorm(8,32)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(112, 1, 0)
        self.fc = nn.Linear(32, 8)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.upsample(x)
        x = self.maxpool(x)
        x = self.deconv(x)
        # x = self.gn(x)
        x = self.tanh(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        # x = x.permute(1,0)
        # x = x.view(1,2,4)
        # x = x.squeeze(0).unsqueeze(0)
        # x = x.clamp(0,1)
        # x = x[:,:4]+torch.ones([4])
        # x = x.mul(torch.ones([5,8]))
        # x = torch.exp(x)
        # x = x[:,:2]*torch.ones([1,2])*5+x[:,2:4]
        x = x[:,2:4]
        return x

class ModelData:
    PLUGIN_LIBRARY=None
    BATCH_SIZE = 1
    MEM_SIZE = 1
    DTYPE = trt.float16
    NP_DTYPE = np.float16
    OUTPUT_SIZE = (8,2)

def run_engine(data,engine_file_path="model.engine"):
    with loadEngine(engine_file_path) as engine,engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        data = data.ravel().astype(ModelData.NP_DTYPE)  # 展成一行
        np.copyto(inputs[0].host, data)
        [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream)
        output = np.reshape(output, ModelData.OUTPUT_SIZE)

        # print(output)

    return output
if __name__=="__main__":
    import onnxruntime
    x = torch.ones([8, 3, 224, 224])
    model = Net().eval()
    print(model(x))
    torch2npz(model)
    torch2onnx(model, x,'model.onnx')

    torch2onnx_with_onnxGraphAPI()
    ort_session = onnxruntime.InferenceSession("model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])


    # OnnxGraphAPI.resize_model()
    # OnnxGraphAPI.modeify_model()
    # OnnxGraphAPI.remove_model2(output_file="add.onnx")
    # OnnxGraphAPI.add_model()
    # OnnxGraphAPI.modeify_model2()


    # onnx2engine('model.onnx', ModelData=ModelData)
    # print(run_engine(x.numpy()))