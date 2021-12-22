"""
# 打开 netron 查看生成的模型结构

参考：tensorrt 安装包的示例： xxx/tensorrt/samples/python/yolov3_onnx
使用onnx python api 将 pytorch模型转成 onnx格式
更多细节可以参考 torch.onnx.export 输出的打印信息
https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
"""
import onnx
from onnx import helper,TensorProto#,GraphProto,AttributeProto
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime
import tensorrt as trt

# from toolsmall.tools.speed.modelTansform import onnx2engine, \
#     torch2onnx, torch2npz,to_numpy,loadEngine,common
from toolcv.tools.accelerate.tensorrt.api.modelTansform import onnx2engine, \
    torch2onnx, torch2npz,to_numpy,loadEngine,common

class OnnxAPI:
    def __init__(self,model_path="model.npz",output_file_path='model.onnx',
                 alpha_lrelu = 0.1,epsilon_bn = 1e-5,momentum_bn = 0.99):
        self.weights = np.load(model_path)
        self.nodes = list()
        self.inputs = list()
        self.outputs = list()
        self.initializer = list()

        self.alpha_lrelu = alpha_lrelu
        self.epsilon_bn = epsilon_bn
        self.momentum_bn = momentum_bn
        self.graph_name = "model_test"
        self.output_file_path = output_file_path

    def layer_input(self,layer_name,input_shape):
        # input_shape:[batch_size, channels, height, width]
        input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT,input_shape)
        self.inputs.append(input_tensor)
        return self

    def layer_output(self,previous_node_name,output_shape):
        # output_shape:[batch_size, channels, height, width]
        output_tensor = helper.make_tensor_value_info(
            str(previous_node_name), TensorProto.FLOAT,output_shape)
        self.outputs.append(output_tensor)
        return self

    def layer_conv(self,layer_name,previous_node_name,kernel_shape=(3,3),strides=(1,1),pads=(1,1,1,1)):
        inputs = [previous_node_name]
        dilations = [1, 1]
        weights_name = layer_name+".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)

        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            dilations=dilations,
            group=1,
            name=layer_name
        )
        self.nodes.append(conv_node)
        return self

    def layer_deconv(self,layer_name,previous_node_name,kernel_shape=(3,3),strides=(2,2),pads=(1,1,1,1),output_padding=(1, 1)):
        inputs = [previous_node_name]
        dilations = [1, 1]
        weights_name = layer_name+".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)

        deconv_node = helper.make_node(
            'ConvTranspose',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            output_padding=output_padding,
            group=1,
            dilations=dilations,
            name=layer_name
        )
        self.nodes.append(deconv_node)
        return self

    def layer_bn(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        param_names = [layer_name+".weight",layer_name+".bias",layer_name+".running_mean",layer_name+".running_var"]
        inputs.extend(param_names)
        for param in param_names:
            self._create_param_tensors(param)

        batchnorm_node = helper.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=[layer_name],
            epsilon=self.epsilon_bn,
            momentum=self.momentum_bn,
            name=layer_name
        )
        self.nodes.append(batchnorm_node)

        return self

    def layer_lrelu(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        lrelu_node = helper.make_node(
            'LeakyRelu',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
            alpha=self.alpha_lrelu
        )
        self.nodes.append(lrelu_node)
        return self

    def layer_relu(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        relu_node = helper.make_node(
            'Relu',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(relu_node)
        return self

    def layer_tanh(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        tanh_node = helper.make_node(
            'Tanh',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(tanh_node)
        return self

    def layer_maxpool(self,layer_name,previous_node_name,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        inputs = [previous_node_name]
        maxpool_node = helper.make_node(
            'MaxPool',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            name=layer_name
        )

        self.nodes.append(maxpool_node)
        return self

    def layer_avgpool(self,layer_name,previous_node_name,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        inputs = [previous_node_name]
        avgpool_node = helper.make_node(
            'AveragePool',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            name=layer_name
        )

        self.nodes.append(avgpool_node)
        return self

    def layer_flatten(self,layer_name,previous_node_name,axis=1):
        inputs = [previous_node_name]
        flatten_node = helper.make_node(
            'Flatten',
            inputs=inputs,
            outputs=[layer_name],
            axis=axis,
            name=layer_name
        )

        self.nodes.append(flatten_node)
        return self

    def layer_fc(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        weights_name = layer_name + ".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)
        fc_node = helper.make_node(
            'Gemm',
            inputs=inputs,
            outputs=[layer_name],
            alpha=1.,
            beta=1.,
            transB=1,
            name=layer_name
        )

        self.nodes.append(fc_node)
        return self

    def layer_sigmoid(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(sigmoid_node)
        return self

    def layer_softmax(self,layer_name,previous_node_name,axis=1):
        inputs = [previous_node_name]
        softmax_node = helper.make_node(
            'Softmax',
            inputs=inputs,
            axis=axis,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(softmax_node)
        return self

    def layer_add(self,layer_name,first_node_name, second_node_name):
        inputs = [first_node_name, second_node_name]
        shortcut_node = helper.make_node(
            'Add',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(shortcut_node)
        return self

    def layer_mul(self, layer_name, first_node_name, second_node_name):
        inputs = [first_node_name, second_node_name]
        mul_node = helper.make_node(
            'Mul',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(mul_node)
        return self

    def layer_concat(self,layer_name,route_node_name=[]):
        inputs = route_node_name
        route_node = helper.make_node(
            'Concat',
            axis=1,
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(route_node)

        return self

    def layer_upsample(self,layer_name,previous_node_name,resize_scale_factors=2):
        scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
        scale_name = layer_name + ".scale"
        roi_name = layer_name + ".roi"
        inputs = [previous_node_name,roi_name,scale_name]

        resize_node = helper.make_node(
            'Resize',
            coordinate_transformation_mode='asymmetric',
            mode='nearest',
            nearest_mode='floor',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(resize_node)

        # 获取权重信息
        shape = scales.shape
        scale_init = helper.make_tensor(
            scale_name, TensorProto.FLOAT,shape, scales)
        scale_input = helper.make_tensor_value_info(
            scale_name, TensorProto.FLOAT, shape)
        self.initializer.append(scale_init)
        self.inputs.append(scale_input)

        # In opset 11 an additional input named roi is required. Create a dummy tensor to satisfy this.
        # It is a 1D tensor of size of the rank of the input (4)
        rank = 4
        roi_input = helper.make_tensor_value_info(roi_name, TensorProto.FLOAT, [rank])
        roi_init = helper.make_tensor(roi_name, TensorProto.FLOAT, [rank], [0, 0, 0, 0])
        self.initializer.append(roi_init)
        self.inputs.append(roi_input)

        return self

    def layer_transpose(self,layer_name,previous_node_name,perm=[0,1,2,3]):
        inputs = [previous_node_name]
        shortcut_node = helper.make_node(
            'Transpose',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
            perm=perm
        )
        self.nodes.append(shortcut_node)
        return self

    def layer_constant(self,layer_name,value,dtype=TensorProto.INT64):
        constant_input = helper.make_tensor_value_info(layer_name, dtype, [len(value)])

        constant_init = helper.make_tensor(
            layer_name,
            dtype,
            [len(value)],
            vals=value
        )
        self.inputs.append(constant_input)
        self.initializer.append(constant_init)
        return self

    def layer_reshape(self,layer_name,previous_node_name,value):
        inputs = [previous_node_name]
        self.layer_constant(layer_name+'_constant',value,TensorProto.INT32)
        inputs.append(layer_name+'_constant')

        reshape_node = helper.make_node(
            "Reshape",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(reshape_node)
        return self

    def layer_slice2(self,layer_name,previous_node_name,start=(0,0,0,0), shape=(2,2,3,3), stride=(1,1,1,1)):
        inputs = [previous_node_name]
        for i,st in enumerate(start):
            self.layer_constant(layer_name+'_constant_axis_%d'%i,[i])
            self.layer_constant(layer_name+'_constant_start_%d'%i,[start[i]])
            self.layer_constant(layer_name+'_constant_shape_%d'%i,[shape[i]])
            self.layer_constant(layer_name+'_constant_stride_%d'%i,[stride[i]])
            inputs.extend([layer_name+'_constant_start_%d'%i,
                           layer_name+'_constant_shape_%d'%i,
                           layer_name + '_constant_axis_%d' % i,
                           layer_name+'_constant_stride_%d'%i])

            if i ==len(start)-1:
                name = layer_name
            else:
                name = layer_name+"_%s"%i


            slice_node = helper.make_node(
                "Slice",
                inputs=inputs,
                outputs=[name],
                name=name
            )
            self.nodes.append(slice_node)

            inputs = [name]

        return self

    def layer_slice(self,layer_name,previous_node_name,start=(0,0,0,0), shape=(2,2,3,3), stride=(1,1,1,1)):
        """
        x = torch.randn([8,8])
        x[:,2:4]

        layer_slice("slice","x",(0,2),(8,4),(1,1))

        """
        inputs = [previous_node_name]

        self.layer_constant(layer_name + '_constant_axis', list(range(len(start))))
        self.layer_constant(layer_name + '_constant_start', start)
        self.layer_constant(layer_name + '_constant_shape', shape)
        self.layer_constant(layer_name + '_constant_stride', stride)
        inputs.extend([layer_name + '_constant_start',
                       layer_name + '_constant_shape',
                       layer_name + '_constant_axis',
                       layer_name + '_constant_stride'])

        slice_node = helper.make_node(
            "Slice",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(slice_node)

        return self


    def layer_squeeze(self,layer_name,previous_node_name,axes=0):
        inputs = [previous_node_name]
        squeeze_node = helper.make_node(
            'Squeeze',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
            axes=[axes]
        )
        self.nodes.append(squeeze_node)
        return self

    def layer_unsqueeze(self,layer_name,previous_node_name,axes=0):
        inputs = [previous_node_name]
        unsqueeze_node = helper.make_node(
            'Unsqueeze',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
            axes=[axes]
        )
        self.nodes.append(unsqueeze_node)
        return self

    def layer_clamp(self,layer_name,previous_node_name,min_value=0.0,max_value=1.0,dtype=TensorProto.FLOAT):
        inputs = [previous_node_name]
        self.layer_constant(layer_name + '_min', [min_value],dtype)
        inputs.append(layer_name + '_min')
        self.layer_constant(layer_name + '_max', [max_value],dtype)
        inputs.append(layer_name + '_max')

        clip_node = helper.make_node(
            "Clip",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(clip_node)
        return self


    def _create_param_tensors(self,param_name):
        param_data = self.weights[param_name]
        param_data_shape = param_data.shape
        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data.ravel())
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape)

        self.initializer.append(initializer_tensor)
        self.inputs.append(input_tensor)
        return self

    def save_model(self):
        graph_def = helper.make_graph(
            nodes=self.nodes,
            name=self.graph_name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializer
        )
        print(helper.printable_graph(graph_def))

        model_def = helper.make_model(graph_def, producer_name='NVIDIA TensorRT sample')
        onnx.checker.check_model(model_def)
        onnx.save(model_def, self.output_file_path)


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
        # x = x.view(8,1)
        # x = x.squeeze(0).unsqueeze(0)
        # x = x.clamp(0.12,0.13)
        x = x[:,:4]
        # x = x.mul(torch.ones([5,8]))
        return x

def torch2onnx_with_onnxAPI(
        input_shape = [1, 3, 224, 224],
        output_shape = [1,8]):
    # -------------使用onnx python api 转成 onnx文件--------------------------------------------
    oapi = OnnxAPI()
    oapi.layer_input("input", input_shape)

    oapi.layer_conv("conv", "input", (3, 3), (2, 2))
    oapi.layer_bn("bn", "conv")
    oapi.layer_lrelu('lrelu', 'bn')
    oapi.layer_upsample('upsample', 'lrelu', 2)
    oapi.layer_maxpool('maxpool', 'upsample')
    oapi.layer_deconv('deconv', 'maxpool', (3, 3), (2, 2))
    oapi.layer_tanh('tanh', 'deconv')
    oapi.layer_maxpool('maxpool2', 'tanh')
    oapi.layer_avgpool('avgpool', 'maxpool2', (112, 112), (1, 1))
    oapi.layer_flatten('flatten', 'avgpool', 1)
    oapi.layer_fc('fc', 'flatten')
    oapi.layer_sigmoid('sigmoid', 'fc')
    oapi.layer_softmax('softmax', "sigmoid", 1)
    # oapi.layer_squeeze("squeeze","softmax",0)
    # oapi.layer_unsqueeze("unsqueeze","squeeze",0)
    # oapi.layer_reshape("reshape","softmax",output_shape)
    # oapi.layer_transpose('transpose','softmax',[1,0])
    # oapi.layer_slice('slice','softmax',(0,0),(4,5),(1,1))
    # oapi.layer_clamp('clip', 'softmax',0.12,0.13)
    oapi.layer_slice("slice",'softmax',(0,0),(1,4),(1,1))

    # oapi.layer_constant('const',np.ones([5,8],np.float32).flatten(),TensorProto.FLOAT)
    # oapi.layer_reshape('reshape','const',[5,8])
    # oapi.layer_mul('mul','softmax','reshape')
    oapi.layer_output('slice', output_shape)

    oapi.save_model()


class ModelData:
    PLUGIN_LIBRARY=None
    BATCH_SIZE = 1
    MEM_SIZE = 1
    DTYPE = trt.float16
    NP_DTYPE = np.float16
    OUTPUT_SIZE = (1,4)

if __name__=="__main__":
    model = Net().eval()

    # x = torch.randn([1,3,224,224])
    x = torch.ones([1, 3, 224, 224])
    print(model(x))
    torch2npz(model)
    # torch2onnx(model, x) # torch.onnx.export
    # exit(0)
    torch2onnx_with_onnxAPI(output_shape=(1,4))
    # exit(0)
    # onnx2engine('model.onnx',ModelData=ModelData)

    ort_session = onnxruntime.InferenceSession("model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])

    # print(run_engine(x.numpy()))
