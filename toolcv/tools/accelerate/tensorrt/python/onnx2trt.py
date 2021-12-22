from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys, os

# sys.path.insert(1, os.path.join(sys.path[0], "."))
# import common
# import layers_trt as lytrt
from toolcv.tools.accelerate.tensorrt.api import common
from toolcv.tools.accelerate.tensorrt.api.layers import layers_trt as lytrt

import time
from functools import wraps, partial

TRT_LOGGER = trt.Logger()


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print("%s cost time: %s" % (func.__name__, end - start))
        return r

    return inner


class DefModelData:
    PLUGIN_LIBRARY = None
    BATCH_SIZE = 1
    MEM_SIZE = 1 << 28  # 256MiB  ; 1 << 28/1024/1024=256
    # MEM_SIZE = common.GiB(1)  # 1G

    DTYPE = trt.float16
    NP_DTYPE = np.float16
    INPUT_SHAPE = [1, 3, 32, 32]
    OUTPUT_SIZE = [-1, 10]  # [-1,10]

    onnx_file_path = "model.onnx"
    engine_file_path = "model.trt"

    model_file_path = "model.npz"
    INPUT_NAME = 'input'
    OUTPUT_NAME = 'output'


def defpopulate_network(network, weights, ModelData):
    # ----------------------------------------------
    # dtype = ModelData.NP_DTYPE
    dtype = np.float32
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=trt.float32, shape=ModelData.INPUT_SHAPE)

    conv1 = lytrt.trt_conv(network, input_tensor, "layer1.0", weights, dtype, (3, 3), (2, 2), (1, 1))
    bn1 = lytrt.trt_bn(network, conv1.get_output(0), "layer1.1", weights, dtype)
    relu1 = lytrt.trt_active(network, bn1.get_output(0))
    maxpool1 = lytrt.trt_pool(network, relu1.get_output(0), trt.PoolingType.MAX, (3, 3), (2, 2), (1, 1))
    maxpool2 = lytrt.trt_pool(network, maxpool1.get_output(0), trt.PoolingType.MAX, (3, 3), (2, 2), (1, 1))
    upsample = lytrt.trt_upsample(network, maxpool2.get_output(0))
    fc1 = lytrt.trt_fc(network, upsample.get_output(0), "fc", weights, dtype)
    # ------------------------------------------------
    softmax = network.add_softmax(fc1.get_output(0))
    softmax.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax.get_output(0))


def weight2engine(ModelData=None, populate_network=None):
    """
    - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html

    通过 tensorrt api 从头搭建网络
    tensorrt api 实现对应的pytorch模型

    1、torch model 转 onnx
    2、使用 python3 -m onnxsim input_onnx_model output_onnx_model （pip install onnx-simplifier） 简化模型
    3、打开 Netron 查看模型结构
    4、根据 #3 中的模型结构 使用 tensorrt api 重新搭建
    5、生成 序列化文件 .trt or .engine
    """
    if ModelData is None: ModelData = DefModelData

    model_file_path = ModelData.model_file_path
    engine_file_path = ModelData.engine_file_path

    fmt = os.path.splitext(model_file_path)[-1]
    if fmt == ".npz":
        weights = np.load(model_file_path)
    else:
        weights = torch.load(model_file_path)
        weights_arg = {}
        for key, value in weights.items():
            weights_arg[key] = value.cpu().numpy()
        weights = weights_arg
        # np.savez(save_path, **weights_arg)

    if populate_network is None: populate_network = defpopulate_network

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network:
            builder.max_batch_size = ModelData.BATCH_SIZE
            builder.max_workspace_size = ModelData.MEM_SIZE  # 1 << 28 # 256MiB
            if ModelData.DTYPE == trt.float16:
                builder.fp16_mode = True
            # Populate the network using weights from the PyTorch model.
            populate_network(network, weights, ModelData)
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


def onnx2engine(ModelData=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    if ModelData is None: ModelData = DefModelData

    onnx_file_path = ModelData.onnx_file_path
    engine_file_path = ModelData.engine_file_path

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = ModelData.MEM_SIZE
            builder.max_batch_size = ModelData.BATCH_SIZE

            if ModelData.DTYPE == trt.float16:
                builder.fp16_mode = True

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]
            if 'INPUT_SHAPE' in ModelData.__dict__.keys():
                network.get_input(0).shape = ModelData.INPUT_SHAPE

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def onnx2trt(ModelData=None):
    # from toolsmall.tools.speed.modelTansform import onnx2engine
    from toolcv.tools.accelerate.tensorrt.api.modelTansform import onnx2engine

    if ModelData is None: ModelData = DefModelData
    onnx2engine(ModelData.onnx_file_path, ModelData.engine_file_path, ModelData)


def torchToTrt(model=None, input_data=None, device='cuda'):
    """torch2trt 0.3.0
    https://toscode.gitee.com/Liusing/torch2trt
    """
    from torch2trt import torch2trt
    import torch
    from torch import nn
    if input_data is None: input_data = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    if model is None:

        class Mymodel(nn.Module):
            def __init__(self, mode=0):
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

        model = Mymodel().to(device)
    input_data = input_data.to(device)
    # model_trt_int8 = torch2trt(model.eval(), [input_data], max_batch_size=int(input_data.size(0)), int8_mode=True)
    model_trt_int16 = torch2trt(model.eval(), [input_data], max_batch_size=int(input_data.size(0)), fp16_mode=True)
    out_trt = model_trt_int16(input_data)
    print(out_trt)

    # Save and load
    torch.save(model_trt_int16.state_dict(), 'model_trt.pth')

    from torch2trt import TRTModule

    model_trt = TRTModule()

    model_trt.load_state_dict(torch.load('model_trt.pth'))


def onnxToTrt(onnx_file='model.onnx', input_data=None, device='CUDA:0'):
    """https://github.com/onnx/onnx-tensorrt
    https://github.com/onnx/onnx-tensorrt/tree/7.2.1  # 要与tensorrt 版本对应
    onnx2trt my_model.onnx -o my_engine.trt
    """
    # ONNX-TensorRT Python Backend Usage
    import onnx
    import onnx_tensorrt.backend as backend
    import numpy as np
    # import pickle

    if input_data is None:
        input_data = np.random.random(size=(1, 3, 32, 32)).astype(np.float32)
    model = onnx.load(onnx_file)
    engine = backend.prepare(model, device=device, int16_mode=True)  # int8_mode=True
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)

    # pickle.dump(engine,open('model.pkl','wb'))


def loadEngine(engine_file_path: str = "./model.engine"):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


@timeit
def runEngineInfer(data=np.ones([1, 3, 32, 32]), ModelData=None):
    if ModelData is None: ModelData = DefModelData
    engine_file_path = ModelData.engine_file_path
    with loadEngine(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        len_data = len(data)
        data = data.ravel().astype(ModelData.NP_DTYPE)  # 展成一行
        np.copyto(inputs[0].host, data)

        # [output] = common.do_inference(context, bindings=bindings, \
        #             inputs=inputs, outputs=outputs, stream=stream, \
        #             batch_size=ModelData.BATCH_SIZE)

        [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream)

        output = np.reshape(output, ModelData.OUTPUT_SIZE)[:len_data]  # 转成[-1,10]
        pred = np.argmax(output, -1)
        print(output)
        print("Prediction: " + str(pred))


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('-onnx', default='model.onnx', type=str, help='onnx model')
    args.add_argument('-trt', default='model.trt', type=str, help='trt model')
    args.add_argument('-mode', default=0, type=int, help='0: onnx2trt 1:trtinfer')
    arg = args.parse_args()

    DefModelData.onnx_file_path = arg.onnx
    DefModelData.engine_file_path = arg.trt

    if arg.mode == 0:
        # onnx2engine()
        # onnx2trt()
        weight2engine()
    else:
        runEngineInfer()
