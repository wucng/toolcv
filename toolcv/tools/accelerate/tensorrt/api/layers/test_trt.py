"""
import tensorrt as trt
TRT_LOGGER = trt.Logger()
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1)
parser = trt.OnnxParser(network, TRT_LOGGER)
"""
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
import numpy as np
from torch.nn import functional as F
import torch

# from toolsmall.tools.speed.modelTansform import common,loadEngine
# from toolsmall.tools.speed.layers.layers_trt import trt_slice,trt_active,trt_concat,trt_transpose,trt_softmax
from toolcv.tools.accelerate.tensorrt.api.modelTansform import common, loadEngine
from toolcv.tools.accelerate.tensorrt.api.layers.layers_trt import trt_slice, trt_active, trt_concat, trt_transpose, \
    trt_softmax


class ModelData(object):
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"  #
    BATCH_SIZE = 5  # 要与 pytorch2onnx 时设置的batch size 对应
    INPUT_SHAPE = (5, 8, 8, 85)
    OUTPUT_SIZE = (5, 8, 8, 85)
    # DTYPE = trt.float16  # 使用半精度 half-float
    # NP_DTYPE = np.float16
    DTYPE = trt.float32
    NP_DTYPE = np.float32
    MEM_SIZE = 1  # 1G内存


def populate_network(network):
    # Configure the network layers based on the weights provided.
    input_size = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    layer1 = trt_slice(network, input_size, (0, 0, 0, 0), (5, 8, 8, 5), (1, 1, 1, 1))  # [5,8,8,5]
    layer2 = trt_slice(network, input_size, (0, 0, 0, 5), (5, 8, 8, 80), (1, 1, 1, 1))  # [5,8,8,80]
    # layer2 = trt_active(network, layer2.get_output(0), trt.ActivationType.SIGMOID)
    layer1 = trt_transpose(network, layer1.get_output(0), (0, 3, 1, 2))  # [5,5,8,8]
    layer2 = trt_transpose(network, layer2.get_output(0), (0, 3, 1, 2))  # [5,80,8,8]
    layer2 = trt_softmax(network, layer2.get_output(0))

    # concat
    layer = trt_concat(network, [layer1.get_output(0), layer2.get_output(0)])  # 只能按第2个维度合并 [5,85,8,8]
    layer = trt_transpose(network, layer.get_output(0), (0, 2, 3, 1))  # [5,8,8,85]

    layer.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=layer.get_output(0))


def build_engine(engine_file_path="model.engine"):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network:
        builder.max_batch_size = ModelData.BATCH_SIZE
        builder.max_workspace_size = common.GiB(ModelData.MEM_SIZE)  # 1 << 28 # 256MiB
        if ModelData.DTYPE == trt.float16:
            builder.fp16_mode = True
        elif ModelData.DTYPE == trt.int8:  # onnx有问题，官方例子caffe是可以的
            builder.int8_mode = True

            # Now we create a calibrator and give it the location of our calibration data.
            # We also allow it to cache calibration data for faster engine building.
            calibration_cache = "calibration.cache"
            calib = common.MNISTEntropyCalibrator(ModelData.data_dir, ModelData.INPUT_SHAPE[-2:],
                                                  cache_file=calibration_cache, batch_size=ModelData.BATCH_SIZE)
            builder.int8_calibrator = calib

        else:
            pass
        # Populate the network using weights from the PyTorch model.
        populate_network(network)
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


def run_engine(data, engine_file_path="model.engine"):
    with loadEngine(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        data = data.ravel().astype(ModelData.NP_DTYPE)  # 展成一行
        np.copyto(inputs[0].host, data)
        [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream)
        output = np.reshape(output, ModelData.OUTPUT_SIZE)

        # print(output)

    return output


if __name__ == "__main__":
    # x = np.random.random_sample((5,8,8,85)).astype(np.float32)
    x = np.ones((5, 8, 8, 85), np.float32)
    x[:3] *= 5

    build_engine()
    trt_x = run_engine(x)

    torch_x = torch.from_numpy(x)
    torch_x[..., 5:] = torch.softmax(torch_x[..., 5:], -1)

    print(np.max(np.abs(torch_x.numpy() - trt_x)))
