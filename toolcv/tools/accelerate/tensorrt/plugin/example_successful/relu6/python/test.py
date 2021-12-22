from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import ctypes
import sys, os

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'build/libmyplugin.so'
)
ctypes.CDLL(CLIP_PLUGIN_LIBRARY)
# lib = ctypes.cdll.LoadLibrary(CLIP_PLUGIN_LIBRARY)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


class DefModelData:
    PLUGIN_LIBRARY = None
    BATCH_SIZE = 1
    MEM_SIZE = 1 << 28  # 256M ; 1 << 28/1024/1024=256

    DTYPE = trt.float32  # trt.float32
    NP_DTYPE = np.float32
    INPUT_SHAPE = [1, 1, 1]
    OUTPUT_SIZE = [-1, 1]

    onnx_file_path = "model.onnx"
    engine_file_path = "model.trt"
    weight_file_path = "model.wts"

    INPUT_NAME = 'input'
    OUTPUT_NAME = 'output'


def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        # print(plugin_creator.name)
        if plugin_creator.name == plugin_name:
            # clipMin_field = trt.PluginField("clipMin", np.array([0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            # clipMax_field = trt.PluginField("clipMax", np.array([6.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)

            # field_collection = trt.PluginFieldCollection([clipMin_field, clipMax_field])
            field_collection = trt.PluginFieldCollection()
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin


def populate_network(network, weights=None, ModelData=None):
    # Configure the network layers based on the weights provided.
    input_layer = network.add_input(name=ModelData.INPUT_NAME, dtype=trt.float32, shape=ModelData.INPUT_SHAPE)
    relu = network.add_plugin_v2(inputs=[input_layer], plugin=get_trt_plugin("Relu6_TRT"))
    relu.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(relu.get_output(0))


def build_engine(weights=None, ModelData=None):
    engine_file_path = ModelData.engine_file_path
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_batch_size = ModelData.BATCH_SIZE  # batch size
        builder.max_workspace_size = ModelData.MEM_SIZE
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights, ModelData)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


# -----------running engine ------------------------------------------------


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def inference(ModelData=None):
    if ModelData is None: ModelData = DefModelData
    ENGINE_PATH = ModelData.engine_file_path
    INPUT_SHAPE = ModelData.INPUT_SHAPE

    runtime = trt.Runtime(TRT_LOGGER)
    assert runtime

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    data = np.ones((np.prod(INPUT_SHAPE)), dtype=ModelData.NP_DTYPE) * 8
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = data

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs[0])


if __name__ == "__main__":
    modelData = DefModelData()
    build_engine(None, modelData)

    inference(modelData)
