"""
# 模型加速
pytorch-->.onnx --> onnxruntime(实现模型推理 训练 GPU与CPU都可以)
onnxruntime
onnxruntime_gpu
onnxruntime_tensorrt_gpu
pytorch-->.onnx --> tensorrt （实现模型推理 GPU）
"""

import torch
import tensorrt as trt

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import ctypes
import onnx  # pip install onnx
import torch.onnx
# import onnxruntime # pip install onnxruntime
import onnxruntime  # pip install onnxruntime-gpu , https://pypi.org/project/onnxruntime-gpu
import onnxmltools  # pip install onnxmltools
from collections import OrderedDict
import numpy as np
from PIL import Image
import os
import struct

from torchvision import transforms as tfs
from torch.utils.data import DataLoader, Dataset
# from toolsmall.data import glob_format
from toolcv.tools.tools import glob_format

# import common
# from . import common
# from .layers import ModelData,populate_network
# from toolsmall.tools.speed import common
from toolcv.tools.accelerate.tensorrt.api import common
# from toolsmall.tools.speed.layers import ModelData,populate_network

import time
from functools import wraps, partial
# __all__=["saveTorchModel","saveTorchWeights","torch2npz",'torch2onnx','onnx2engine','weight2engine']

from .calibration import (
    TensorBatchDataset,
    DatasetCalibrator,
    DEFAULT_CALIBRATION_ALGORITHM,
)


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print("%s cost time: %s" % (func.__name__, end - start))
        return r

    return inner


def saveTorchWeights(model: torch.nn.Module, save_path="./model_weights.pth"):
    torch.save(model.state_dict(), save_path)  # 只保留模型权重文件
    # model.load_state_dict(torch.load(save_path))


def saveTorchModel(model: torch.nn.Module, save_path="./model.pth"):
    torch.save(model, save_path)  # 保留模型结构和权重文件，推理速度比直接保持权重文件的快
    # model = torch.load(save_path)


def runPth(x: torch.Tensor, model_path: str = "./model.pth"):
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        pred = torch.softmax(model(x), -1).topk(5, -1)

    return pred


@timeit
def runPth_batch(test_loader: torch.utils.data.DataLoader, model_path: str = "./model.pth", device="cpu"):
    model = torch.load(model_path)
    model.eval().to(device)
    pred_list = []
    with torch.no_grad():
        for data, target in test_loader:
            # pred = torch.softmax(model(x), -1).topk(5, -1)
            pred = torch.argmax(torch.softmax(model(data.to(device)), -1), -1)
            pred_list.extend(pred)
    return pred_list


@timeit
def runPth_batch2(test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device="cpu"):
    model.eval().to(device)
    pred_list = []
    with torch.no_grad():
        for data, target in test_loader:
            # pred = torch.softmax(model(x), -1).topk(5, -1)
            pred = torch.argmax(torch.softmax(model(data.to(device)), -1), -1)
            pred_list.extend(pred)
    return pred_list


# ------------------用于C++ tensorrt API---------------------------------
def torch2wts(model: torch.nn.Module, save_path: str = "./model.wts"):
    """用于C++ tensorrt API"""
    model.eval()
    with open(save_path, 'w') as fp:
        fp.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print("key:", k)
            print("value:", v.shape)
            vr = v.reshape(-1).cpu().numpy()
            fp.write("{} {}".format(k, len(vr)))
            for vv in vr:
                fp.write(" ")
                fp.write(struct.pack(">f", float(vv)).hex())
            fp.write("\n")


# --------------------用于python tensorrt API---------------------------------
def torch2npz(model: torch.nn.Module, save_path: str = "./model.npz"):
    model.eval()
    dict_weights = model.state_dict()
    # save to npz
    weights_arg = {}
    for key, value in dict_weights.items():
        weights_arg[key] = value.cpu().numpy()

    np.savez(save_path, **weights_arg)

    print("save to %s success!" % (save_path))
    # weights = np.load(save_path)


# --------------------------------------------------------------------
def torch2onnx(model: torch.nn.Module, x: torch.Tensor, save_path: str = "./model.onnx"):
    """pytorch模型保存成 .onnx格式"""
    # x = torch.rand([32,3,224,224])
    model.eval()
    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                      #               'output': {0: 'batch_size'}} # tensorrt 中没法执行，要注销这句
                      )

    # input_names = ["input"]
    # output_names = ["output"]
    #
    # torch.onnx.export(model, x, save_path, verbose=True, opset_version=8, input_names=input_names,
    #                   output_names=output_names)


def checkonnx(model_path: str = "./model.onnx"):
    """验证保存的.onnx格式是否正确"""
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def runOnnxInfer(x: torch.Tensor, model_path: str = "./model.onnx", batch_size: int = 32):
    """
    加载 .onnx文件执行推理，可以使用CPU或GPU，也可以结合tensorrt，
    单张图推理 速度比较快，多张图并行处理 推荐使用GPU
    """
    ort_session = onnxruntime.InferenceSession(model_path)
    # compute ONNX Runtime output prediction
    len_x = len(x)
    if len_x < batch_size:  # 不足一个batch 需要填充到一个batch 否则会报错
        tmp = torch.zeros([batch_size, *(x.shape[1:])])
        tmp[:len_x] = x
        x = tmp
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    ort_outs = ort_outs[0][:len_x]
    ort_outs = torch.softmax(torch.from_numpy(ort_outs), -1).topk(5, -1)

    return ort_outs


@timeit
def runOnnxInfer_batch(test_loader: torch.utils.data.DataLoader, model_path: str = "./model.onnx",
                       batch_size: int = 32):
    """
    加载 .onnx文件执行推理，可以使用CPU或GPU，也可以结合tensorrt，
    单张图推理 速度比较快，多张图并行处理 推荐使用GPU
    """
    ort_session = onnxruntime.InferenceSession(model_path)
    # compute ONNX Runtime output prediction
    pred_list = []
    for x, target in test_loader:
        len_x = len(x)
        if len_x < batch_size:  # 不足一个batch 需要填充到一个batch 否则会报错
            tmp = torch.zeros([batch_size, *(x.shape[1:])])
            tmp[:len_x] = x
            x = tmp
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        ort_outs = ort_outs[0][:len_x]
        # ort_outs = torch.softmax(torch.from_numpy(ort_outs), -1).topk(5, -1)
        pred = torch.argmax(torch.softmax(torch.from_numpy(ort_outs), -1), -1)
        pred_list.extend(pred)

    return pred_list


def keras2onnx2(all_file_onnx='mobilenet_save.onnx'):
    onnx_model = onnxmltools.convert_keras(base_model)
    onnx.save(onnx_model, all_file_onnx)


# --------------------------------------------------------------------

class ValidDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.paths = glob_format(root)
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        target = {"path": self.paths[idx]}
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def loadData(root, batch_size, resize=(256, 256), lasize=(224, 224)):
    tfs_val = tfs.Compose(
        [
            # tfs.Resize(lasize),
            tfs.Resize(resize),
            tfs.CenterCrop(lasize),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    valData = DataLoader(ValidDataset(root, transforms=tfs_val), batch_size)

    return valData


# --------------------------------------------------------------------

def onnx2engine(onnx_file_path: str = "./model.onnx", engine_file_path: str = "./model.engine", ModelData=None):
    # 加载自定义的plugin插件
    if ModelData.PLUGIN_LIBRARY:
        if not os.path.isfile(ModelData.PLUGIN_LIBRARY):
            raise IOError("\n{}\n{}\n{}\n".format(
                "Failed to load library ({}).".format(ModelData.PLUGIN_LIBRARY),
                "Please build the Clip sample plugin.",
                "For more information, see the included README.md"
            ))
        ctypes.CDLL(ModelData.PLUGIN_LIBRARY)

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = ModelData.BATCH_SIZE
            builder.max_workspace_size = common.GiB(ModelData.MEM_SIZE)  # 1 # 1 << 28 # 256MiB
            # builder.strict_type_constraints = False

            if ModelData.DTYPE == trt.float16:
                builder.fp16_mode = True
            elif ModelData.DTYPE == trt.int8:  # onnx有问题，官方例子caffe是可以的
                builder.int8_mode = True
                # default to use input tensors for calibration
                # if int8_calib_dataset is None:
                inputs = [ModelData.INPUT_SAMPLE]  # INPUT_SAMPLE=torch.ones([1,3,416,416],torch.float32)
                inputs_in = inputs

                # copy inputs to avoid modifications to source data
                inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry

                if isinstance(inputs, list):
                    inputs = tuple(inputs)
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)

                int8_calib_dataset = TensorBatchDataset(inputs_in)

                # @TODO(jwelsh):  Should we set batch_size=max_batch_size?  Need to investigate memory consumption
                builder.int8_calibrator = DatasetCalibrator(
                    inputs, int8_calib_dataset, batch_size=ModelData.BATCH_SIZE, algorithm=DEFAULT_CALIBRATION_ALGORITHM
                )

            else:
                pass
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
            # network.get_input(0).shape = ModelData.INPUT_SHAPE
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


# The UFF path is used for TensorFlow models. You can convert a frozen TensorFlow graph to UFF using the included convert-to-uff utility.
def uuf2engine(model_file: str = "./model.uff", engine_file_path: str = "./model.engine", ModelData=None):
    # 加载自定义的plugin插件
    if ModelData.PLUGIN_LIBRARY:
        if not os.path.isfile(ModelData.PLUGIN_LIBRARY):
            raise IOError("\n{}\n{}\n{}\n".format(
                "Failed to load library ({}).".format(ModelData.PLUGIN_LIBRARY),
                "Please build the Clip sample plugin.",
                "For more information, see the included README.md"
            ))
        ctypes.CDLL(ModelData.PLUGIN_LIBRARY)

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.UffParser() as parser:
            builder.max_batch_size = ModelData.BATCH_SIZE
            builder.max_workspace_size = common.GiB(ModelData.MEM_SIZE)  # 1 # 1 << 28 # 256MiB
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
            # Parse model file
            # We need to manually register the input and output nodes for UFF.
            parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
            parser.register_output(ModelData.OUTPUT_NAME)
            # Load the UFF model and parse it in order to populate the TensorRT network.
            if model_file.rsplit(".")[-1] == "pb":
                model_file = common.model_to_uff(model_file)
            parser.parse(model_file, network)
            print('Building an engine from file {}; this may take a while...'.format(model_file))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


# The Caffe path is used for Caffe2 models.
def caffe2engine(model_file="", deploy_file="", engine_file_path: str = "./model.engine", ModelData=None):
    # 加载自定义的plugin插件
    if ModelData.PLUGIN_LIBRARY:
        if not os.path.isfile(ModelData.PLUGIN_LIBRARY):
            raise IOError("\n{}\n{}\n{}\n".format(
                "Failed to load library ({}).".format(ModelData.PLUGIN_LIBRARY),
                "Please build the Clip sample plugin.",
                "For more information, see the included README.md"
            ))
        ctypes.CDLL(ModelData.PLUGIN_LIBRARY)

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.CaffeParser() as parser:
            builder.max_batch_size = ModelData.BATCH_SIZE
            builder.max_workspace_size = common.GiB(ModelData.MEM_SIZE)  # 1 # 1 << 28 # 256MiB
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
            # Load the Caffe model and parse it in order to populate the TensorRT network.
            # This function returns an object that we can query to find tensors by name.
            model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
            # For Caffe, we need to manually mark the output of the network.
            # Since we know the name of the output tensor, we can find it in model_tensors.
            network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
            print('Building an engine from file {}; this may take a while...'.format(model_file))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


# tensorrt python api
def weight2engine(model_file_path: str = "./model.pth", engine_file_path: str = "./model.engine", ModelData=None,
                  populate_network=None):
    fmt = os.path.splitext(model_file_path)[-1]
    if fmt == ".npz":
        weights = np.load(model_file_path)
    else:
        weights = torch.load(model_file_path)
        weights_arg = {}
        for key, value in weights.items():
            weights_arg[key] = value.cpu().numpy()
        weights = weights_arg

    def build_engine():
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
            populate_network(network, weights)
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


def loadEngine(engine_file_path: str = "./model.engine"):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


@timeit
def runEngineInfer(test_loader: torch.utils.data.DataLoader, engine_file_path: str = "./model.engine", ModelData=None):
    with loadEngine(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        pred_list = []
        for data, target in test_loader:
            data = data.numpy()  # data.numpy().ravel().astype(np.float32)  # 展成一行
            # 不足一个batch填充0,补齐
            len_data = len(data)
            if len_data != ModelData.BATCH_SIZE:
                tmp_data = np.zeros(ModelData.INPUT_SHAPE)
                tmp_data[:len_data] = data
                data = tmp_data

            data = data.ravel().astype(ModelData.NP_DTYPE)  # 展成一行
            np.copyto(inputs[0].host, data)

            # [output] = common.do_inference(context, bindings=bindings, \
            #             inputs=inputs, outputs=outputs, stream=stream, \
            #             batch_size=ModelData.BATCH_SIZE)

            [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                              stream=stream)

            output = np.reshape(output, [-1, ModelData.OUTPUT_SIZE])[:len_data]  # 转成[-1,10]
            pred = np.argmax(output, -1)
            # print("Prediction: " + str(pred))
            pred_list.extend(pred)

        return pred_list


# ------------------------自定义数据生成器------------------------------------------
import cv2


# from collections import Iterator,Iterable

class MyDataset(object):
    def __init__(self, root, transform=None,
                 resize=(72, 72),
                 crop_size=(64, 64),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]
                 ):
        self.paths = glob_format(root)
        self.transform = transform
        self.mean = np.asarray(mean)[None, None, :]
        self.std = np.array(std)[None, None, :]
        self.resize = resize
        self.crop_size = crop_size

    def _shuffle(self, seed=1):
        index = list(range(0, self.__len__()))
        np.random.seed(seed)
        np.random.shuffle(index)
        # self.paths = self.paths[index]
        self.paths = [self.paths[i] for i in index]
        return self

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        paths = self.paths[idx]
        imgs = []
        for path in paths:
            # imgs.append(self._load(path))
            imgs.append(self._load2(path))

        if paths:
            return np.stack(imgs, 0), paths
        else:
            return imgs, paths

    def _load(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.resize[::-1], interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # crop
        crop_s = (self.resize[0] // 2 - self.crop_size[0] // 2, self.resize[1] // 2 - self.crop_size[1] // 2)
        crop_e = (self.resize[0] // 2 + self.crop_size[0] // 2, self.resize[1] // 2 + self.crop_size[1] // 2)
        img = img[crop_s[0]:crop_e[0], crop_s[1]:crop_e[1]]

        # normanization
        img = img / 255.
        img = (img - self.mean) / self.std

        return img.astype(np.float32)

    def _load2(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = tfs.CenterCrop(self.crop_size)(tfs.Resize(self.resize)(img))

        # normanization
        img = np.asarray(img, np.float32)
        img = img / 255.
        img = (img - self.mean) / self.std

        return img.astype(np.float32)


class Generator(object):

    def __init__(self, DS, batch_size, shuffle=False, seed=1):
        self.DS = DS
        self.batch_size = batch_size
        self.index = 0
        if shuffle: self.DS._shuffle(seed)

    def __iter__(self):
        while True:
            strat = self.index * self.batch_size
            end = (self.index + 1) * self.batch_size
            imgs, paths = self.DS[strat:end]
            if len(paths) == 0: raise StopIteration("Iterative completion")
            yield (imgs, paths)
            self.index += 1


if __name__ == "__main__":
    from torchvision.models.resnet import resnet18

    model = resnet18(True)
    # save
    saveTorchModel(model)
    # torch2npz(model)
    # torch2onnx(model,torch.rand([32,3,224,224]))

    # inference
    # test_loader = loadData("/media/wucong/225A6D42D4FA828F1/datas/samples",32)
    # for data,target in test_loader:
    #     # print(runPth(data,"model.pth"))
    #     print(runOnnxInfer(data,"model.onnx",32))
    #     break

    # onnx2engine()
    # runEngineInfer(test_loader)
    # print(runPth_batch(test_loader))

    # weight2engine("model.npz")

    # [439 151 920 285 104 843 400 404 585 249  21 340 349 829 651 982 906 805 400 834 457 982 982 457 834 652 430 612 355]

    data = "/media/wucong/225A6D42D4FA828F1/datas/samples"
    Ds = MyDataset(data)
    for epoch in range(1):
        data_loader = Generator(Ds, batch_size=8, shuffle=True)
        for batch_x, batch_y in data_loader:
            if batch_y:
                print(len(batch_y))
