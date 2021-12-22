"""
1、论文、博客
2、代码（github）
3、转onnx模型，onnx-simplifier 简化 onnx模型，使用netron查看

pip install onnx-simplifier

python3 -m onnxsim input_onnx_model output_onnx_model # 简化 onnx模型

pip install timm
mmcv,mmdetect
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import struct
# from torchsummary import summary
import numpy as np


def model_summary(model):
    # pip install torchsummary
    from torchsummary import summary
    summary(model.cuda(), input_size=(3, 224, 224), batch_size=-1)


def model_stats(model):
    # pip install torchstat
    from torchstat import stat
    stat(model, (3, 224, 224))


def model_profile(model, input):
    # pip install thop
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    return flops, params


def print_model(model):
    # 打印模型名称与shape
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# ------------------------------------------------------------------------------
def torch2pth(model: torch.nn.Module, save_path: str = 'model.pth', device="cpu"):
    # print('cuda device count: ', torch.cuda.device_count())
    model.eval()
    torch.save(model, save_path)
    # summary(model, (3, 224, 224))


# ------------------用于C++ tensorrt API---------------------------------
def torch2wts(model: torch.nn.Module, save_path: str = "./model.wts"):
    """用于C++ tensorrt API"""
    model.eval()
    with open(save_path, 'w') as fp:
        fp.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print("key:", k, "value:", v.shape)
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
        print(key, value.shape)
        weights_arg[key] = value.cpu().numpy()

    np.savez(save_path, **weights_arg)

    print("save to %s success!" % (save_path))
    # weights = np.load(save_path)


# --------------------------------------------------------------------
def torch2onnx(model: torch.nn.Module, x: torch.Tensor, save_path: str = "./model.onnx", dynamic_axes=False):
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
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}} if dynamic_axes else None  # tensorrt 中没法执行，要注销这句
                      )

    # input_names = ["input"]
    # output_names = ["output"]
    #
    # torch.onnx.export(model, x, save_path, verbose=True, opset_version=8, input_names=input_names,
    #                   output_names=output_names)


def export_to_onnx(model, inputs=(torch.rand([1, 3, 224, 224]),),
                   export_file="model.onnx", opset_version=11, dynamic_batch=False):
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


if __name__ == '__main__':
    model = torchvision.models.alexnet(pretrained=True)
    model = model.to('cuda:0')

    torch2onnx(model, torch.rand([1, 3, 224, 224]).to('cuda:0'))
    torch2wts(model)
