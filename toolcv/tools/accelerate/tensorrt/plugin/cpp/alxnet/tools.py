import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import struct
from torchsummary import summary
import numpy as np


def torch2pth(model: torch.nn.Module, save_path: str = 'model.pth'):
    print('cuda device count: ', torch.cuda.device_count())
    if model is None:
        model = torchvision.models.alexnet(pretrained=True)
        model = model.to('cuda:0')
    model.eval()
    print(model)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = model(tmp)
    print('alexnet out:', out.shape)
    torch.save(model, save_path)

    summary(model, (3, 224, 224))


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


if __name__ == '__main__':
    model = torchvision.models.alexnet(pretrained=True)
    model = model.to('cuda:0')

    torch2onnx(model, torch.rand([1, 3, 224, 224]).to('cuda:0'))
    torch2wts(model)
