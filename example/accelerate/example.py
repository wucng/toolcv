import torch
from torch import nn
import numpy as np
import os
import struct

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


model = Mymodel().eval()
x = torch.ones([1,3,32,32],dtype=torch.float32)
export_to_onnx(model, (x,), dynamic_batch=False)
torch2npz(model)
torch2wts(model)
