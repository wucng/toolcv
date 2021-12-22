import torch
from torch import nn
import numpy as np
import os

from toolcv.tools.accelerate.tensorrt.python.torch2onnx import torch2npz, export_to_onnx, onnx_inference
from toolcv.tools.accelerate.tensorrt.python.onnx2trt import onnx2engine, runEngineInfer, weight2engine


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

data = np.ones([1, 3, 32, 32], np.float32)
# x = torch.ones([1,3,32,32],dtype=torch.float32)
x = torch.from_numpy(data)
print("------run pytorch-------------")
print(model(x))

export_to_onnx(model, (x,), dynamic_batch=False)
torch2npz(model)

print("------run onnx-------------")
onnx_inference(data=data)

print("------onnx2trt-------------")
onnx2engine()
runEngineInfer(data)
os.remove("model.trt")

print("------weight2trt trt api-------------")
weight2engine()
runEngineInfer(data)
