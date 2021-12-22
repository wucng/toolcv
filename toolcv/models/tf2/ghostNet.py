# @Time: 2021/4/8 13:08 
# @Author: wucong
# @File: ghostNet.py 
# @Software: PyCharm
# @Version: win10 Python 3.7.6
# @Version: torch 1.8.1+cu111 torchvision 0.9.1+cu111
# @Version: tensorflow 2.4.1+cu111  keras 2.4.3
# @Describe: https://arxiv.org/pdf/1911.11907.pdf
# https://github.com/d-li14/ghostnet.pytorch/blob/master/ghostnet.py

"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
# import torch
# import torch.nn as nn
import math

import tensorflow as tf
# from cnn_model.base.cnn_base import CNNBase

__all__ = ['ghost_net']

Model = tf.keras.Model
nn = tf.keras.layers
keras = tf.keras

tensor_shape = "NHWC"
# tensor_shape = "NCHW"
if tensor_shape == "NCHW":
    data_format = 'channels_first'
    # Conv2D;MaxPool2D;GlobalAveragePooling2D;AvgPool2D;Flatten;AvgPool2D
    dims = 1
    # tf.concat;Concatenate;nn.Reshape;tf.pad;tf.split;BatchNormalization
else:
    data_format = 'channels_last'
    dims = 3

# class ghostNet(CNNBase):
#     def __init__(self, **params):
#         super().__init__(params)
#
#     def build_model(self,input_shape,dropout,num_classes):
#         return ghost_net(num_classes=num_classes,dropout=dropout)

def create_model(num_classes,dropout):
    return ghost_net(num_classes=num_classes,dropout=dropout)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(Model):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = keras.Sequential([nn.GlobalAveragePooling2D(data_format=data_format),nn.Flatten(data_format=data_format)])
        self.fc = keras.Sequential(
                [
                    nn.Dense(channel // reduction),
                    nn.ReLU(),
                    nn.Dense(channel),
                    # nn.Reshape((1,1,channel)),
                    nn.Reshape((1, 1, channel)) if tensor_shape == "NHWC" else nn.Reshape((channel, 1, 1))
                ])

    def call(self, x):
        b, _, _,c = x.get_shape()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = tf.clip_by_value(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return keras.Sequential([
        nn.Conv2D(oup,kernel_size,stride,'same',groups=inp,use_bias=False,data_format=data_format),
        nn.BatchNormalization(dims),
        nn.ReLU() if relu else keras.Sequential([])
    ])


class GhostModule(Model):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = keras.Sequential([
            nn.Conv2D(init_channels,kernel_size,stride,'same',use_bias=False,data_format=data_format),
            nn.BatchNormalization(dims),
            nn.ReLU() if relu else keras.Sequential([])
        ])


        self.cheap_operation = keras.Sequential([
            nn.Conv2D(new_channels,dw_size,1,'same',groups=init_channels,use_bias=False,data_format=data_format),
            nn.BatchNormalization(dims),
            nn.ReLU() if relu else keras.Sequential([])
        ])

    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1,x2],dims)
        return out[:,:,:,:self.oup] if tensor_shape=="NHWC" else out[:,:self.oup,:,:]


class GhostBottleneck(Model):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = keras.Sequential([
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else keras.Sequential([]),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else keras.Sequential([]),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False)
        ])

        if stride == 1 and inp == oup:
            self.shortcut = keras.Sequential([])
        else:
            self.shortcut = keras.Sequential([
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2D(oup,1,1,'valid',use_bias=False,data_format=data_format),
                nn.BatchNormalization(dims)
            ])

    def call(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(Model):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.,dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [keras.Sequential([
            nn.Conv2D(output_channel,3,2,'same',use_bias=False,data_format=data_format),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ])]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = keras.Sequential(layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = keras.Sequential([
            nn.Conv2D(output_channel, 1, 1, 'valid', use_bias=False,data_format=data_format),
            nn.BatchNormalization(dims),
            nn.ReLU(),
            nn.GlobalAveragePooling2D(data_format=data_format),
            nn.Flatten(data_format=data_format)
        ])
        input_channel = output_channel

        output_channel = 1280
        self.classifier = keras.Sequential([
            nn.Dense(output_channel,use_bias=False),
            nn.BatchNormalization(dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Dense(num_classes)
        ])


    def call(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = self.classifier(x)
        return x

def ghost_net(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    import os
    import numpy as np
    # os.environ['CUDA_VISIBLE_DEVICES']='1'

    input_shape = (32, 32, 3)
    model = ghost_net()

    model(nn.Input(input_shape))
    model.summary()

    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])

    x = tf.convert_to_tensor(np.random.random([1, *input_shape]))
    pred = model.predict(x)
    print(pred.shape)
    model.save_weights("model.ckpt")