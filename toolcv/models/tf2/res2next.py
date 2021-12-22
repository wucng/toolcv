# @Time: 2021/4/8 15:47 
# @Author: wucong
# @File: res2next.py
# @Software: PyCharm
# @Version: win10 Python 3.7.6
# @Version: torch 1.8.1+cu111 torchvision 0.9.1+cu111
# @Version: tensorflow 2.4.1+cu111  keras 2.4.3
# @Describe: https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2next.py

import math

import tensorflow as tf
# from cnn_model.base.cnn_base import CNNBase

__all__ = ['res2next50']

Model = tf.keras.Model
nn = tf.keras.layers
keras = tf.keras

tensor_shape = "NHWC"
# tensor_shape = "NCHW"
if tensor_shape == "NCHW":
    data_format = 'channels_first'
    # Conv2D;MaxPool2D;GlobalAveragePooling2D;AvgPool2D;Flatten;AvgPool2D
    dims = 1
    # tf.concat;Concatenate;nn.Reshape;tf.pad;tf.split;BatchNormalization(dims)
else:
    data_format = 'channels_last'
    dims = 3

# class res2next(CNNBase):
#     def __init__(self, **params):
#         super().__init__(params)
#
#     def build_model(self,input_shape,dropout,num_classes):
#         # layers = [3, 4, 23, 3] # 101
#         layers = [3, 4, 6, 3] # 50
#         return Res2NeXt(Bottle2neckXSE, layers = layers, baseWidth = 4,
#                         cardinality=8, scale = 4, num_classes=num_classes,dropout=dropout)

def create_model(num_classes,dropout):
    layers = [3, 4, 6, 3]  # 50
    return Res2NeXt(Bottle2neckXSE, layers = layers, baseWidth = 4,
                        cardinality=8, scale = 4, num_classes=num_classes,dropout=dropout)


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


class Bottle2neckX(Model):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality


        self.conv1 = nn.Conv2D(D*C*scale,1,1,'valid',use_bias=False,data_format=data_format)
        self.bn1 = nn.BatchNormalization(dims)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(3,stride,'same',data_format=data_format)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2D(D*C,3,stride,'same',groups=C,use_bias=False,data_format=data_format))
          bns.append(nn.BatchNormalization(dims))
        self.convs = convs
        self.bns = bns

        self.conv3 = nn.Conv2D(planes * 4,1,1,'valid',use_bias=False,data_format=data_format)
        self.bn3 = nn.BatchNormalization(dims)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = tf.split(out,self.scale,dims)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = tf.concat((out, sp),dims)
        if self.scale != 1 and self.stype=='normal':
          out = tf.concat((out, spx[self.nums]), dims)
        elif self.scale != 1 and self.stype=='stage':
          out = tf.concat((out, self.pool(spx[self.nums])), dims)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottle2neckXSE(Model):
    # 加入SE 模块
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckXSE, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality


        self.conv1 = nn.Conv2D(D*C*scale,1,1,'valid',use_bias=False,data_format=data_format)
        self.bn1 = nn.BatchNormalization(dims)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(3,stride,'same',data_format=data_format)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2D(D*C,3,stride,'same',groups=C,use_bias=False,data_format=data_format))
          bns.append(nn.BatchNormalization(dims))
        self.convs = convs
        self.bns = bns

        self.conv3 = nn.Conv2D(planes * 4,1,1,'valid',use_bias=False,data_format=data_format)
        self.bn3 = nn.BatchNormalization(dims)
        self.relu = nn.ReLU()

        self.seblock = SELayer(planes * 4)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = tf.split(out,self.scale,dims)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = tf.concat((out, sp),dims)
        if self.scale != 1 and self.stype=='normal':
          out = tf.concat((out, spx[self.nums]), dims)
        elif self.scale != 1 and self.stype=='stage':
          out = tf.concat((out, self.pool(spx[self.nums])),dims)

        out = self.conv3(out)
        out = self.bn3(out)

        # seblock
        out = self.seblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2NeXt(Model):
    def __init__(self, block, baseWidth, cardinality, layers, num_classes, scale=4,dropout=0.2):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
            scale: scale in res2net
        """
        super(Res2NeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.scale = scale

        self.conv1 = nn.Conv2D(64,7,2,'same',use_bias=False,data_format=data_format)
        self.bn1 = nn.BatchNormalization(dims)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(3,2,'same',data_format=data_format)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = keras.Sequential([nn.GlobalAveragePooling2D(data_format=data_format),
                                         nn.Flatten(data_format=data_format),nn.Dropout(dropout)])
        self.fc = nn.Dense(num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                nn.Conv2D(planes * block.expansion,1,stride,use_bias=False,data_format=data_format),
                nn.BatchNormalization(dims)
            ])

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return keras.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

def res2next50(**kwargs):
    """    Construct Res2NeXt-50.
    The default scale is 4.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        # layers = [3, 4, 23, 3] # 101
        layers = [3, 4, 6, 3] # 50
    """
    # model = Res2NeXt(Bottle2neckX, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4, num_classes=1000)
    model = Res2NeXt(Bottle2neckXSE, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4, **kwargs)

    return model

if __name__=='__main__':
    import os
    import numpy as np
    # os.environ['CUDA_VISIBLE_DEVICES']='1'

    input_shape = (32, 32, 3)
    model = res2next50()

    model(nn.Input(input_shape))
    model.summary()

    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])

    x = tf.convert_to_tensor(np.random.random([1, *input_shape]))
    pred = model.predict(x)
    print(pred.shape)
