# @Time: 2021/4/2 09:10
# @Author: wucong
# @File: SE_Inception_resNet_v2.py
# @Software: PyCharm
# @Version: win10 Python 3.7.6
# @Version: torch 1.8.1+cu111 torchvision 0.9.1+cu111
# @Version: tensorflow 2.4.1+cu111  keras 2.4.3
# @Describe:
# keras模型构建的三种方式: https://blog.csdn.net/Doctor_Wei/article/details/111169930
# tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',kernel_initializer=tf.keras.initializers.he_normal(stddev=0.02))     #   适合深层次
# tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))   #   适合浅层次

import tensorflow as tf
# from cnn_model.base.cnn_base import CNNBase

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

# class SE_Inception_resnet_v2(CNNBase):
#     def __init__(self, **params):
#         super().__init__(params)
#
#     def build_model(self,input_shape,dropout,num_classes):
#         return SEnet(input_shape,dropout,num_classes)

def create_model(num_classes,dropout):
    return SEnet(None,dropout,num_classes)


class CBR(Model):
    def __init__(self,filters,kernel,stride=1,padding='SAME',activation=True,use_bn=True):
        super().__init__()
        _m = [
            nn.Conv2D(filters, kernel, stride, padding=padding, use_bias=not use_bn,data_format=data_format)
        ]
        if use_bn:
            _m.append(nn.BatchNormalization(dims))

        if activation:
            _m.append(nn.ReLU())

        self.m = keras.Sequential(_m)

    def call(self,x):
        return self.m(x)

class Stem(Model):
    def __init__(self):
        super().__init__()
        self.m = [
                keras.Sequential([
                    CBR(32, 3, 2, 'VALID'),
                    CBR(32, 3, 1, 'VALID'),
                    CBR(64, 3)
                ]),
                nn.MaxPool2D(3, 2,data_format=data_format),
                CBR(96, 3, 2, 'valid'),

                keras.Sequential([
                    CBR(64, 1),
                    CBR(96, 3, 1, 'valid'),
                ]),
                keras.Sequential([
                    CBR(64, 1),
                    CBR(64, (7, 1)),
                    CBR(64, (1, 7)),
                    CBR(96, 3, padding='valid'),
                ]),

                CBR(192,3,2,'valid'),
                nn.MaxPool2D(3, 2,data_format=data_format),

                keras.Sequential([
                    nn.BatchNormalization(dims),
                    nn.ReLU()
                ])
             ]
    def call(self,x):
        block_1 = self.m[0](x)
        split_max_x = self.m[1](block_1)
        split_conv_x = self.m[2](block_1)
        x = tf.concat([split_max_x,split_conv_x],dims)

        split_conv_x1 = self.m[3](x)
        split_conv_x2 = self.m[4](x)
        x = tf.concat([split_conv_x1, split_conv_x2], dims)

        split_conv_x = self.m[5](x)
        split_max_x = self.m[6](x)

        x = tf.concat([split_conv_x, split_max_x], dims)

        x = self.m[7](x)

        return x

class InceptionResnetA(Model):
    def __init__(self):
        super().__init__()
        self.m = [
            CBR(32,1),
            CBR(32,1),
            CBR(32,3),
            CBR(32,1),
            CBR(48,3),
            CBR(64,3),
            nn.Concatenate(dims),
            CBR(384,1,use_bn=False,activation=False),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ]

    def call(self,x):
        init = x
        split_conv_x1 = self.m[0](x)

        split_conv_x2 = self.m[1](x)
        split_conv_x2 = self.m[2](split_conv_x2)

        split_conv_x3 = self.m[3](x)
        split_conv_x3 = self.m[4](split_conv_x3)
        split_conv_x3 = self.m[5](split_conv_x3)

        x = self.m[6]([split_conv_x1,split_conv_x2,split_conv_x3])
        x = self.m[7](x)

        x = x*0.1+init

        x = self.m[8](x)
        x = self.m[9](x)

        return x

class InceptionResnetB(Model):
    def __init__(self):
        super().__init__()
        self.m = [
            CBR(192,1),
            CBR(128,1),
            CBR(160,(1,7)),
            CBR(192,(7,1)),
            nn.Concatenate(dims),
            CBR(1152,1,use_bn=False,activation=False),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ]

    def call(self, x):
        init = x
        split_conv_x1 = self.m[0](x)

        split_conv_x2 = self.m[1](x)
        split_conv_x2 = self.m[2](split_conv_x2)
        split_conv_x2 = self.m[3](split_conv_x2)
        x = self.m[4]([split_conv_x1, split_conv_x2])
        x = self.m[5](x)

        x = init + x*0.1
        x = self.m[6](x)
        x = self.m[7](x)

        return x

class InceptionResnetC(Model):
    def __init__(self):
        super().__init__()
        self.m=[
            CBR(192,1),
            CBR(192,1),
            CBR(224,(1,3)),
            CBR(256,(3,1)),
            nn.Concatenate(dims),
            CBR(2144,1,activation=False,use_bn=False),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ]

    def call(self,x):
        init = x
        split_conv_x1 = self.m[0](x)
        split_conv_x2 = self.m[1](x)
        split_conv_x2 = self.m[2](split_conv_x2)
        split_conv_x2 = self.m[3](split_conv_x2)
        x = self.m[4]([split_conv_x1, split_conv_x2])
        x = self.m[5](x)
        x = init + x * 0.1
        x = self.m[6](x)
        x = self.m[7](x)

        return x

class ReductionA(Model):
    def __init__(self):
        super().__init__()
        k = 256
        l = 256
        m = 384
        n = 384

        self.m = [
            nn.MaxPool2D(3,2,data_format=data_format),
            CBR(n,3,2,'valid'),
            CBR(k,1),
            CBR(l,3),
            CBR(m,3,2,'valid'),
            nn.Concatenate(dims),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ]

    def call(self,x):
        split_max_x = self.m[0](x)
        split_conv_x1 = self.m[1](x)
        split_conv_x2 = self.m[2](x)
        split_conv_x2 = self.m[3](split_conv_x2)
        split_conv_x2 = self.m[4](split_conv_x2)

        x = self.m[5]([split_max_x, split_conv_x1, split_conv_x2])

        x = self.m[6](x)
        x = self.m[7](x)

        return x

class ReductionB(Model):
    def __init__(self):
        super().__init__()
        self.m = [
            nn.MaxPool2D(3,2,data_format=data_format),
            CBR(256,1),
            CBR(384,3,2,'valid'),

            CBR(256,1),
            CBR(288,3,2,'valid'),

            CBR(256,1),
            CBR(288,3),
            CBR(320,3,2,'valid'),

            nn.Concatenate(dims),
            nn.BatchNormalization(dims),
            nn.ReLU()
        ]

    def call(self,x):
        split_max_x = self.m[0](x)
        split_conv_x1 = self.m[1](x)
        split_conv_x1 = self.m[2](split_conv_x1)

        split_conv_x2 = self.m[3](x)
        split_conv_x2 = self.m[4](split_conv_x2)

        split_conv_x3 = self.m[5](x)
        split_conv_x3 = self.m[6](split_conv_x3)
        split_conv_x3 = self.m[7](split_conv_x3)

        x = self.m[8]([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])
        x = self.m[9](x)
        x = self.m[10](x)

        return x

class SqueezeExcitationLayer(Model):
    def __init__(self,out_dim, ratio):
        super().__init__()
        self.m = keras.Sequential(
            [nn.GlobalAveragePooling2D(data_format=data_format),
             nn.Flatten(data_format=data_format),
             nn.Dense(out_dim // ratio,"relu"),
             nn.Dense(out_dim,"sigmoid"),
             nn.Reshape((1,1,out_dim)) if tensor_shape=="NHWC" else nn.Reshape((out_dim,1,1))
             ]
        )
    def call(self,x):
        return self.m(x)*x

class PadLayer(Model):
    def __init__(self):
        super().__init__()

    def call(self,x):
        return tf.pad(x, [[0, 0], [32, 32], [32, 32], [0, 0]]) if tensor_shape=="NHWC" else \
            tf.pad(x, [[0, 0], [0, 0],[32, 32], [32, 32]])


class BlockA(Model):
    def __init__(self,nums=5,reduction_ratio=16):
        super().__init__()
        _m = [
            InceptionResnetA(),
            SqueezeExcitationLayer(384,reduction_ratio)
        ]
        _m2 = []
        for _ in range(nums):
            _m2.extend(_m)

        self.m = keras.Sequential(_m2)

        self.m2 = keras.Sequential([
            ReductionA(),
            SqueezeExcitationLayer(1152, reduction_ratio),
        ])


    def call(self,x):
        return self.m2(self.m(x))

class BlockB(Model):
    def __init__(self,nums=10,reduction_ratio=16):
        super().__init__()
        _m = [
            InceptionResnetB(),
            SqueezeExcitationLayer(1152,reduction_ratio)
        ]
        _m2 = []
        for _ in range(nums):
            _m2.extend(_m)

        self.m = keras.Sequential(_m2)
        self.m2 = keras.Sequential([
            ReductionB(),
            SqueezeExcitationLayer(2144,reduction_ratio)
        ])

    def call(self,x):
        return self.m2(self.m(x))

class BlockC(Model):
    def __init__(self,nums=5,reduction_ratio=16):
        super().__init__()
        _m = [
            InceptionResnetC(),
            SqueezeExcitationLayer(2144,reduction_ratio)
        ]
        _m2 = []
        for _ in range(nums):
            _m2.extend(_m)

        self.m = keras.Sequential(_m2)

    def call(self,x):
        return self.m(x)

class SEnet(Model):
    def __init__(self,input_shape=None,dropout=0.5,num_classes=1000):
        super().__init__()
        reduction_ratio = 16

        self.m = [
            PadLayer(),
            Stem(),
            BlockA(5,reduction_ratio),
            BlockB(10,reduction_ratio),
            BlockC(5,reduction_ratio),

            nn.GlobalAveragePooling2D(data_format=data_format),
            nn.Dropout(dropout),
            nn.Flatten(data_format=data_format),
            nn.Dense(num_classes)
        ]

    def call(self,x):
        x = self.m[0](x)
        x = self.m[1](x)
        x = self.m[2](x)
        x = self.m[3](x)
        x = self.m[4](x)
        x = self.m[5](x)
        x = self.m[6](x)
        x = self.m[7](x)
        x = self.m[8](x)

        return x


if __name__ == "__main__":
    import numpy as np

    input_shape = (32, 32, 3)
    model = SEnet()

    model(nn.Input(input_shape))
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    x = tf.convert_to_tensor(np.random.random([1, *input_shape]))
    pred = model.predict(x)
    print(pred.shape)