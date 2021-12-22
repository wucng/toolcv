# @Time: 2021/4/1 13:50 
# @Author: wucong
# @File: SE_ResNeXt.py
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
    data_format = 'channels_first' # Conv2D;MaxPool2D;GlobalAveragePooling2D;AveragePooling2D;Flatten
    dims = 1 # tf.concat;Concatenate;nn.Reshape;tf.pad;BatchNormalization
else:
    data_format = 'channels_last'
    dims = 3

# class SE_ResNeXt(CNNBase):
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
        self.m = CBR(32,3,2)

    def call(self,x):
        return self.m(x)

class TransformLayer(Model):
    def __init__(self,stride):
        super().__init__()
        self.m = keras.Sequential(
            [
                CBR(64,1,1),
                CBR(64,3,stride)
             ]
        )
    def call(self,x):
        return self.m(x)

class TransitionLayer(Model):
    def __init__(self,out_dim):
        super().__init__()
        self.m = CBR(out_dim,1,1,activation=False)

    def call(self,x):
        return self.m(x)

class SplitLayer0(Model):
    def __init__(self,stride):
        super().__init__()
        cardinality = 8
        # self.m = keras.Sequential([TransformLayer(stride) for _ in range(cardinality)])
        self.m = [TransformLayer(stride) for _ in range(cardinality)]
        self.cardinality = cardinality

    def call(self,x):
        return tf.concat([self.m[i](x) for i in range(self.cardinality)],dims)

class SplitLayer(Model):
    def __init__(self,stride):
        super().__init__()
        cardinality = 8
        # self.m = keras.Sequential([TransformLayer(stride) for _ in range(cardinality)])
        self.m = [TransformLayer(stride) for _ in range(cardinality)]

        self.cardinality = cardinality

    def call(self,x):
        x_list = tf.split(x,self.cardinality,dims)
        return tf.concat([self.m[i](x_list[i]) for i in range(self.cardinality)],dims)



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
    def __init__(self,channel):
        super().__init__()
        self.channel = channel

    def call(self,x):
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [self.channel, self.channel]]) \
                if tensor_shape=="NHWC" else \
            tf.pad(x, [[0, 0],[self.channel, self.channel], [0, 0], [0, 0]])


class BasicBlock(Model):
    def __init__(self,out_dim,stride):
        super().__init__()
        reduction_ratio = 16
        self.m = keras.Sequential(
            [SplitLayer(stride),
             TransitionLayer(out_dim),
             SqueezeExcitationLayer(out_dim,reduction_ratio)
             ]
        )

        if stride==2:
            channel = out_dim // 4
            self.m2 = keras.Sequential(
                [
                    nn.AvgPool2D(2,2,data_format=data_format),
                    PadLayer(channel)
                ]
            )
        self.stride = stride

        self.m3 = nn.ReLU()

    def call(self,x):
        # return self.m3(self.m(x)+x if self.stride==1 else self.m2(x))
        if self.stride==1:
            return self.m3(self.m(x)+x)
        else:
            return self.m3(self.m(x)+self.m2(x))

class ResidualLayer(Model):
    def __init__(self,out_dim,stride,res_block=3):
        super().__init__()
        self.res_block = res_block

        self.m = keras.Sequential([BasicBlock(out_dim,stride if i==0 else 1) for i in range(res_block)])

    def call(self,x):
        return self.m(x)

class SEnet(Model):
    def __init__(self,input_shape=None,dropout=0.5,num_classes=1000):
        super().__init__()
        self.m = keras.Sequential(
            [
                Stem(),
                ResidualLayer(64,2,3),
                ResidualLayer(128,2,5),
                ResidualLayer(256,2,5),
                ResidualLayer(512,2,3),

                nn.GlobalAveragePooling2D(data_format=data_format),
                nn.Dropout(dropout),
                nn.Flatten(data_format=data_format),
                nn.Dense(num_classes)
             ]
        )
    def call(self,x):
        return self.m(x)

if __name__ == "__main__":
    import numpy as np
    input_shape = (32,32,3)
    model = SEnet()
    model(nn.Input(input_shape))
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    x = tf.convert_to_tensor(np.random.random([1,*input_shape]))
    pred = model.predict(x)
    print(pred.shape)
