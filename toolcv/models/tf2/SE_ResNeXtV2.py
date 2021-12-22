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
    def __init__(self,filters,kernel,stride=1,padding='SAME',activation=True,use_bn=True,groups=1):
        super().__init__()
        _m = [
            nn.Conv2D(filters, kernel, stride, padding=padding, use_bias=not use_bn,data_format=data_format,groups=groups)
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


class BottleBlock(Model):
    def __init__(self,out_dim,stride):
        super().__init__()
        expand = 4
        in_c = out_dim
        hide_c = in_c
        out_c = out_dim*expand

        reduction_ratio = 16

        self.m = keras.Sequential([
            CBR(in_c,1,1),
            CBR(hide_c,3,stride,groups=32),
            CBR(out_c,1,activation=False)
        ])
        self.downsample = CBR(out_c,1,stride,activation=False) if stride>1 else keras.Sequential([])
        self.seblock = SqueezeExcitationLayer(out_c,reduction_ratio)
        self.act = nn.ReLU()

    def call(self,x):
        return self.act(self.seblock(self.m(x))+self.downsample(x))


class ResidualLayer(Model):
    def __init__(self,out_dim,stride,res_block=3):
        super().__init__()
        self.res_block = res_block

        self.m = keras.Sequential([BottleBlock(out_dim,stride if i==0 else 1) for i in range(res_block)])

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
