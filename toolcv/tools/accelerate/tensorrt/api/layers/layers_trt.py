"""
https://www.jianshu.com/p/2d949d05430d
https://github.com/NVIDIA-AI-IOT/torch2trt/tree/master/torch2trt/converters
https://github.com/NVIDIA/TensorRT/tree/master/plugin
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html

TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0
pytorch '1.5.0'
nvidia Driver Version: 440.82
"""
import numpy as np
import tensorrt as trt

class ModelData(object):
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"  #
    BATCH_SIZE = 128 # 要与 pytorch2onnx 时设置的batch size 对应
    INPUT_SHAPE = (128,3, 64, 64)
    OUTPUT_SIZE = 10
    DTYPE = trt.float16  # 使用半精度 half-float
    NP_DTYPE = np.float16
    # DTYPE = trt.float32
    # NP_DTYPE = np.float32
    MEM_SIZE = 1 # 1G内存
    PLUGIN_LIBRARY = "build/libfcplugin.so" # 命名规则是"lib+name+plugin.so"，只有name能修改

def trt_scale(network,input_size,shift=0,scale=1,power=1,mode=trt.ScaleMode.CHANNEL):
    """
        :mode
            trt.ScaleMode.CHANNEL
            CHANNEL：每个通道的系数。 假定通道尺寸是倒数第三个尺寸。
                INPUT_SHAPE = (1,2,3, 3)
                shift=np.asarray([1,1],np.float32) # shape [2,]
                scale=np.asarray([1,2],np.float32) # shape [2,]
                power=np.asarray([1,1],np.float32) # shape [2,]

            UNIFORM：张量所有元素的系数相同。
                INPUT_SHAPE = (1,2,3, 3)
                INPUT_SHAPE = (5,2,3, 3)

                shift=np.ones([1],np.float32) # 只能是一个标量值
                scale=np.ones([1],np.float32) # 只能是一个标量值
                power=np.ones([1],np.float32) # 只能是一个标量值

            ELEMENTWISE：逐元素系数。
                INPUT_SHAPE = (1,2,3, 3)
                shift=np.ones([1,2,3,3],np.float32)
                scale=np.ones([1,2,3,3],np.float32)
                power=np.ones([1,2,3,3],np.float32)
    """
    """output=(input∗scale+shift)^power"""
    return network.add_scale(input=input_size, mode=mode, shift=shift,
                            scale=scale,power=power)

def trt_bn(network,input_size,name,weights,dtype,belta=1e-5):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    power0 = np.ones(len(g0), dtype=dtype)
    # bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,power=power0)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0)
    return bn1

def trt_bn1d(network,input_size,name,weights,new_shape,belta=1e-5):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0

    # reshape to 2D
    shuffle = network.add_shuffle(input_size)
    shuffle.reshape_dims = (new_shape, new_shape, 1)

    # power0 = np.ones(len(g0), dtype=dtype)
    # bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,power=power0)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0)

    # reshape to 1D
    shuffle = network.add_shuffle(bn1.get_output(0))
    shuffle.reshape_dims = (new_shape, new_shape, 1)

    return shuffle

def trt_conv(network,input_size,name,weights,dtype,kernel_shape,stride,padding,padding_mode=None):
    """padding_mode = trt.PaddingMode.SAME_LOWER, EXPLICIT_ROUND_DOWN"""
    conv1_w = weights[name+'.weight']
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)
    conv1 = network.add_convolution(input=input_size, num_output_maps=conv1_w.shape[0],
                                    kernel_shape=kernel_shape, kernel=conv1_w, bias=conv1_b)
    conv1.stride = stride
    if padding_mode:
        conv1.padding_mode = padding_mode
    else:
        conv1.padding = padding

    return conv1

def trt_deconv(network,input_size,name,weights,dtype,kernel_shape,stride,padding_mode=trt.PaddingMode.SAME_LOWER):
    """
    x = np.random.random_sample([3,2,3,3]).astype(np.float32)

     F.conv_transpose2d(torch.from_numpy(x),torch.ones([2,5,3,3],requires_grad=True),
                                            torch.ones([5],requires_grad=True),2,1,1)

    layer = network.add_deconvolution(input=trt_tensor,num_output_maps = 5,kernel_shape = (3,3),
                                      kernel = np.ones([2,5,3,3],np.float32),
                                      bias = np.ones([5],np.float32))
    layer.stride = (2,2)
    # layer.padding = (1,1)
    layer.padding_mode = trt.PaddingMode.SAME_LOWER
    # layer.dilation = (1,1)

    :return shape [3,5,6,6]
    """
    conv1_w = weights[name + '.weight']
    conv1_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([conv1_w.shape[1]], dtype=dtype)

    layer = network.add_deconvolution(input=input_size, num_output_maps=conv1_w.shape[1], kernel_shape=kernel_shape,
                                      kernel=conv1_w,bias=conv1_b)
    layer.stride = stride
    layer.padding_mode = padding_mode

    return layer

def trt_pool(network,input_size,type,pool_size,stride,padding=(0,0)):
    """
    :param network:
    :param input_size:
    :param type: trt.PoolingType.MAX , trt.PoolingType.AVERAGE
    :param pool_size:
    :param stride:
    :param padding:
    :return:
    """
    pool1 = network.add_pooling(input=input_size, type=type, window_size=pool_size)
    pool1.stride = stride
    pool1.padding = padding
    return pool1

def trt_active(network,input_size,type=trt.ActivationType.RELU,alpha=None,beta=None):
    """
    :type:
        trt.ActivationType.SOFTPLUS

        SOFTPLUS : Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)

        RELU : Rectified Linear activation

        LEAKY_RELU : Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0

        SOFTSIGN : Softsign activation: f(x) = x / (1 + abs(x))

        THRESHOLDED_RELU : Thresholded Relu activation: f(x) = x if x > alpha, f(x) = 0 if x <= alpha

        HARD_SIGMOID : Hard sigmoid activation: f(x) = max(0, min(1, alpha * x + beta))

        CLIP : Clip activation: f(x) = max(alpha, min(beta, x))

        SIGMOID : Sigmoid activation

        SELU : Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0

        TANH : Hyperbolic Tangent activation

        ELU : Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0

        SCALED_TANH : Scaled Tanh activation: f(x) = alpha * tanh(beta * x)
    """
    # if type == trt.ActivationType.LEAKY_RELU:
    layer = network.add_activation(input=input_size, type=type)
    if alpha:layer.alpha = alpha
    if beta:layer.beta = beta

    return layer

def trt_clamp(network,input_size,min_value=0.0,max_value=6.0):
    # 裁剪到某个范围
    layer = network.add_activation(input=input_size, type=trt.ActivationType.CLIP)
    layer.alpha = min_value
    layer.beta = max_value
    return layer

def trt_add(network,input_size1,input_size2):
    return network.add_elementwise(input1=input_size1,
                            input2=input_size2,
                            op=trt.ElementWiseOperation.SUM)

def trt_elementWiseLayer(network,input_size1,input_size2,op):
    """
    # 两层做逐元素操作
    trt.ElementWiseOperation.SUM

    DIV : Divide the first element by the second

    SUB : Subtract the second element from the first

    GREATER : Check if element in first tensor is greater than corresponding element in second tensor

    POW : The first element to the power of the second element

    OR : Logical OR of two elements

    EQUAL : Check if two elements are equal

    XOR : Logical XOR of two elements

    MAX : Max of the two elements

    MIN : Min of the two elements

    SUM : Sum of the two elements

    PROD : Product of the two elements

    LESS : Check if element in first tensor is less than corresponding element in second tensor

    AND : Logical AND of two elements

    FLOOR_DIV : Floor division of the first element by the second
    """
    return network.add_elementwise(input1=input_size1,
                                   input2=input_size2,
                                   op=op)


def trt_concat(network,inputs=[]):
    # 只能按第2个维度合并
    return network.add_concatenation(inputs)

def trt_upsample(network,input_size,scale_factor = 2,align_corners = False,mode="nearest"):
    """
    x = np.random.random_sample([1,3,3,3]).astype(np.float32)
    torch_x = F.interpolate(torch.from_numpy(x),scale_factor=(2,2),mode="nearest").numpy()
    """
    upsample = network.add_resize(input_size)
    if mode=="nearest":
        upsample.resize_mode = trt.ResizeMode.NEAREST
    elif mode=="linear":
        upsample.resize_mode = trt.ResizeMode.LINEAR
    else:
        raise "%s must in ['nearest','linear']"%mode
    upsample.align_corners = align_corners
    upsample.scales = [1, 1, scale_factor, scale_factor]
    upsample.shape = (int(input_size.shape[0]), int(input_size.shape[1]),
                      int(input_size.shape[2] * scale_factor), int(input_size.shape[3] * scale_factor))

    return upsample

def trt_fc(network,input_size,name,weights,dtype):
    fc1_w = weights[name + '.weight']
    fc1_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)

    fc2 = network.add_fully_connected(input=input_size, num_outputs=fc1_w.shape[0], kernel=fc1_w, bias=fc1_b)
    return fc2


def trt_padding(network,input_size,pre_padding=(1,1),post_padding=(1,1)):
    """
       input_size 维数必须是4维
       默认填充值 0
       pre_padding: HW 指定 左边与上边填充的数量
       post_padding: HW 指定 右边与下边填充的数量
    """
    return network.add_padding(input_size,pre_padding=pre_padding,post_padding=post_padding)

def trt_softmax(network,input_size):
    """
        input_size: shape [32,10,1,1] 按 dim=1 求
        input_size: shape [32,10,1] 按行 dim = 0求
        input_size: shape [32,10] 按行 dim = 0 求
    """
    return network.add_softmax(input_size)

def trt_transpose(network,input_size,aixs=(0,2,3,1)):
    """
    x = np.random.random_sample([5,2,3,3]).astype(np.float32)
    torch_x = torch.from_numpy(x).permute(0,2,3,1)
    """
    layer = network.add_shuffle(input=input_size)
    layer.first_transpose = trt.Permutation(aixs)

    return layer

def trt_squeeze(network,input_size,reshape_dims=(5,2,3,3,1)):
    """
    # unsqueeze
    x = np.random.random_sample([5,2,3,3]).astype(np.float32)
    torch_x = torch.from_numpy(x).unsqueeze(-1)

    # squeeze
    x = np.random.random_sample([5,2,3,3,1]).astype(np.float32)
    torch_x = torch.from_numpy(x).squeeze(-1)
    """
    layer = network.add_shuffle(input=input_size)
    # layer.reshape_dims = [input_size.shape[0], input_size.shape[1], input_size.shape[2], input_size.shape[3], 1]
    layer.reshape_dims = reshape_dims

    return layer

def trt_view(network,input_size,reshape_dims=(5,2,3,3,1)):
    """
    x = np.random.random_sample((5,8,8,3)).astype(np.float32)
    np_x = torch.from_numpy(x).contiguous().view(5,4,2,4,2,3).numpy()

    layer = network.add_shuffle(input=input_size)
    layer.reshape_dims = (5,4,2,4,2,3)
    """
    layer = network.add_shuffle(input=input_size)
    layer.reshape_dims = reshape_dims

    return layer


def trt_math(network,input_size,op):
    """# 数学运算
    trt.UnaryOperation.EXP

    FLOOR : Floor

    ACOS : Inverse cosine

    ACOSH : Inverse hyperbolic cosine

    NEG : Negation

    COS : Cosine

    ASINH : Inverse hyperbolic sine

    SQRT : Square root

    RECIP : Reciprocal

    SINH : Hyperbolic sine

    COSH : Hyperbolic cosine

    ASIN : Inverse sine

    LOG : Log (base e)

    ABS : Absolute value

    EXP : Exponentiation

    ATANH : Inverse hyperbolic tangent

    CEIL : Ceiling

    TAN : Tangent

    SIN : Sine

    ATAN : Inverse tangent

    ERF : Gauss error function

    NOT : Not
    """
    layer = network.add_unary(input=input_size, op=op)
    return layer

def trt_reduce(network,input_size,op,axes=2,keep_dims=False):
    """
    torch.from_numpy(x).mean(dim=1,keepdim=True) # dim 从索引 0 开始计算
    :param network:
    :param input_size:
    :param op:
            # trt.ReduceOperation.AVG
            PROD : 连乘
            MAX :
            AVG :
            MIN :
            SUM :
    :param axes: 从索引 1 开始计算
    :param keep_dims:
    :return:
    """
    layer = network.add_reduce(input=input_size, op=op, axes=axes, keep_dims=keep_dims)
    return layer

def trt_constant(network,shape,weights):
    """
    layer = network.add_constant(
        shape=(2, 3, 3),
        weights=np.ones((2, 3, 3), np.float32)
    )
    :return  返回一个常量 tensor
    """
    layer = network.add_constant(
        shape=shape,
        weights=weights)

    return layer

def trt_fill(network,shape,op):
    """
    :param network:
    :param shape:
    :param op:  trt.FillOperation.RANDOM_UNIFORM,LINSPACE
    :return:
    """
    layer = network.add_fill(shape=shape, op=op)
    return layer

def trt_getshape(network,input_size):
    """等价于: input_size.shape"""
    layer = network.add_shape(input_size)
    return layer

def trt_slice(network,input_size,start,shape,stride):
    """
    x = np.random.random_sample([5,2,3,3]).astype(np.float32)
    x[:2,...]
    network.add_slice(input_size, start=(0,0,0,0), shape=(2,2,3,3), stride=(1,1,1,1))
    """
    layer = network.add_slice(input_size, start=start, shape=shape, stride=stride)

    return layer


def trt_topk(network,input_size,op,k,axes):
    """
    x = np.random.random_sample([5,5]).astype(np.float32)
    torch.topk(torch.from_numpy(x),3,1)[0]
    layer = network.add_topk(input_size,trt.TopKOperation.MAX,3,axes=2)
    ------------------------------------
    x = np.random.random_sample([5,5,3,3]).astype(np.float32)
    torch_x=torch.topk(torch.from_numpy(x),3,1)[0]
    layer = network.add_topk(input_size,trt.TopKOperation.MAX,3,axes=2)


    :axes 从1 开始计算
    :op trt.TopKOperation.MAX,MIN
    :return:
    """
    layer = network.add_topk(input_size, op, k, axes=axes)

    return layer

def trt_matmul(network,input_size0,op0,input_size1,op1):
    """
    op0:
    op1:
    TRANSPOSE : Transpose each matrix

    VECTOR : Treat operand as collection of vectors

    NONE :
    ------------------------------------------------------
    x = np.random.random_sample([5,3]).astype(np.float32)
    np_x=np.matmul(x,x.T)
    layer = network.add_matrix_multiply(input_size,trt.MatrixOperation.NONE,input_size,trt.MatrixOperation.TRANSPOSE)
    """
    layer = network.add_matrix_multiply(input_size0, op0, input_size1, op1)

    return layer

def trt_identity(network,input_size):
    return network.add_identity(input_size)


def trt_example(network,input_size):
    """
    INPUT_SHAPE = (5,8,8,85)
    OUTPUT_SIZE = (5,8,8,85)

    x = np.random.random_sample((5,8,8,85)).astype(np.float32)
    torch_x = torch.from_numpy(x)
    torch_x[...,5:] = torch.sigmoid(torch_x[...,5:])
    ----------------------------------------------------------------------
    layer1 = trt_slice(network,input_size,(0,0,0,0),(5,8,8,5),(1,1,1,1)) # [5,8,8,5]
    layer2 = trt_slice(network,input_size,(0,0,0,5),(5,8,8,80),(1,1,1,1)) # [5,8,8,80]
    layer2 = trt_active(network,layer2.get_output(0),trt.ActivationType.SIGMOID)
    layer1 = trt_transpose(network,layer1.get_output(0),(0,3,1,2)) # [5,5,8,8]
    layer2 = trt_transpose(network,layer2.get_output(0),(0,3,1,2)) # [5,80,8,8]
    # layer2 = trt_softmax(network,layer2.get_output(0))

    # concat
    layer = trt_concat(network,[layer1.get_output(0),layer2.get_output(0)]) # 只能按第2个维度合并 [5,85,8,8]
    layer = trt_transpose(network,layer.get_output(0),(0,2,3,1)) # [5,8,8,85]
    :return:
    """
    pass




# ------------add_plugin_v2---------------------------------------------------
"""
add_plugin_v2(self: tensorrt.tensorrt.INetworkDefinition, 
            inputs: List[tensorrt.tensorrt.ITensor], 
            plugin: tensorrt.tensorrt.IPluginV2) → tensorrt.tensorrt.IPluginV2Layer
功能：注册插件
Parameters :  input1 - 输入tensor列表，
              plugin - 插件函数

Returns:  一个新的layer或None

除了已经编好的层之外，还有一些特别的插件可以自定义一些操作，官方有写好的插件，也可以自己定义自己的插件。目前主要介绍一些官方的插件：
"""

"""
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
TRT_LOGGER = trt.Logger()
#加载插件库
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
#获得所有支持的插件
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name)

输出:
RnRes2Br1Br2c_TRT
RnRes2Br1Br2c_TRT
CustomSkipLayerNormPluginDynamic
GroupNormalizationPlugin
CustomEmbLayerNormPluginDynamic
CustomGeluPluginDynamic
CgPersistentLSTMPlugin_TRT
CustomQKVToContextPluginDynamic
CustomFCPluginDynamic
SingleStepLSTMPlugin
RnRes2Br2bBr2c_TRT
RnRes2Br2bBr2c_TRT
GridAnchor_TRT
NMS_TRT
Reorg_TRT
Region_TRT
Clip_TRT
LReLU_TRT
PriorBox_TRT
Normalize_TRT
RPROI_TRT
BatchedNMS_TRT
FlattenConcat_TRT
CropAndResize
DetectionLayer_TRT
Proposal
ProposalLayer_TRT
PyramidROIAlign_TRT
ResizeNearest_TRT
Split
SpecialSlice_TRT
InstanceNormalization_TRT
"""
# ---------------------------------------------------------------
def trt_clip(network,input_size,min_value=0.0,max_value=6.0,init=False):
    # 有问题，后面解析会报错, 推荐使用 trt_clamp
    if init:
        #使用插件，必须加载插件库
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        #获得所有支持的插件
        PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
    ###"Clip_TRT"
    def get_trt_plugin(plugin_name):
            plugin = None
            for plugin_creator in PLUGIN_CREATORS:
                if plugin_creator.name == plugin_name:
                    # 收集参数，各个参数的意义参考https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h.html#af308dcae61dab659073bc91c6ba63a7e
                    Clip_slope_field = trt.PluginField("clipMin", np.array([min_value], dtype=np.float32), \
                                                        trt.PluginFieldType.FLOAT32)
                    Clip_slope_field2 = trt.PluginField("clipMax", np.array([max_value], dtype=np.float32),\
                                                        trt.PluginFieldType.FLOAT32)
                    field_collection = trt.PluginFieldCollection([Clip_slope_field,Clip_slope_field2])
                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
            return plugin

    return network.add(inputs=[input_size], plugin=get_trt_plugin("Clip_TRT"))

def trt_lrelu(network,input_size,neg_slope=0.1,init=False):
    # 有问题，后面解析会报错, 推荐使用 trt_active
    if init:
        #使用插件，必须加载插件库
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        #获得所有支持的插件
        PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
    ###"LReLU_TRT"
    def get_trt_plugin(plugin_name):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                lrelu_slope_field = trt.PluginField("neg_slope", np.array([neg_slope], dtype=np.float32),
                                                    trt.PluginFieldType.FLOAT32)
                field_collection = trt.PluginFieldCollection([lrelu_slope_field])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

    return network.add(inputs=[input_size], plugin=get_trt_plugin("LReLU_TRT"))

# ------------add_plugin_v2---------------------------------------------------

def populate_network(network, weights):
    dtype = ModelData.NP_DTYPE
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1 = trt_conv(network,input_tensor,"conv1",weights,dtype,(5,5),(1,1),(0,0))
    bn1 = trt_bn(network,conv1.get_output(0),"bn1",weights,dtype)
    relu1 = trt_active(network,bn1.get_output(0))
    maxpool1 = trt_pool(network,relu1.get_output(0),trt.PoolingType.MAX,(2,2),(2,2))

    conv2 = trt_conv(network, maxpool1.get_output(0), "conv2", weights, dtype, (5, 5), (1, 1), (0, 0))
    bn2 = trt_bn(network, conv2.get_output(0), "bn2", weights, dtype)
    relu2 = trt_active(network, bn2.get_output(0))
    maxpool2 = trt_pool(network, relu2.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    conv3 = trt_conv(network, maxpool2.get_output(0), "conv3", weights, dtype, (5, 5), (1, 1), (0, 0))
    bn3 = trt_bn(network, conv3.get_output(0), "bn3", weights, dtype)
    relu3 = trt_active(network, bn3.get_output(0))
    maxpool3 = trt_pool(network, relu3.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    conv4 = trt_conv(network, maxpool3.get_output(0), "conv4", weights, dtype, (3, 3), (1, 1), (1, 1))
    bn4 = trt_bn(network, conv4.get_output(0), "bn4", weights, dtype)
    relu4 = trt_active(network, bn4.get_output(0))
    maxpool4 = trt_pool(network, relu4.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    fc1 = trt_fc(network,maxpool4.get_output(0),"fc1",weights,dtype)
    relufc1 = trt_active(network, fc1.get_output(0))

    fc2 = trt_fc(network, relufc1.get_output(0), "fc2", weights, dtype)

    """
    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))
    """
    # or add softmax layer
    softmax = network.add_softmax(fc2.get_output(0))
    softmax.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax.get_output(0))
    # """