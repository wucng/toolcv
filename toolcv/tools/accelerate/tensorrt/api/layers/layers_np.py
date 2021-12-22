"""
只使用numpy实现
输入：[bs,c,h,w]格式，权重格式：[32,3,5,5]
"""
import numpy as np
from scipy.signal import convolve,convolve2d,fftconvolve

def np_bn(x,name,weights,belta=1e-5):
    g0 = weights[name + '.weight']  # .reshape(-1)
    b0 = weights[name + '.bias']  # .reshape(-1)
    m0 = weights[name + '.running_mean']  # .reshape(-1)
    v0 = weights[name + '.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    # power0 = np.ones(len(g0), dtype=dtype)
    bn1 = x*scale0[None,:,None,None]+shift0[None,:,None,None]

    return bn1


def np_convV2(x,name,weights,strides=(1,1),padding=(0,0)):
    x = x.transpose(0,2,3,1)
    conv1_w = weights[name+'.weight'].transpose(0,2,3,1)
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=conv1_w.dtype)
    """
    x:[8,3,224,224] => [8,224,224,3]
    conv1_w:[32,3,5,5] => [32,5,5,3]
    conv1_b:[32]
    """
    kernel = conv1_w
    bs,h,w,c = x.shape
    kout,kh, kw, kc = kernel.shape
    # padding
    if sum(padding)==0:
        pass
    else:
        # mode = "same"
        tmp = np.zeros([bs,h+2*padding[0],w+2*padding[1],c],np.float32)
        tmp[:,padding[0]:padding[0]+h,padding[1]:padding[1]+w,:]=x
        x =tmp

    bs, h, w, c = x.shape
    kernel = np.reshape(kernel, [kout, -1]).T

    # img2col
    h = int((h-kh)/strides[0]+1)
    w = int((w-kw)/strides[1]+1)
    res = np.zeros([bs,h, w, kh, kw, kc], np.float32)
    for i in range(0, h, strides[0]):
        for j in range(0, w, strides[1]):
            res[:,i, j] = x[:,i:i + kh, j:j + kw]
    res = np.reshape(res, [bs,h, w, -1])

    return (np.dot(res, kernel)+conv1_b[None,None,None,:]).transpose(0,3,1,2)

def np_conv(x,name,weights,strides=(1,1),padding=(0,0)):
    conv1_w = weights[name+'.weight']
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)
    """
    x:[8,3,224,224]
    conv1_w:[32,3,5,5]
    conv1_b:[32]
    """
    kernel = conv1_w
    bs,c,h,w = x.shape
    kout, kc,kh, kw = kernel.shape
    # padding
    if sum(padding)==0:
        pass
    else:
        # mode = "same"
        tmp = np.zeros([bs,c,h+2*padding[0],w+2*padding[1]],np.float32)
        tmp[:,:,padding[0]:padding[0]+h,padding[1]:padding[1]+w]=x
        x =tmp

    bs,c,h,w = x.shape
    kernel = np.reshape(kernel, [kout, -1]).T

    # img2col
    h = int((h-kh)/strides[0]+1)
    w = int((w-kw)/strides[1]+1)
    res = np.zeros([bs,h, w, kc, kh, kw], np.float32)
    for i in range(0, h, strides[0]):
        for j in range(0, w, strides[1]):
            res[:,i, j] = x[:,:,i:i + kh, j:j + kw]
    res = np.reshape(res, [bs,h, w, -1])

    return (np.dot(res, kernel)+conv1_b[None,None,None,:]).transpose(0,3,1,2)

def softmax(x:np.array):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

def relu(x:np.array):
    s = np.where(x < 0, 0, x)
    return s

def np_fc(x:np.array,name,weights):
    fc_w = weights[name + '.weight']
    fc_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([fc_w.shape[0]], dtype=fc_w.dtype)
    x = np.matmul(x, fc_w.T) + fc_b

    return x

def np_maxpool2x2(x:np.array,ks=(2,2),strides=(2,2)):
    kh,kw = ks
    bs, c, h, w = x.shape
    res = np.zeros([bs, c, h//2, w//2], np.float32)
    for i in range(0, h, strides[0]):
        for j in range(0, w, strides[1]):
            if i//2>=h//2 or j//2>=w//2:continue
            res[:,:,i//2,j//2] = x[:,:,i:i+kh,j:j+kw].max((2,3))

    return res

def np_network(x,weights):
    x = np_conv(x,'conv1',weights)
    x = np_bn(x,'bn1',weights)
    x = relu(x)
    x = np_maxpool2x2(x) # [8,32,30,30]

    x = np_conv(x, 'conv2', weights)
    x = np_bn(x, 'bn2', weights)
    x = relu(x)
    x = np_maxpool2x2(x) # [8,64,13,13]

    x = np_conv(x, 'conv3', weights)
    x = np_bn(x, 'bn3', weights)
    x = relu(x)
    x = np_maxpool2x2(x) # [8,128,4,4]

    x = np_conv(x, 'conv4', weights,padding=(1,1))
    x = np_bn(x, 'bn4', weights)
    x = relu(x)
    x = np_maxpool2x2(x)# [8,256,2,2]

    x = np.reshape(x,(-1,1024))

    x = np_fc(x,'fc1',weights)
    x = relu(x)

    x = np_fc(x, 'fc2', weights)
    x = softmax(x)

    return x

if __name__=="__main__":
    weights = np.load("./model.npz")
    x = np.random.rand(8,3,64,64).astype(np.float32)

    x=np_network(x,weights)
    print(x.shape)
