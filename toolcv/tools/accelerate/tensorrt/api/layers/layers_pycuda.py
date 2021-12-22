"""
只使用numpy+pycuda实现（所以numpy格式必须都转成float32,float64会报错）
输入：[bs,h,w,c]格式，权重格式：[32,5,5,3]
"""
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Stream

import numpy as np

def cuda_bn(g_x,name,weights,belta=1e-5):
    """x:[bs,h,w,c]"""
    g0 = weights[name + '.weight']  # .reshape(-1)
    b0 = weights[name + '.bias']  # .reshape(-1)
    m0 = weights[name + '.running_mean']  # .reshape(-1)
    v0 = weights[name + '.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0

    # to gpu
    g_scale = cuda.to_device(scale0)
    g_shift = cuda.to_device(shift0)

    mod = SourceModule(
        """
        __global__ void gpu_bn(float *a_inOut,float *scale,float *shift)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            
            int idx = bz*gridDim.y*gridDim.x+by*gridDim.x+bx;
            
            a_inOut[idx]=a_inOut[idx]*scale[bx]+shift[bx];
        }
        """
    )

    x_shape = g_x.array.shape

    block = (1, 1, 1)
    grid = (x_shape[-1],x_shape[2], x_shape[1]*x_shape[0])
    func = mod.get_function("gpu_bn")
    func(g_x, g_scale, g_shift, grid=grid, block=block, shared=0,stream=Stream(0))

    # free
    g_scale.free()
    g_shift.free()

    return g_x

def np_bn(x,name,weights,belta=1e-5):
    g0 = weights[name + '.weight']  # .reshape(-1)
    b0 = weights[name + '.bias']  # .reshape(-1)
    m0 = weights[name + '.running_mean']  # .reshape(-1)
    v0 = weights[name + '.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    # power0 = np.ones(len(g0), dtype=dtype)
    bn1 = x*scale0[None,None,None,:]+shift0[None,None,None,:]

    return bn1

def img2col(g_x,kernel,strides=(1,1),padding=(0,0)):
    bs, h, w, c = g_x.array.shape
    kh, kw, kc, kout = kernel.shape

    # padding
    if sum(padding) == 0:
        pass
    else:
        # mode = "same"
        tmp = np.zeros([bs, h + 2 * padding[0], w + 2 * padding[1], c], np.float32)
        tmp[:, padding[0]:padding[0] + h, padding[1]:padding[1] + w, :] = g_x.array
        del g_x
        g_x = cuda.InOut(tmp)

    bs, h, w, c = g_x.array.shape
    img_h = h
    img_w = w
    # img2col
    h = int((h - kh) / strides[0] + 1)
    w = int((w - kw) / strides[1] + 1)
    mod = SourceModule(
        """
        __global__ void img2col(float *g_x,float *res,int *g_strides)
        {
            int bx = blockIdx.x; // w -> j
            int by = blockIdx.y; // h -> i
            int bz = blockIdx.z; // bs -> b
            int tx = threadIdx.x; // kc -> kcc
            int ty = threadIdx.y; // kw -> kj
            int tz = threadIdx.z; // kh -> ki
            int tid = tx + ty * blockDim.x+tz*blockDim.x*blockDim.y;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            int idx = bid*blockDim.x*blockDim.y*blockDim.z+tid;

            int img_h = g_strides[0];
            int img_w = g_strides[1];
            int img_c = g_strides[2];

            int bid_gx= bz*img_h*img_w+(by+tz)*img_w+(bx+ty);
            int idx_gx = bid_gx*img_c+tx;
            res[idx] = g_x[idx_gx];    
        }
        
         __global__ void img2col2(float *g_x,float *res,int *g_strides)
        {
            int bx = blockIdx.x; // w -> j
            int by = blockIdx.y; // h -> i
            int bz = blockIdx.z; // bs -> b
            // int tx = threadIdx.x; // kc -> kcc
            int ty = threadIdx.y; // kw -> kj
            int tz = threadIdx.z; // kh -> ki
            // int tid = tx + ty * blockDim.x+tz*blockDim.x*blockDim.y;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            // int idx = bid*blockDim.x*blockDim.y*blockDim.z+tid;
            
            int img_h = g_strides[0];
            int img_w = g_strides[1];
            int img_c = g_strides[2];
            
            int bid_gx= bz*img_h*img_w+(by+tz)*img_w+(bx+ty);
            
            int idx =0;
            int idx_gx =0;
            for(int tx=0;tx<img_c;++tx)
            {
                idx = bid*img_c*blockDim.y*blockDim.z+tx + ty * img_c+tz*img_c*blockDim.y;
                idx_gx = bid_gx*img_c+tx;
                res[idx] = g_x[idx_gx];   
            }
                     
        }
        
         __global__ void img2col3(float *g_x,float *res,int *g_strides)
        {
            int bx = blockIdx.x; // w -> j
            int by = blockIdx.y; // h -> i
            int bz = blockIdx.z; // bs -> b
            int tx = threadIdx.x; // kc -> kcc
            int ty = threadIdx.y; // kw -> kj
            int tz = threadIdx.z; // kh -> ki
            // int tid = tx + ty * blockDim.x+tz*blockDim.x*blockDim.y;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            // int idx = bid*blockDim.x*blockDim.y*blockDim.z+tid;
            
            int img_h = g_strides[0];
            int img_w = g_strides[1];
            int img_c = g_strides[2];
            
            int bid_gx= bz*img_h*img_w+(by+tz)*img_w+(bx+ty);
            
            int idx =0;
            int idx_gx =0;
            
            while(tx<img_c)
            {
                idx = bid*img_c*blockDim.y*blockDim.z+tx + ty * img_c+tz*img_c*blockDim.y;
                idx_gx = bid_gx*img_c+tx;
                res[idx] = g_x[idx_gx];   
                tx += blockDim.x;
            }
                     
        }
        """
    )

    g_strides = cuda.to_device(np.asarray([img_h, img_w, c, strides[0], strides[1]], np.int32))
    # block = (kc, kw, kh)  # (x,y,z)
    block = (32, kw, kh)  # (x,y,z)
    grid = (w, h, bs)  # (x,y,z)
    res = np.zeros([bs, h, w, kh, kw, kc], np.float32)
    res = cuda.InOut(res)
    func = mod.get_function("img2col3")
    func(g_x, res, g_strides, grid=grid, block=block, shared=0, stream=Stream(0))

    del g_x
    g_strides.free()

    return res

def cuda_conv(g_x,name,weights,strides=(1,1),padding=(0,0)):
    """
    conv1_w:[32,3,5,5] => [32,5,5,3]
    conv1_b:[32]
    """
    conv1_w = weights[name+'.weight'].transpose(2,3,1,0)
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=conv1_w.dtype)

    kernel = conv1_w
    kh, kw, kc,kout = kernel.shape

    # res = np.zeros([bs, h, w, kh, kw, kc], np.float32)
    # for i in range(0, h, strides[0]):
    #     for j in range(0, w, strides[1]):
    #         res[:,i, j] = x[:,i:i + kh, j:j + kw]
    # #

    # res2 = np.zeros([bs, h, w, kh, kw, kc], np.float32)
    # for b in range(bs):
    #     for i in range(0, h, strides[0]):
    #         for j in range(0, w, strides[1]):
    #             for ki in range(kh):
    #                 for kj in range(kw):
    #                     for kcc in range(kc):
    #                         res2[b,i, j,ki,kj,kcc] = x[b,i + ki,j + kj,kcc]


    # res = np.reshape(res, [bs,h, w, -1])


    # ------------------------------------------
    mod = SourceModule(
        """
        __global__ void matmul(float *res,float *g_w,float *g_out)
        {
            int bz = blockIdx.z; // bs*h
            int by = blockIdx.y;  // w
            int bx = blockIdx.x;  // kc*kw*kh
            
            int tx = threadIdx.x; // kout
            
            //g_out[bz*gridDim.y*blockDim.x+by*blockDim.x+tx] += 
            //res[bz*gridDim.y*gridDim.x+by*gridDim.x+bx]*g_w[bx*blockDim.x+tx];//+g_b[tx]
            
            atomicAdd(&g_out[bz*gridDim.y*blockDim.x+by*blockDim.x+tx],
            res[bz*gridDim.y*gridDim.x+by*gridDim.x+bx]*g_w[bx*blockDim.x+tx]); // +g_b[tx]
              
        }
        
        __global__ void add(float *g_out,float *g_b)
        {
             int bz = blockIdx.z; // bs*h
             int by = blockIdx.y;  // w
             int bx = blockIdx.x;  // kout
             
             g_out[bz*gridDim.y*gridDim.x+by*gridDim.x+bx] += g_b[bx];
        }
        
        """
        )

    res = img2col(g_x,kernel,strides,padding)
    bs, h, w, kh, kw, kc = res.array.shape

    # -------------------------------------------------------
    func = mod.get_function("matmul")
    g_w = cuda.to_device(kernel)
    g_b = cuda.to_device(conv1_b)
    g_out = cuda.InOut(np.zeros([bs, h, w, kout], np.float32))
    # block = (kc, kw, kh)  # (x,y,z)
    kernel_in = kc*kw*kh
    block = (kout,1,1)  # (x,y,z)
    grid = (kernel_in,w,bs*h)  # (x,y,z)
    func(res,g_w,g_out,grid=grid, block=block, shared=0, stream=Stream(0))

    # -------------------------------------------------------
    func = mod.get_function("add")
    block = (1, 1, 1)  # (x,y,z)
    grid = (kout, w, bs * h)  # (x,y,z)
    func(g_out,g_b,grid=grid, block=block, shared=0, stream=Stream(0))

    # ---------------------------------------------
    # free
    g_w.free()
    g_b.free()
    del res

    return g_out

def np_convV2(x,name,weights,strides=(1,1),padding=(0,0)):
    # x = x.transpose(0,2,3,1)
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

    return (np.dot(res, kernel)+conv1_b[None,None,None,:])


def softmax(x:np.array):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

def cuda_softmax(g_x):
    x_shape = g_x.array.shape
    mod = SourceModule("""
        __global__ void fexp(float *g_x)
        {
            int by = blockIdx.y; // bs
            int bx = blockIdx.x; // 1000
            
            g_x[by*gridDim.x+bx] = expf(g_x[by*gridDim.x+bx]);
        }
        
        __global__ void fsum(float *g_x,float *g_out)
        {
            int by = blockIdx.y; // bs
            int bx = blockIdx.x; // 1000
            
            atomicAdd(&g_out[by],g_x[by*gridDim.x+bx]);
        }
        
        __global__ void fdiv(float *g_x,float *g_out)
        {
            int by = blockIdx.y; // bs
            int bx = blockIdx.x; // 1000
            
            g_x[by*gridDim.x+bx] /=g_out[by];
        }
        
    """)
    func = mod.get_function("fexp")
    func(g_x,grid=(x_shape[1],x_shape[0],1),block=(1,1,1))

    func = mod.get_function("fsum")
    g_out = cuda.to_device(np.zeros([x_shape[0],],np.float32))
    func(g_x,g_out, grid=(x_shape[1], x_shape[0], 1), block=(1, 1, 1))

    func = mod.get_function("fdiv")
    func(g_x, g_out, grid=(x_shape[1], x_shape[0], 1), block=(1, 1, 1))

    # free
    g_out.free()

    return g_x

def relu(x:np.array):
    s = np.where(x < 0, 0, x)
    return s

def cuda_relu(g_x):
    x_shape = g_x.array.shape
    x_size = np.prod(x_shape)

    mod =SourceModule("""
        __global__ void relu(float *g_out,int *g_shape)
        {
            int tx = threadIdx.x;
            const int N = g_shape[0];
            while(tx<N)
            {
                g_out[tx] = fmaxf(g_out[tx],0);
                // g_out[tx] = g_out[tx]<0?0:g_out[tx];
                tx += blockDim.x;
            }
        }
    """)
    g_shape = cuda.to_device(np.asarray([x_size],np.int32))
    func = mod.get_function("relu")
    func(g_x,g_shape,grid=(1,1,1),block=(256,1,1),shared=0,stream=Stream(0))

    # free
    g_shape.free()

    return g_x

def np_fc(x:np.array,name,weights):
    fc_w = weights[name + '.weight']
    fc_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([fc_w.shape[0]], dtype=fc_w.dtype)
    x = np.matmul(x, fc_w.T) + fc_b

    return x

def cuda_fc(g_x,name,weights):
    fc_w = weights[name + '.weight'].T
    fc_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([fc_w.shape[0]], dtype=fc_w.dtype)
    kc, kout = fc_w.shape
    bs = g_x.array.shape[0]

    mod = SourceModule(
        """
        __global__ void matmul(float *g_x,float *g_w,float *g_out)
        {
            int bz = blockIdx.z; // bs
            int by = blockIdx.y;  // kc
            int bx = blockIdx.x;  // kout

            atomicAdd(&g_out[bz*gridDim.x+bx],g_x[bz*gridDim.y+by]*g_w[by*gridDim.x+bx]);
        }

        __global__ void add(float *g_out,float *g_b)
        {
             int by = blockIdx.y;  // bs
             int bx = blockIdx.x;  // kout

             g_out[by*gridDim.x+bx] += g_b[bx];
        }

        """
    )

    func = mod.get_function("matmul")
    g_w = cuda.to_device(fc_w)
    g_b = cuda.to_device(fc_b)
    g_out = cuda.InOut(np.zeros([bs,kout], np.float32))
    block = (1, 1, 1)  # (x,y,z)
    grid = (kout, kc, bs)  # (x,y,z)
    func(g_x, g_w,g_out, grid=grid, block=block, shared=0, stream=Stream(0))
    # ------------------------------------------------
    func = mod.get_function("add")
    block = (1, 1, 1)  # (x,y,z)
    grid = (kout, bs, 1)  # (x,y,z)
    func(g_out, g_b, grid=grid, block=block, shared=0, stream=Stream(0))


    # free
    g_w.free()
    g_b.free()
    del g_x

    return g_out

def np_maxpool2x2(x:np.array,ks=(2,2),strides=(2,2)):
    kh,kw = ks
    bs,  h, w,c = x.shape
    res = np.zeros([bs,  h//2, w//2,c], np.float32)
    for i in range(0, h, strides[0]):
        for j in range(0, w, strides[1]):
            if i//2>=h//2 or j//2>=w//2:continue
            res[:,i//2,j//2,:] = x[:,i:i+kh,j:j+kw,:].max((1,2))

    return res

def cuda_maxpool2x2(g_x):
    bs, h, w, c = g_x.array.shape
    g_out = cuda.InOut(np.zeros([bs,h//2,w//2,c],np.float32))
    g_shape = cuda.to_device(np.asarray([h,w,c],np.int32))

    new_h,new_w = h//2*2,w//2*2

    mod = SourceModule("""
        __global__ void maxpool2d(float *g_x,float *g_out,int *g_shape)
        {
            int img_h = g_shape[0];
            int img_w = g_shape[1];
            int img_c = g_shape[2];
            
            int out_h = img_h/2;
            int out_w = img_w/2;
            
            int bs = blockIdx.z;
            int hi = blockIdx.y;
            int hj = blockIdx.x;
            
            int hc = threadIdx.x;
            
            if(hi%2!=0 || hj%2!=0) return;
            
            while(hc<img_c)
            {
                g_out[bs*out_h*out_w*img_c+hi/2*out_w*img_c+hj/2*img_c+hc] = fmaxf(fmaxf(
                                    g_x[bs*img_h*img_w*img_c+hi*img_w*img_c+hj*img_c+hc],
                                    g_x[bs*img_h*img_w*img_c+(hi+1)*img_w*img_c+hj*img_c+hc]),                                    
                                    fmaxf(
                                    g_x[bs*img_h*img_w*img_c+hi*img_w*img_c+(hj+1)*img_c+hc],
                                    g_x[bs*img_h*img_w*img_c+(hi+1)*img_w*img_c+(hj+1)*img_c+hc]));
                
                hc += blockDim.x;
            }
        }
    """)

    func = mod.get_function("maxpool2d")
    func(g_x,g_out,g_shape,grid=(new_w,new_h,bs),block=(256,1,1), shared=0, stream=Stream(0))

    # free
    del g_x
    g_shape.free()

    return g_out

def cuda_reshape(g_x):
    """[8,2,2,256]->[8,256,2,2]"""
    x_shape = g_x.array.shape
    bs,h,w,c = x_shape
    in_s = h*w*c
    mod = SourceModule("""
        __global__ void reshape(float *g_x,float *g_out,int *g_shape)
        {
            int h = g_shape[1];
            int w = g_shape[2];
            int c = g_shape[3];
            
            int by = blockIdx.y;// bs
            int bx = blockIdx.x;// h*w*c
            
            int tc = bx/(h*w); // c
            int th = bx%(h*w)/w; //h
            int tw = bx%(h*w)%w; //w
            
            int idx = th*w*c+tw*c+tc;
            
            g_out[by*gridDim.x+bx]=g_x[by*gridDim.x+idx];
            
        }
    """)
    g_shape = cuda.to_device(np.array(x_shape,np.int32))
    g_out = cuda.InOut(np.zeros([bs,c,h,w],np.float32))
    func = mod.get_function("reshape")
    func(g_x,g_out,g_shape,grid=(in_s,bs,1),block=(1,1,1))

    # free
    g_shape.free()
    del g_x

    return g_out

def np_network(x,weights):
    x = np_convV2(x, 'conv1', weights)
    x = np_bn(x, "bn1", weights)
    x = relu(x)
    x = np_maxpool2x2(x)

    x = np_convV2(x, 'conv2', weights)
    x = np_bn(x, "bn2", weights)
    x = relu(x)
    x = np_maxpool2x2(x)

    x = np_convV2(x, 'conv3', weights)
    x = np_bn(x, "bn3", weights)
    x = relu(x)
    x = np_maxpool2x2(x)

    x = np_convV2(x, 'conv4', weights, padding=(1, 1))
    x = np_bn(x, 'bn4', weights)
    x = relu(x)
    x = np_maxpool2x2(x)  # [8,2,2,256]

    x = x.transpose([0,3,1,2]) # [8,256,2,2]
    x = np.reshape(x, (-1, 1024))

    x = np_fc(x, 'fc1', weights)
    x = relu(x)

    x = np_fc(x, 'fc2', weights)
    x = softmax(x)
    return x

def cuda_network(x,weights):
    g_x = cuda.InOut(x)
    g_x = cuda_conv(g_x, 'conv1', weights)
    g_x = cuda_bn(g_x, "bn1", weights)
    g_x = cuda_relu(g_x)
    g_x = cuda_maxpool2x2(g_x)

    g_x = cuda_conv(g_x, 'conv2', weights)
    g_x = cuda_bn(g_x, "bn2", weights)
    g_x = cuda_relu(g_x)
    g_x = cuda_maxpool2x2(g_x)

    g_x = cuda_conv(g_x, 'conv3', weights)
    g_x = cuda_bn(g_x, "bn3", weights)
    g_x = cuda_relu(g_x)
    g_x = cuda_maxpool2x2(g_x)

    g_x = cuda_conv(g_x, 'conv4', weights,padding=(1,1))
    g_x = cuda_bn(g_x, "bn4", weights)
    g_x = cuda_relu(g_x)
    g_x = cuda_maxpool2x2(g_x)

    g_x = cuda_reshape(g_x)

    g_x = cuda_fc(g_x,"fc1",weights)
    g_x = cuda_relu(g_x)

    g_x = cuda_fc(g_x, "fc2", weights)
    g_x = cuda_softmax(g_x)
    # return g_x
    return g_x.array

if __name__=="__main__":
    weights = np.load("./model.npz")
    x = np.random.rand(32,64,64,3).astype(np.float32)
    np_x = np_network(x,weights)
    g_x = cuda_network(x,weights)

    print(np.max((np_x-g_x)**2))