#include <reluKernel.h>

template <typename T>
__device__ __forceinline__ T& filter(const T& a,const T& b) // b=0
{
    return a<b?b:a;
} 


template <typename T, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void reluKernel(
        int n,
        const T* input,
        T* output)
{
    int idx= blockIdx.x * nthdsPerCTA + threadIdx.x;
    // int idx= blockIdx.x * blockDim.x + threadIdx.x;
    while(idx<n)
    {
        output[idx] = filter<T>(input[idx],(T)0);
        idx += gridDim.x * nthdsPerCTA;
    }

    /*
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = filter<T>(input[i],(T)0);
    }
    */
}


int reluInference(
    cudaStream_t stream, // 流，一般默认为0
    int n, // 总数量 一般resize为一维
    const void* input, // 输入
    void* output) // 输出
{
    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;

    reluKernel<float,blockSize><<<gridSize, blockSize, 0, stream>>>(n,
                                        static_cast<const float*>(input),
                                        static_cast<float*>(output));
    
    return 0;
}