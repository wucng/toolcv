#ifndef RELU_KERNEL_H
#define RELU_KERNEL_H
#include "NvInfer.h"

int reluInference(
    cudaStream_t stream, // 流，一般默认为0
    int n, // 总数量 一般resize为一维
    const void* input, // 输入
    void* output); // 输出

#endif