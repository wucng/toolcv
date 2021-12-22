/**
 * 需修改
 * ReluPlugin
        * ReluPlugin() //创建对象
        * ReluPlugin(const void* data, size_t length) // 反序列化 与ReluPlugin()相反
        * serialize //返回需要的内存大小，与ReluPlugin(const void* data, size_t length)对应
        * getSerializationSize // 设置创建对象需要的内存大小 与serialize对应
        * enqueue() //执行内核计算，重点修改
*/

#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "relu.h"

namespace nvinfer1
{
    
    const char *PLUGIN_NAME{"Relu6_TRT"};
    const char *PLUGIN_VERSION{"1"};

    ReluPlugin::ReluPlugin()
    {
    }

    ReluPlugin::~ReluPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    ReluPlugin::ReluPlugin(const void* data, size_t length)
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);//创建对象所需的内存大小
    }

    void ReluPlugin::serialize(void* buffer) const
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }

    size_t ReluPlugin::getSerializationSize() const
    {  
        return sizeof(input_size_);
    }

    int ReluPlugin::initialize()
    { 
        return 0;
    }

    Dims ReluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void ReluPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* ReluPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType ReluPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool ReluPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool ReluPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void ReluPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void ReluPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void ReluPlugin::detachFromContext() {}

    const char* ReluPlugin::getPluginType() const
    {
        return PLUGIN_NAME;
    }

    const char* ReluPlugin::getPluginVersion() const
    {
        return PLUGIN_VERSION;
    }

    void ReluPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* ReluPlugin::clone() const
    {
        ReluPlugin *p = new ReluPlugin();
        p->input_size_ = input_size_; 
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    /*
    __device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

    __device__ float softplus_kernel(float x, float threshold = 20) {
        if (x > threshold) return x;                // too large
        else if (x < -threshold) return expf(x);    // too small
        return logf(expf(x) + 1);
    }
    */
    __global__ void relu_kernel(const float *input, float *output, int num_elem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        //float t = exp(input[idx]);
        //if (input[idx] > 20.0) {
        //    t *= t;
        //    output[idx] = (t - 1.0) / (t + 1.0);
        //} else {
        //    float tt = t * t;
        //    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
        //}
        //output[idx] *= input[idx];
        // output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
        // output[idx] = input[idx]>=0?input[idx]:0.0f; // Relu
        output[idx] = fmin(fmax(input[idx],0.0f),6.0f); // 0.0~6.0 Relu6
    }

    void ReluPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        relu_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize);
    }

    int ReluPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection ReluPluginCreator::mFC{};
    std::vector<PluginField> ReluPluginCreator::mPluginAttributes;

    ReluPluginCreator::ReluPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* ReluPluginCreator::getPluginName() const
    {
            return PLUGIN_NAME;
    }

    const char* ReluPluginCreator::getPluginVersion() const
    {
            return PLUGIN_VERSION;
    }

    const PluginFieldCollection* ReluPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* ReluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        ReluPlugin* obj = new ReluPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* ReluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call ReluPlugin::destroy()
        ReluPlugin* obj = new ReluPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    REGISTER_TENSORRT_PLUGIN(ReluPluginCreator); // 注册到plugin

}