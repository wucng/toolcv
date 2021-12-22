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


// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

namespace nvinfer1
{
    
    const char *PLUGIN_NAME{"LRelu_TRT_2"};
    const char *PLUGIN_VERSION{"1"};

    ReluPlugin::ReluPlugin()
    {
    }

    // LeakyReLU {{{
    ReluPlugin::ReluPlugin(float negSlope): mNegSlope(negSlope)
    {
    }

    ReluPlugin::~ReluPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    // ReluPlugin::ReluPlugin(const void* data, size_t length)
    // {
    //     assert(length == sizeof(input_size_));
    //     input_size_ = *reinterpret_cast<const int*>(data);//创建对象所需的内存大小
    // }

    ReluPlugin::ReluPlugin(const void* buffer, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(buffer), *a = d;
        mNegSlope = read<float>(d);
        assert(d == a + length);

        // input_size_ = d;
    }


    void ReluPlugin::serialize(void* buffer) const
    {
        // *reinterpret_cast<int*>(buffer) = input_size_;
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mNegSlope);
        assert(d == a + getSerializationSize());

    }

    size_t ReluPlugin::getSerializationSize() const
    {  
        // return sizeof(input_size_);
        // mNegSlope, mBatchDim
        // return sizeof(float) + sizeof(int);
        // mNegSlope
        return sizeof(float);
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
        // return inputs[0];
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
        ReluPlugin *p = new ReluPlugin(mNegSlope);
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
    __global__ void relu_kernel(const float *input, float *output, int num_elem,float mNegSlope) {

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
        // output[idx] = input[idx]>=0?input[idx]:0.0f;
        output[idx] = input[idx] > 0 ? input[idx] : input[idx] * mNegSlope;
    }

    void ReluPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize,float mNegSlope) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        relu_kernel<<<grid_size, block_size,0,stream>>>(inputs[0], output, input_size_ * batchSize,mNegSlope);
    }

    int ReluPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize,mNegSlope);
        return 0;
    }

    PluginFieldCollection ReluPluginCreator::mFC{};
    std::vector<PluginField> ReluPluginCreator::mPluginAttributes;

    ReluPluginCreator::ReluPluginCreator()
    {
        // mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));
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
        float negSlope;
        const PluginField* fields = fc->fields;
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kFLOAT32);
        negSlope = *(static_cast<const float*>(fields[0].data));
        ReluPlugin* obj = new ReluPlugin(negSlope);
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