#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>
#include "relu6.h"

namespace nvinfer1
{
    Relu6Plugin::Relu6Plugin()
    {
    }

    Relu6Plugin::~Relu6Plugin()
    {
    }

    // create the plugin at runtime from a byte stream
    Relu6Plugin::Relu6Plugin(const void* data, size_t length)
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);
    }

    void Relu6Plugin::serialize(void* buffer) const
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }

    size_t Relu6Plugin::getSerializationSize() const
    {  
        return sizeof(input_size_);
    }

    int Relu6Plugin::initialize()
    { 
        return 0;
    }

    Dims Relu6Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void Relu6Plugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* Relu6Plugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType Relu6Plugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool Relu6Plugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool Relu6Plugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void Relu6Plugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void Relu6Plugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void Relu6Plugin::detachFromContext() {}

    const char* Relu6Plugin::getPluginType() const
    {
        return "Relu6_TRT";
    }

    const char* Relu6Plugin::getPluginVersion() const
    {
        return "1";
    }

    void Relu6Plugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* Relu6Plugin::clone() const
    {
        Relu6Plugin *p = new Relu6Plugin();
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
    __global__ void relu6_kernel(const float *input, float *output, int num_elem) {

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
        output[idx] = fmin(fmax(input[idx],0.0f),6.0f); // 0.0~6.0
    }

    void Relu6Plugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        relu6_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize);
    }

    int Relu6Plugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection Relu6PluginCreator::mFC{};
    std::vector<PluginField> Relu6PluginCreator::mPluginAttributes;

    Relu6PluginCreator::Relu6PluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* Relu6PluginCreator::getPluginName() const
    {
            return "Relu6_TRT";
    }

    const char* Relu6PluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* Relu6PluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* Relu6PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        Relu6Plugin* obj = new Relu6Plugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* Relu6PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call Relu6Plugin::destroy()
        Relu6Plugin* obj = new Relu6Plugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}

