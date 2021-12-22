/**
 * 需修改
 * FcPlugin
        * FcPlugin() //创建对象
        * FcPlugin(const void* data, size_t length) // 反序列化 与FcPlugin()相反
        * serialize //返回需要的内存大小，与FcPlugin(const void* data, size_t length)对应
        * getSerializationSize // 设置创建对象需要的内存大小 与serialize对应
        * enqueue() //执行内核计算，重点修改
*/

#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cublas_v2.h"
// #include <cuda_runtime_api.h>

// #include "Utils.h"
#include "fc.h"


namespace nvinfer1
{
    
    const char *PLUGIN_NAME{"Fc_TRT"};
    const char *PLUGIN_VERSION{"1"};

    const int INSIZE{512};
    const int OUTSIZE{1000};

    FcPlugin::FcPlugin()
    {
    }

    FcPlugin::FcPlugin(float *weights,float *bias)
    {
        hweight = weights;
        hbias = bias;

        std::cout<<__LINE__<<std::endl;
        for (int i=0;i<INSIZE*OUTSIZE;++i)
        {
            if(i%500==0)
                std::cout<<hweight[i]<<",";
        }
        std::cout<<std::endl;
        for (int i=0;i<OUTSIZE;++i)
        {
            if(i%500==0)
                std::cout<<hbias[i]<<",";
        }
       std::cout<<__LINE__<<std::endl;
    }

    // FcPlugin::FcPlugin(float *weights,int len_weight,float *bias,int len_bias):len_weight(len_weight),len_bias(len_bias)
    // {   
    //     hweight = weights;
    //     hbias = bias;
    //     // len_weight = len_weight;
    //     // len_bias =len_bias;

    //     std::cout<<__LINE__<<","<<len_weight<<","<<len_bias<<std::endl;
    // //     std::cout<<len_weight<<","<<len_bias<<std::endl;
    // //     for (int i=0;i<len_weight;++i)
    // //     {
    // //         if(i%500==0)
    // //             std::cout<<hweight[i]<<",";
    // //     }
    // //     std::cout<<std::endl;
    // //     for (int i=0;i<len_bias;++i)
    // //     {
    // //         if(i%500==0)
    // //             std::cout<<hbias[i]<<",";
    // //     }
    // //    std::cout<<std::endl;
    // }

    FcPlugin::~FcPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    FcPlugin::FcPlugin(const void* data, size_t length)
    {
        // assert(length == sizeof(input_size_));
        // input_size_ = *reinterpret_cast<const int*>(data);//+(len_bias+len_weight)*sizeof(float); //创建对象所需的内存大小
        // using namespace Tn;
        const char *d = static_cast<const char *>(data), *a = d;
        // read<int>(d,thread_count_);
        auto kernelSize =  INSIZE*OUTSIZE * sizeof(float);
        memcpy(hweight, d, kernelSize);
         d+=kernelSize;
        auto biasSize =  OUTSIZE * sizeof(float);
        memcpy(hbias, d, biasSize);
        d+=biasSize;

        assert(d == a + length);
    }

    void FcPlugin::serialize(void* buffer) const
    {
        // *reinterpret_cast<int*>(buffer) = input_size_;
        // using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        auto kernelSize =  INSIZE*OUTSIZE * sizeof(float);
        memcpy(hweight, d, kernelSize);
         d+=kernelSize;
        auto biasSize =  OUTSIZE * sizeof(float);
        memcpy(hbias, d, biasSize);
        d+=biasSize;

        assert(d == a + getSerializationSize());

    }

    size_t FcPlugin::getSerializationSize() const
    {  
        // return sizeof(input_size_);
        return  INSIZE*OUTSIZE * sizeof(float)+OUTSIZE * sizeof(float);
    }

    int FcPlugin::initialize()
    { 
        return 0;
    }

    Dims FcPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        // input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        // return Dims3(OUTSIZE, inputs[0].d[1], inputs[0].d[2]);
        return Dims3(OUTSIZE,1,1);
    }

    // Set plugin namespace
    void FcPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* FcPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType FcPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool FcPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool FcPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void FcPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void FcPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void FcPlugin::detachFromContext() {}

    const char* FcPlugin::getPluginType() const
    {
        return PLUGIN_NAME;
    }

    const char* FcPlugin::getPluginVersion() const
    {
        return PLUGIN_VERSION;
    }

    void FcPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* FcPlugin::clone() const
    {
        // FcPlugin *p = new FcPlugin();
        FcPlugin *p = new FcPlugin(hweight,hbias);
        p->input_size_ = input_size_; 
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

/*
    void FcPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        //C=alpha*A*B+beta*C
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        float alpha = 1.0f;
        float beta = 0.0f;

        const int len = INSIZE*OUTSIZE;
        float *dB = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        // float dB[len];
        for (int i=0;i<len;++i)
        {
            dB[i]=1.0f;
        }

        // for(int i=0;i<batchSize*OUTSIZE;++i)
        // {
        //     output[i]=0.0f;
        // }

        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,OUTSIZE,batchSize,INSIZE,&alpha,dB,OUTSIZE,inputs[0],INSIZE,&beta,output,OUTSIZE);
    }
*/

   __global__ void matmul_kernel(const float *input, float *dweight,float *dbias,float *output, const int num_elem)
   {
       int bx = blockIdx.x;//output
       int by = blockIdx.y;//batchsize

        for(int i=0;i<num_elem;++i)//insize
        {
            // output[by,bx] += input[by,i]*dweight[i,bx];
            output[by*gridDim.x+bx] += input[by*num_elem+i]*dweight[i*gridDim.x+bx];
        }
        // output[by,bx] += dbias[bx];
        output[by*gridDim.x+bx] += dbias[bx];
   }

    void FcPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        std::cout<<__LINE__<<std::endl;

        float* buffers[2];
        int len_weight = INSIZE*OUTSIZE;
        int len_bias = OUTSIZE;
        std::cout<<__LINE__<<std::endl;
        cudaMalloc(&buffers[0], len_weight * sizeof(float));
        cudaMalloc(&buffers[1], len_bias* sizeof(float));
        std::cout<<__LINE__<<std::endl;
        cudaMemcpyAsync(buffers[0], hweight, len_weight * sizeof(float) , cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[1], hbias, len_bias * sizeof(float) , cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        std::cout<<__LINE__<<std::endl;
        dim3 grid(len_bias,batchSize,1); // (x,y,z)
        matmul_kernel<<<grid,1>>>(inputs[0],buffers[0],buffers[1],output,INSIZE);
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        std::cout<<__LINE__<<","<<batchSize<<","<<len_bias<<std::endl;

        float hout[batchSize*len_bias];
        cudaMemcpyAsync(hout, output, batchSize*len_bias * sizeof(float) , cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i=0;i<batchSize*len_bias;++i)
        {
            if (i%500==0)
                std::cout<<hout[i]<<",";
        }

    }

    int FcPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection FcPluginCreator::mFC{};
    std::vector<PluginField> FcPluginCreator::mPluginAttributes;

    FcPluginCreator::FcPluginCreator()
    {
        mPluginAttributes.clear();
        //-------------------------------------------------------
        mPluginAttributes.emplace_back(nvinfer1::PluginField("weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        //--------------------------------------------------------
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* FcPluginCreator::getPluginName() const
    {
            return PLUGIN_NAME;
    }

    const char* FcPluginCreator::getPluginVersion() const
    {
            return PLUGIN_VERSION;
    }

    const PluginFieldCollection* FcPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    // IPluginV2IOExt* FcPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    // {
    //     FcPlugin* obj = new FcPlugin();
    //     obj->setPluginNamespace(mNamespace.c_str());
    //     return obj;
    // }

    IPluginV2IOExt* FcPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        float *weight,*bias;
        int len_weight,len_bias;
        const nvinfer1::PluginField* fields = fc->fields;
         for (int i=0; i<fc->nbFields; i++) {
            const char* attrName = fields[i].name;
            if(strcmp(attrName, "weight")==0) {
                len_weight = fields[i].length;
                cudaMallocHost((void**)&weight,len_weight*sizeof(float));
                memcpy(weight,fields[i].data,len_weight);
                
            }
            if(strcmp(attrName, "bias")==0) {
                len_bias = fields[i].length;
                cudaMallocHost((void**)&bias,len_bias*sizeof(float));
                memcpy(bias,fields[i].data,len_bias);
            }
         }

        std::cout<<__LINE__<<","<<len_weight<<","<<len_bias<<std::endl;
        FcPlugin* obj = new FcPlugin(weight,bias);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* FcPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call FcPlugin::destroy()
        FcPlugin* obj = new FcPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}