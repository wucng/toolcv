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

#define CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

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
    
    const char *PLUGIN_NAME{"FC_TRT_2"};
    const char *PLUGIN_VERSION{"1"};

    const int INSIZE{3};
    const int OUTSIZE{5};

    ReluPlugin::ReluPlugin()
    {
    }

    ReluPlugin::ReluPlugin(float *weights,float *bias)
    {
        hweight = weights;
        hbias = bias;

        std::cout<<__LINE__<<std::endl;

        for (int i=0;i<INSIZE*OUTSIZE;++i)
        {
            std::cout<<hweight[i]<<",";
        }
        std::cout<<std::endl;

        for (int i=0;i<OUTSIZE;++i)
        {
            std::cout<<hbias[i]<<",";
        }
        std::cout<<std::endl;
       std::cout<<__LINE__<<std::endl;
    }


    ReluPlugin::~ReluPlugin()
    {
        if(!hweight)
        {
            CHECK(cudaFreeHost(hweight));
            hweight = nullptr;
        }
        if(!hbias)
        {
            CHECK(cudaFreeHost(hbias));
            hbias = nullptr;
        }
    }

    ReluPlugin::ReluPlugin(const void* buffer, size_t length)
    {
        // 反系列化文件 从 buffer 读取 权重
        const char *d = reinterpret_cast<const char*>(buffer), *a = d;
        // mClipMin = read<float>(d);
        // mClipMax = read<float>(d);
        // assert(d == a + length);

        auto kernelSize =  INSIZE*OUTSIZE * sizeof(float);
        CHECK(cudaMallocHost((void**)&hweight,kernelSize));
        memcpy(hweight, d, kernelSize);
        d+=kernelSize;

        auto biasSize =  OUTSIZE * sizeof(float);
        CHECK(cudaMallocHost((void**)&hbias,biasSize));
        memcpy(hbias, d, biasSize);
        d+=biasSize;

        assert(d == a + length);

        // input_size_ = d;

        std::cout<<__LINE__<<std::endl;

        for (int i=0;i<INSIZE*OUTSIZE;++i)
        {
            std::cout<<hweight[i]<<",";
        }
        std::cout<<std::endl;

        for (int i=0;i<OUTSIZE;++i)
        {
            std::cout<<hbias[i]<<",";
        }
        std::cout<<std::endl;
       std::cout<<__LINE__<<std::endl;

    }


    void ReluPlugin::serialize(void* buffer) const
    {
        // 系列化文件 将权重 写入 buffer
        // *reinterpret_cast<int*>(buffer) = input_size_;
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        // write(d, mClipMin);
        // write(d, mClipMax);
        // assert(d == a + getSerializationSize());
        auto kernelSize =  INSIZE*OUTSIZE * sizeof(float);
        // memcpy(hweight, d, kernelSize);
        memcpy(d,hweight, kernelSize);
        d+=kernelSize;
        auto biasSize =  OUTSIZE * sizeof(float);
        // memcpy(hbias, d, biasSize);
        memcpy(d, hbias, biasSize);
        d+=biasSize;

        assert(d == a + getSerializationSize());
    }

    size_t ReluPlugin::getSerializationSize() const
    {  
        // return sizeof(input_size_);
        // mNegSlope, mBatchDim
        // return sizeof(float) + sizeof(int);
        // mClipMin + mClipMax
        // return sizeof(float)+sizeof(float);
        return  INSIZE*OUTSIZE * sizeof(float)+OUTSIZE * sizeof(float); // weigih + bias
    }

    int ReluPlugin::initialize()
    { 
        return 0;
    }

    Dims ReluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        // input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        // return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
        // return inputs[0];
        return Dims3(OUTSIZE,1,1);
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
        ReluPlugin *p = new ReluPlugin(hweight,hbias);
        p->input_size_ = input_size_; 
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    /*
    void ReluPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        // https://blog.csdn.net/weixin_33708432/article/details/86365370
        // 使用 cublasSgemm  C=alpha*A*B+beta*C
        // 分配GPU空间
        // float *d_w,*d_b;
        float *d_w;
        CHECK(cudaMalloc((void **)&d_w,INSIZE*OUTSIZE*sizeof(float)));
        // CHECK(cudaMalloc((void **)&d_b,OUTSIZE*sizeof(float)));

        // CPU -> GPU
        CHECK(cudaMemcpy(d_w,hweight,INSIZE*OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_b,hbias,OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
        if(batchSize==1)
        {
            CHECK(cudaMemcpy(output,hbias,OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
        }
        else
        {
            for(int i =0;i < batchSize;i++)
            {
                CHECK(cudaMemcpy(output+i*OUTSIZE,hbias,OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
            }
        }

        float alpha = 1;
        float beta = 1;

        // A: inputs[0] batchSize x INSIZE
        // B: d_w INSIZE x OUTSIZE
        // C: output batchSize x OUTSIZE
        // C=A*B+C
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle,
            CUBLAS_OP_N,  
            CUBLAS_OP_N, 
            OUTSIZE, //矩阵B的列数
            batchSize, //矩阵A的行数
            INSIZE, //矩阵A的列数
            &alpha,
            d_w,
            OUTSIZE,
            inputs[0],
            INSIZE,
            &beta,
            output,
            OUTSIZE);
        
        // 销毁句柄
	    cublasDestroy(handle);
        CHECK(cudaFree(d_w));
    }
    */

    
    __global__ void matmul_kernel(const float *input, float *dweight,float *dbias,float *output, const int num_elem)
    {
        int bx = blockIdx.x; //OUTSIZE
        int by = blockIdx.y; //batchsize
        int idx = blockIdx.x + gridDim.x*blockIdx.y;

        for(int i=0;i<num_elem;++i){ //insize
            // output[by,bx] += input[by,i]*dweight[i,bx];
            output[by*gridDim.x+bx] += input[by*num_elem+i]*dweight[i*gridDim.x+bx];
        }
        // output[by,bx] += dbias[bx];
        output[by*gridDim.x+bx] += dbias[bx];
    }

    void ReluPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        // 使用 cuda 实现 C = A*B + b
        // 分配GPU空间
        float *d_w,*d_b;
        CHECK(cudaMalloc((void **)&d_w,INSIZE*OUTSIZE*sizeof(float)));
        CHECK(cudaMalloc((void **)&d_b,OUTSIZE*sizeof(float)));

        // CPU -> GPU
        // CHECK(cudaMemcpy(d_w,hweight,INSIZE*OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_b,hbias,OUTSIZE*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyAsync(d_w,hweight,INSIZE*OUTSIZE*sizeof(float),cudaMemcpyHostToDevice,stream));
        CHECK(cudaMemcpyAsync(d_b,hbias,OUTSIZE*sizeof(float),cudaMemcpyHostToDevice,stream));
        CHECK(cudaStreamSynchronize(stream));

        // C = A*B + b
        // A: inputs[0] batchSize x INSIZE
        // B: d_w INSIZE x OUTSIZE
        // b: d_b OUTSIZE
        // C: output batchSize x OUTSIZE

        dim3 grid(OUTSIZE,batchSize,1); // (x,y,z)
        matmul_kernel<<<grid,1,0,stream>>>(inputs[0],d_w,d_b,output,INSIZE);
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaDeviceSynchronize());

        // float hout[batchSize*OUTSIZE];
        // cudaMemcpyAsync(hout, output, batchSize*OUTSIZE * sizeof(float) , cudaMemcpyDeviceToHost, stream);
        // cudaStreamSynchronize(stream);

        CHECK(cudaFree(d_w));
        CHECK(cudaFree(d_b));

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
        // mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1)); // 最后一个参数 对应 数据长度
        // mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

        mPluginAttributes.emplace_back(nvinfer1::PluginField("weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, INSIZE*OUTSIZE));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, OUTSIZE));

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
        float *weight,*bias;
        int len_weight,len_bias;
        const nvinfer1::PluginField* fields = fc->fields;

        for (int i=0; i<fc->nbFields; i++) {
            const char* attrName = fields[i].name;
            if(strcmp(attrName, "weight")==0) {
                len_weight = fields[i].length;
                CHECK(cudaMallocHost((void**)&weight,len_weight*sizeof(float)));
                memcpy(weight,fields[i].data,len_weight*sizeof(float));
                
            }
            if(strcmp(attrName, "bias")==0) {
                len_bias = fields[i].length;
                CHECK(cudaMallocHost((void**)&bias,len_bias*sizeof(float)));
                memcpy(bias,fields[i].data,len_bias*sizeof(float));
            }
         }
        std::cout<<__LINE__<<","<<len_weight<<","<<len_bias<<std::endl;
        ReluPlugin* obj = new ReluPlugin(weight,bias);
        
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;

        /*
        std::array<Weights, 2> weights{};
        float *weight,*bias;
        int len_weight,len_bias;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("weight") == 0)
            {
                weights[0].values = fc->fields[i].data;
                weights[0].count = fc->fields[i].length;
                weights[0].type = nvinfer1::DataType::kFLOAT;

                len_weight = weights[0].count;
                cudaMallocHost((void**)&weight,len_weight*sizeof(float));
                // memcpy(weight,weights[0].values,len_weight);
                for(int i=0;i<weights[0].count;i++)
                    weight[i] = *((float*)weights[0].values+i);

                std::cout<<__LINE__<<std::endl;
                for(int i=0;i<weights[0].count;i++)
                    std::cout<<*((float*)weight+i)<<",";
                std::cout<<__LINE__<<std::endl;
            }
            if (fieldName.compare("bias") == 0)
            {
                weights[1].values = fc->fields[i].data;
                weights[1].count = fc->fields[i].length;
                weights[1].type = nvinfer1::DataType::kFLOAT;

                len_bias = weights[1].count;
                cudaMallocHost((void**)&bias,len_bias*sizeof(float));
                // memcpy(bias,weights[1].values,len_bias);
                for(int i=0;i<weights[1].count;i++)
                    bias[i] = *((float*)weights[1].values+i);

                std::cout<<__LINE__<<std::endl;
                for(int i=0;i<weights[1].count;i++)
                    std::cout<<*((float*)bias+i)<<",";
                std::cout<<__LINE__<<std::endl;
            }
        }

        // return new ReluPlugin(static_cast<void*>(weights.data()), weights.size());
        ReluPlugin* obj = new ReluPlugin(weight,bias);
        
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
        */
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