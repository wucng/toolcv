#ifndef _RELU_PLUGIN_H
#define _RELU_PLUGIN_H

#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"
#include <array>

// #include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
// #include <cuda_runtime_api.h>

namespace nvinfer1
{
    class ReluPlugin: public IPluginV2IOExt
    {
        public:
            explicit ReluPlugin();

            // ReluPlugin(const Weights* weights, int nbWeights);

            ReluPlugin(float *weights,float *bias);

            ReluPlugin(const void* data, size_t length);

            ~ReluPlugin();

            /**
             * @brief 获取图层的输出数量。
             * INetworkDefinition和IBuilder的实现调用此函数。 特别是，它在对initialize() 的任何调用之前被调用。
            */
            int getNbOutputs() const override
            {
                return 1;
            }

            /**
             * @brief 获取输出tensor的维度。
             * @param index 输出张量的索引。
             * @param inputs 输入tensor
             *  @param nbInputDims 输入tensor的数量。
             * INetworkDefinition和IBuilder的实现调用此函数。 特别是，它在对initialize（）的任何调用之前被调用。
            */
            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            /**
             * @brief 初始化执行层。 创建引擎时将调用此方法。
             * @return 0为成功，否则为非零（这将导致引擎终止）。
            */
            int initialize() override;

            /**
             * @brief 释放在插件层初始化期间获取的资源。 engine销毁时调用。
             * 与  initialize() 对应
            */
            virtual void terminate() override {};

            /**
             * @brief 查找该层所需的工作空间大小。
             * 在initialize（）之后，在引擎启动期间调用此函数。 返回的工作空间大小应足以容纳最大数量的任何批处理大小。
            */
            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            /**
             * @brief 执行层
             * @param batchSize 输入的batch大小
             * @param inputs 输入tensor内存
             * @param outputs 输出tenosr内存
             * @param workspace  执行工作空间。
             * @param stream 执行内核的流
             * @return 0为成功，否则为非零（这将导致引擎终止）。
            */
            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            /**
             * @brief 查找所需的序列化缓冲区的大小。 
             * @return 序列化缓冲区的大小。
            */
            virtual size_t getSerializationSize() const override;

            /**
             * @brief 序列化层
             * @param buffer 指向要序列化数据的缓冲区的指针。 缓冲区的大小必须等于getSerializationSize返回的值。
             * @see getSerializationSize()
            */
            virtual void serialize(void* buffer) const override;

            /**
             * @brief 如果插件支持pos索引的输入/输出的格式和数据类型，则返回true。
             * 
             * 对于此方法，输入的编号为0 ..（nbInputs-1），输出的编号为nbInputs ..（nbInputs + nbOutputs-1）。
             * 使用此编号，pos是InOut的索引，其中0 <= pos <nbInputs + nbOutputs- 1。
             * 
             * examples:
             * 
             *      仅支持FP16 NCHW的插件的定义：
             *      return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
             * 
             *       一个插件的定义，该插件的两个输入仅支持FP16 NCHW，单个输出仅支持FP32 NCHW：
             *      return inOut.format[pos] == TensorFormat::kLINEAR && (inOut.type[pos] == pos < 2 ?  DataType::kHALF : DataType::kFLOAT);
             * 
             *      具有两个输入和一个支持任何格式或类型的输出的“多态”插件的定义，但是输入和输出必须具有相同的格式和类型：
             *      return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
            */
            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            /**
             * @return 返回插件类型字符串，如："Relu_TRT" 。 应与相应插件创建者返回的插件名称匹配
             * see IPluginCreator::getPluginName() 获取 "Relu_TRT"
            */
            const char* getPluginType() const override;


            /**
             * @return 返回插件版本号。 应与相应插件创建者返回的插件版本匹配
             * see IPluginCreator::getPluginVersion() 获取 返回的版本号，必须匹配
            */
            const char* getPluginVersion() const override;

            /**
             * @brief 销毁插件对象。 当网络，构建器或引擎被销毁时，将调用此方法。
            */
            void destroy() override;

            /**
             * @brief 克隆插件对象。 这将复制内部插件参数，并返回带有这些参数的新插件对象。
             * 
             * 克隆插件对象。 这也会复制内部插件参数，并返回带有这些参数的新插件对象。
             * 如果源插件已使用configurePlugin（）进行了预配置，则返回的对象也应进行预配置。 
             * 返回的对象应允许attachToContext（）具有新的执行上下文。 克隆的插件对象可以与源对象
             * （例如通过引用计数）共享相同的每个引擎不变资源（例如权重），以避免重复。
            */
            IPluginV2IOExt* clone() const override;

            /**
             * @brief 设置此插件对象所属的命名空间。 理想情况下，来自同一插件库的所有插件对象应具有相同的命名空间。
            */
            void setPluginNamespace(const char* pluginNamespace) override;

            /**
             * @brief 返回插件对象的命名空间。
            */
            const char* getPluginNamespace() const override;

            /**
             * 在请求的索引处返回插件输出的DataType。默认行为是返回第一个输入的类型，如果该层没有输入，
             * 则返回DataType :: kFLOAT。 返回的数据类型必须具有插件支持的格式。
             * see supportsFormat()
             * @warning  DataType:kBOOL 不支持
            */
            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            /**
             * @brief 如果输出张量跨批次广播，则返回true。
             * @param outputIndex 输出索引
             * @param inputIsBroadcasted 如果第i个输入的张量在批处理中广播，则ith元素为true。
             * @param nbInputs 输出的数量
             * 
             *  inputIsBroadcasted中的值是指语义级别的广播，即不受canBroadcastInputAcrossBatch方法是否请求值的物理复制的影响。
            */
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            /**
             * @brief 如果插件可以使用跨批广播而不复制的输入，则返回true。
             * @param inputIndex 可以广播的输入的索引。
             * 
             * 对于每个张量在语义上跨批次广播的输入，TensorRT会在调用configurePlugin之前调用此方法。
             * 如果canBroadcastInputAcrossBatch返回true，TensorRT将不会复制输入张量;即，插件应在整个
             * 批次中共享一个副本。 如果返回false，TensorRT将复制输入张量，使其看起来像未广播的张量。
             * 
             * 此方法仅针对可广播的输入被调用。
            */
            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            /**
             * @brief 将插件对象附加到执行上下文，并向插件授予对某些上下文资源的访问权限。
             * @param cudnn  执行上下文的cudnn上下文句柄
             * @param cublas 执行上下文的cublas上下文句柄
             * @param allocator 行上下文使用的分配器
             * 
             * @ 创建新的执行上下文时，将自动为每个插件调用此函数。 如果插件需要按上下文分配资源，则可以在此处分配它。
             * 插件也可以在此处获取上下文拥有的CUDNN和CUBLAS上下文。
            */
            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
            

            /**
             * @brief 使用输入和输出数据类型配置层。
             * 构建器在initialize（）之前调用此函数。 它为该层提供了一个机会，可以根据其权重，维度，数据类型和最大批处理大小来选择算法。
             * @param inputDims  输入tensor的维度
             * @param nbInputs 输入的数量
             * @param outputDims 输出tensor的维度
             * @param nbOutputs 输出的数量
             * @param inputTypes 为插件输入选择的数据类型。
             * @param outputTypes 为插件输出选择的数据类型。
             * @param inputIsBroadcast 对于插件必须在批次中广播的每个输入为True。
             * @param outputIsBroadcast TensorRT将在批次中广播的每个输出为True。
             * @param floatFormat 为引擎选择的浮点输入/输出格式。
             * @param maxBatchSize 最大batchsize
             * 
             * 此处传递的尺寸不包括最外面的批量尺寸（即对于2D图像网络，它们将是3维CHW尺寸）。 
             * 当inputIsBroadcast或outputIsBroadcast为true时，应将该输入或输出的最外面的批处理大小视为1。

            * 仅当输入在批处理中在语义上进行广播并且canBroadcastInputAcrossBatch（i）返回true时，inputIsBroadcast [i]才为true。

            * 仅当以下情况时outputIsBroadcast [i]为true
            * isOutputBroadcastAcrossBatch（i）返回true。

            * 对于floatFormat字段的警告，将不会传入值PluginFormat :: kCHW4，PluginFormat :: kCHW16和PluginFormat :: kCHW32，
            * 这是为了保持与TensorRT 5.x系列的向后兼容性。 将PluginV2IOExt或PluginV2DynamicExt用于其他PluginFormat。
             * 
            */
            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            /**
             * @brief 从其执行上下文中分离插件对象。
             * 
             * 销毁执行上下文时，将为每个插件自动调用此函数。 如果该插件拥有按上下文资源，则可以在此处发布它。
            */
            void detachFromContext() override;

            int input_size_;
        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            const char* mPluginNamespace;

            float *hweight=nullptr;
            // int len_weight;
            float *hbias=nullptr;
            // int len_bias;
    };

    class ReluPluginCreator : public IPluginCreator
    {
        public:
            ReluPluginCreator();

            ~ReluPluginCreator() override = default;

            /**
             * @brief 获取定义的plugin的名称，与IPluginV2IOExt::getPluginType对应
            */
            const char* getPluginName() const override;

            /**
             * @brief 获取定义的plugin的版本，与IPluginV2IOExt::getPluginVersion对应
            */
            const char* getPluginVersion() const override;

            /**
             * @brief 返回需要传递给createPlugin的字段列表。
             * @see PluginFieldCollection
            */
            const PluginFieldCollection* getFieldNames() override;

            /**
             * @brief 返回一个插件对象。 出现错误时返回nullptr。
            */
            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            /**
             * @brief 在插件层反序列化期间调用。 返回一个插件对象。
            */
            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            /**
             * @brief 根据插件创建者所属的插件库设置其名称空间。 可以在注册插件创建者时设置。
             * @see IPluginRegistry::registerCreator()
            */
            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            /**
             * @brief 返回插件创建者对象的名称空间。
            */
            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
};
#endif 
