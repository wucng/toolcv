
/**
 * 由v5版本的 IPluginV2Ext，
 * 到v6版本的 IPluginV2IOExt 和 IPluginV2DynamicExt(动态plugin)。
*/
#ifndef RELU_PLUGIN_H
#define RELU_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>


using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class ReluPlugin : public IPluginV2
{
public:
    /***********************修改的地方***********************************/
    ReluPlugin(const std::string name); // 
    //ReluPlugin(const std::string name, float clipMin, float clipMax);

    ReluPlugin(const std::string name, const void* data, size_t length);
    // It doesn't make sense to make UpsamplePlugin without arguments, so we delete default constructor.
    ReluPlugin() = delete;
    /**********************************************************/

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    /***********************修改的地方***********************************/
    // float mClipMin, mClipMax;
    /**********************************************************/
    size_t mInputVolume;
    std::string mNamespace;

};


class ReluPluginCreator : public IPluginCreator
{
public:
    /***********************修改的地方***********************************/
    ReluPluginCreator();
    /**********************************************************/
    
    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    
    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif