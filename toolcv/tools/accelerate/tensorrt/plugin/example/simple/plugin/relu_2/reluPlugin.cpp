#include "reluPlugin.h" // 修改
#include "NvInfer.h"
#include "reluKernel.h" // 修改

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Relu plugin specific constants
namespace { // 修改
    static const char* RELU_PLUGIN_VERSION{"1"};
    static const char* RELU_PLUGIN_NAME{"ReluPlugin"};
}

/**********************修改************************************/
// Static class fields initialization
PluginFieldCollection ReluPluginCreator::mFC{};
std::vector<PluginField> ReluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ReluPluginCreator);
/**********************************************************/

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

/**********************修改************************************/
ReluPlugin::ReluPlugin(const std::string name)
    : mLayerName(name)
{
}
/**********************************************************/

/**********************修改************************************/
ReluPlugin::ReluPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    // const char *d = static_cast<const char *>(data);
    // const char *a = d;

    // mClipMin = readFromBuffer<float>(d);
    // mClipMax = readFromBuffer<float>(d);

    // assert(d == (a + length));
}
/**********************************************************/

/**********************修改************************************/
const char* ReluPlugin::getPluginType() const
{
    return RELU_PLUGIN_NAME;
}

const char* ReluPlugin::getPluginVersion() const
{
    return RELU_PLUGIN_VERSION;
}

int ReluPlugin::getNbOutputs() const
{
    return 1;
}

Dims ReluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;
}
/**********************************************************/

int ReluPlugin::initialize()
{
    return 0;
}

int ReluPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = reluInference(stream, mInputVolume * batchSize, inputs[0], output);

    return status;
}

size_t ReluPlugin::getSerializationSize() const
{
    //return 2 * sizeof(float);
    return 0 * sizeof(float);
}

void ReluPlugin::serialize(void* buffer) const 
{
    // char *d = static_cast<char *>(buffer);
    // const char *a = d;

    // writeToBuffer(d, mClipMin);
    // writeToBuffer(d, mClipMax);

    // assert(d == a + getSerializationSize());
}

void ReluPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ReluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void ReluPlugin::terminate() {}

void ReluPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* ReluPlugin::clone() const
{
    return new ReluPlugin(mLayerName);
    // return new ClipPlugin(mLayerName, mClipMin, mClipMax);
}

void ReluPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* ReluPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

ReluPluginCreator::ReluPluginCreator()
{
    // Describe ReluPlugin's required PluginField arguments
    // mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    // mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    // mFC.nbFields = mPluginAttributes.size();
    // mFC.fields = mPluginAttributes.data();
}

const char* ReluPluginCreator::getPluginName() const
{
    return RELU_PLUGIN_NAME;
}

const char* ReluPluginCreator::getPluginVersion() const
{
    return RELU_PLUGIN_VERSION;
}

const PluginFieldCollection* ReluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* ReluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    // float clipMin, clipMax;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    // assert(fc->nbFields == 2);
    // for (int i = 0; i < fc->nbFields; i++){
    //     if (strcmp(fields[i].name, "clipMin") == 0) {
    //         assert(fields[i].type == PluginFieldType::kFLOAT32);
    //         clipMin = *(static_cast<const float*>(fields[i].data));
    //     } else if (strcmp(fields[i].name, "clipMax") == 0) {
    //         assert(fields[i].type == PluginFieldType::kFLOAT32);
    //         clipMax = *(static_cast<const float*>(fields[i].data));
    //     }
    // }
    return new ReluPlugin(name);
}


IPluginV2* ReluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ReluPlugin::destroy()
    return new ReluPlugin(name, serialData, serialLength);
}

void ReluPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* ReluPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}