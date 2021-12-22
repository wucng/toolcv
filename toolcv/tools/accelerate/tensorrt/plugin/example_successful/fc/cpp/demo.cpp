// https://blog.csdn.net/han2529386161/article/details/102723545
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "parserOnnxConfig.h" // 解析 onnx文件
#include <fstream>
#include <map>
#include <chrono>
#include <iostream>

#include "relu.h" 


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;
using namespace nvonnxparser;


// -------------------------------------执行注册---------------------------------------------------------
// REGISTER_TENSORRT_PLUGIN(ClipPluginCreator); // relu.cu 已经注册过了
// -----------------------------------------------------------------------------------------------------------

static Logger gLogger;

// initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

struct SampleParams
{
    int INPUT_H{1};
    int INPUT_W{1};
    int OUTPUT_SIZE{3};
    std::string INPUT_BLOB_NAME{"input"};
    std::string OUTPUT_BLOB_NAME{"output"};
    DataType dt{DataType::kFLOAT}; // DataType::kFLOAT kHALF kINT8

    int MEM_SIZE{1<<28}; //内存大小
    int BATCH_SIZE{1};                  //!< Number of inputs in a batch
    // int dlaCore{-1};                   //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    // std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    // std::vector<std::string> inputTensorNames;
    // std::vector<std::string> outputTensorNames;
    std::string onnxFile; //!<  ONNX file of a network
    std::string wtsFile; //!<  wts file of a network
    std::string engineFile; // 输出的engine文件
};


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config,SampleParams mParams)
{

    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* inputLayer = network->addInput(mParams.INPUT_BLOB_NAME.c_str(), DataType::kFLOAT, Dims3{3, mParams.INPUT_H, mParams.INPUT_W});
    assert(inputLayer);

    // ------------------------使用自定义plugin 实现relu------------------------------------------------------------------------
    float weight[15]={1.0f,1.0f,1.0f,1.0f,1.0f,
                      1.0f,1.0f,1.0f,1.0f,1.0f,
                      1.0f,1.0f,1.0f,1.0f,1.0f};
    float bias[5] = {1.0f,1.0f,1.0f,1.0f,1.0f};
    // for(auto item : weight)
    //     std::cout << item << ",";
    // std::cout << std::endl;
    
    auto creator = getPluginRegistry()->getPluginCreator("FC_TRT_2", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    PluginField* fields = (nvinfer1::PluginField*)pluginData->fields;
    
    // pluginData->nbFields=2; // clipMin + clipMax
    fields[0].name = "weight";
    fields[0].length = 15;
    fields[0].type = nvinfer1::PluginFieldType::kFLOAT32;
    fields[0].data = weight;
    

    fields[1].name = "bias";
    fields[1].length = 5;
    fields[1].type = nvinfer1::PluginFieldType::kFLOAT32;
    fields[1].data = bias;
    

    IPluginV2 *pluginObj = creator->createPlugin("FC_TRT_2", pluginData);
    ITensor* inputTensors[] = {inputLayer};
    auto pluginLayer = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
    // ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    // auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    // ----------------------------------------------------------------------------------------------------------------------------------------
    assert(pluginLayer);

    pluginLayer->getOutput(0)->setName(mParams.OUTPUT_BLOB_NAME.c_str());
    std::cout << "set name out" << std::endl;
    network->markOutput(*pluginLayer->getOutput(0));
    
    // Build engine
    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);//kFP16 = 0 kINT8 = 1
    // if(mParams.fp16) builder->setHalf2Mode(true); // 这个不起作用？？？
    builder->setMaxBatchSize(mParams.BATCH_SIZE);
    config->setMaxWorkspaceSize(mParams.MEM_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    // for (auto& mem : weightMap)
    // {
    //     free((void*) (mem.second.values));
    // }

    return engine;
}

void WtsToModel(IHostMemory** modelStream,SampleParams mParams)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(builder, config, mParams);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

/*---------------------实现 onnx模型转 trt---------------------*/

void OnnxToModel(IHostMemory** modelStream,SampleParams mParams)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    IBuilderConfig* config = builder->createBuilderConfig();

    /*实现 解析 onnx文件生成engine */
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    auto parsed = parser->parseFromFile(mParams.onnxFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()));


    // constructNetwork
    builder->setMaxBatchSize(mParams.BATCH_SIZE);
    config->setMaxWorkspaceSize(mParams.MEM_SIZE);
    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);
    // if(mParams.fp16) builder->setHalf2Mode(true); // 这个不起作用？？？

    // if (mParams.int8)
    // {
    //     config->setFlag(BuilderFlag::kINT8);
    //     samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    // }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // Build engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    /*
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);
    */

    // Don't need the network any more
    network->destroy();
    parser->destroy();
    /*----------------------------------------------*/
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

/*-----------------------------------------------------------*/

void transform2trt(SampleParams mParams,std::string mode)
{
    char *trtModelStream{nullptr};
    IHostMemory* modelStream{nullptr};

    if (mode=="onnx")
    {
        OnnxToModel(&modelStream,mParams);
    }
    else
    {
        WtsToModel(&modelStream,mParams);
    }

    assert(modelStream != nullptr);

    std::ofstream p(mParams.engineFile, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
}


void doInference(IExecutionContext& context, float* input, float* output, SampleParams mParams)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(mParams.INPUT_BLOB_NAME.c_str());
    const int outputIndex = engine.getBindingIndex(mParams.OUTPUT_BLOB_NAME.c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], mParams.BATCH_SIZE * 3 * mParams.INPUT_H * mParams.INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], mParams.BATCH_SIZE * mParams.OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, mParams.BATCH_SIZE * 3 * mParams.INPUT_H * mParams.INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(mParams.BATCH_SIZE, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], mParams.BATCH_SIZE * mParams.OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void Inference(SampleParams mParams){
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(mParams.engineFile, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // Subtract mean from image
    float data[3 * mParams.INPUT_H * mParams.INPUT_W];
    for (int i = 0; i < 3 * mParams.INPUT_H * mParams.INPUT_W; i++)
        data[i] = 2.0f;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[mParams.OUTPUT_SIZE];
    // for (int i = 0; i < 100; i++) {
    //     auto start = std::chrono::system_clock::now();
    //     doInference(*context, data, prob, mParams);
    //     auto end = std::chrono::system_clock::now();
    //     std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // }

    doInference(*context, data, prob, mParams);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";

    for(int i = 0;i < mParams.OUTPUT_SIZE; i++)
    {
        std::cout<<prob[i]<<",";
    }
    std::cout<<std::endl;

}


int main(int argc, char** argv)
{
    std::cout <<"./alexnet -s float32 onnx" << std::endl;
    std::cout <<"./alexnet -d float32 onnx" << std::endl;

    if (argc < 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./alexnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./alexnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    SampleParams mParams;
    mParams.onnxFile = "../model.onnx";
    mParams.wtsFile = "../model.wts";
    mParams.engineFile = "../model.trt";

    if (std::string(argv[2]) == "float32")
    {
        mParams.fp16 = false;
        mParams.dt = DataType::kFLOAT;
    }
    else{
        mParams.fp16 = true;
        mParams.dt = DataType::kHALF;
    }

    if (std::string(argv[1]) == "-s") {
        transform2trt(mParams,std::string(argv[3]));
        return 1;
    } 
    else if (std::string(argv[1]) == "-d") 
    {
        Inference(mParams);
        return 0;
    } 
    else 
    {
        return -1;
    }
}
