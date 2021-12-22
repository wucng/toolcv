#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "parserOnnxConfig.h" // 解析 onnx文件
#include <fstream>
#include <map>
#include <chrono>
//#include <iostream>

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

static Logger gLogger;


struct SampleParams
{
    int INPUT_H{224};
    int INPUT_W{224};
    int OUTPUT_SIZE{1000};
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



// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config,SampleParams mParams)
{
    
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(mParams.INPUT_BLOB_NAME.c_str(), DataType::kFLOAT, Dims3{3, mParams.INPUT_H, mParams.INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(mParams.wtsFile);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{11, 11}, weightMap["features.0.weight"], weightMap["features.0.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5}, weightMap["features.3.weight"], weightMap["features.3.bias"]);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{2, 2});
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer* conv3 = network->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    assert(conv3);
    conv3->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    assert(conv4);
    conv4->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    IConvolutionLayer* conv5 = network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.10.weight"], weightMap["features.10.bias"]);
    assert(conv5);
    conv5->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    assert(relu5);
    IPoolingLayer* pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool3->getOutput(0), 4096, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    assert(fc1);

    IActivationLayer* relu6 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu6);

    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu6->getOutput(0), 4096, weightMap["classifier.4.weight"], weightMap["classifier.4.bias"]);
    assert(fc2);

    IActivationLayer* relu7 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu7);

    IFullyConnectedLayer* fc3 = network->addFullyConnected(*relu7->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);
    assert(fc3);

    fc3->getOutput(0)->setName(mParams.OUTPUT_BLOB_NAME.c_str());
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc3->getOutput(0));

    // Build engine
    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);//kFP16 = 0 kINT8 = 1
    // if(mParams.fp16) builder->setHalf2Mode(true); // 这个不起作用？？？
	// builder->setFp16Mode(true); builder->setInt8Mode(true); 
    builder->setMaxBatchSize(mParams.BATCH_SIZE);
    config->setMaxWorkspaceSize(mParams.MEM_SIZE);
	// builder->setMaxWorkspaceSize(1_GB);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

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
        data[i] = 1;

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
    // for (unsigned int i = 0; i < mParams.OUTPUT_SIZE; i++)
    // {
    //     std::cout << prob[i] << ", ";
    //     if (i % 10 == 0) std::cout << i / 10 << std::endl;
    // }

    // argmax 值
    float max_v = prob[0];
    int index = 0;
    for(int i = 1;i < mParams.OUTPUT_SIZE; i++)
    {
        if (max_v < prob[i]) {
            max_v = prob[i];
            index = i;
        }
    }

    std::cout << "argmax" << index << std::endl;
    // 输出前10个
    for(int i = 0;i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;

    // 输出最后10个
    for(int i = mParams.OUTPUT_SIZE-10;i < mParams.OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;

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
