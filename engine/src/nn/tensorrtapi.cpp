/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: TensorrtAPI.cpp
 * Created on 05.02.2020
 * @author: queensgambit
 */

#ifdef TENSORRT
#include "tensorrtapi.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "EntropyCalibrator.h"
#include "constants.h"
#include "../util/communication.h"

TensorrtAPI::TensorrtAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, const string& strPrecision):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true)
{
    // select the requested device
    cudaSetDevice(deviceID);
    // in ONNX, the model architecture and parameters are in the same file
    modelFilePath = modelDirectory + "model-bsize-" + to_string(batchSize) + ".onnx";
    enginePath = modelDirectory + "model-bsize" + to_string(batchSize) + "-";

    if (strPrecision == "float32") {
        precision = float32;
        enginePath += "fp32";
    }
    else if (strPrecision == "float16") {
        precision = float16;
        enginePath += "fp16";
    }
    else if (strPrecision == "int8") {
        precision = int8;
        enginePath += "int8";
    }
    else {
        info_string("Fallback to float32. Invalid precision type given:", precision);
        precision = float32;
        enginePath += "fp32";
    }
    enginePath += ".trt";
    gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);

    load_model();
    check_if_policy_map();
    bind_executor();
}

TensorrtAPI::~TensorrtAPI()
{
    for (auto memory : deviceMemory) {
        CHECK(cudaFree(memory));
    }
    CHECK(cudaStreamDestroy(stream));
}

void TensorrtAPI::load_model()
{
    // load an engine from file or build an engine from the ONNX network
    engine = shared_ptr<nvinfer1::ICudaEngine>(get_cuda_engine(), samplesCommon::InferDeleter());
    idxInput = engine->getBindingIndex("data");
#ifdef MODE_CRAZYHOUSE
    idxValueOutput = engine->getBindingIndex("value_tanh0");
    idxPolicyOutput = engine->getBindingIndex("policy_softmax");
#else
    idxValueOutput = engine->getBindingIndex("value_out");
    idxPolicyOutput = engine->getBindingIndex("policy_softmax");
#endif
}

void TensorrtAPI::load_parameters()
{
    // do nothing
}

void TensorrtAPI::bind_executor()
{
    // create an exectution context for applying inference
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    // create buffers object with respect to the engine and batch size
    CHECK(cudaStreamCreate(&stream));
    memorySizes[idxInput] = batchSize * NB_VALUES_TOTAL * sizeof(float);
    memorySizes[idxValueOutput] = batchSize * sizeof(float);
    memorySizes[idxPolicyOutput] = policyOutputLength * sizeof(float);
    CHECK(cudaMalloc(&deviceMemory[idxInput], memorySizes[idxInput]));
    CHECK(cudaMalloc(&deviceMemory[idxValueOutput], memorySizes[idxValueOutput]));
    CHECK(cudaMalloc(&deviceMemory[idxPolicyOutput], memorySizes[idxPolicyOutput]));
}

void TensorrtAPI::check_if_policy_map()
{
    if (policyOutputDims.d[1] != NB_LABELS) {
        isPolicyMap = true;
        policyOutputLength = NB_LABELS_POLICY_MAP * batchSize;
    }
}

void TensorrtAPI::predict(float* inputPlanes, float* valueOutput, float* probOutputs)
{
    // copy input planes from host to device
    CHECK(cudaMemcpyAsync(deviceMemory[idxInput], inputPlanes, memorySizes[idxInput],
                          cudaMemcpyHostToDevice, stream));

    // run inference for given data
    context->enqueueV2(deviceMemory, stream, nullptr);

    // copy output from device back to host
    CHECK(cudaMemcpyAsync(valueOutput, deviceMemory[idxValueOutput],
                          memorySizes[idxValueOutput], cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(probOutputs, deviceMemory[idxPolicyOutput],
                          memorySizes[idxPolicyOutput], cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
}

ICudaEngine* TensorrtAPI::create_cuda_engine_from_onnx()
{
    info_string("build TensorRT engine");
    // create an engine builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    builder->setMaxBatchSize(int(batchSize));

    // create an ONNX network object
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    SampleUniquePtr<nvinfer1::IBuilderConfig> config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    unique_ptr<IInt8Calibrator> calibrator;
    unique_ptr<ChessBatchStream> calibrationStream;
    set_config_settings(config, network, 1_GiB, calibrator, calibrationStream);

    // conversion of ONNX model to TensorRT
    // parse the ONNX model file along with logger object for reporting info
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "failed to parse onnx file");
        exit(EXIT_FAILURE);
        return nullptr;
    }
    configure_network(network);

    // build an engine from the TensorRT network with a given configuration struct
    return builder->buildEngineWithConfig(*network, *config);
}

ICudaEngine* TensorrtAPI::get_cuda_engine() {
    ICudaEngine* engine{nullptr};

    // try to read an engine from file
    size_t bufferSize;
    const char* buffer = read_buffer(enginePath, bufferSize);
    if (buffer) {
        info_string("deserialize engine:", enginePath);
        unique_ptr<IRuntime, samplesCommon::InferDeleter> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer, bufferSize, nullptr);
    }

    if (!engine) {
        // fallback: Create engine from scratch
        engine = create_cuda_engine_from_onnx();

        if (engine) {
            info_string("serialize engine:", enginePath);
            // serialized engines are not portable across platforms or TensorRT versions
            // engines are specific to the exact GPU model they were built on
            IHostMemory *serializedModel = engine->serialize();
            unique_ptr<IHostMemory, samplesCommon::InferDeleter> enginePlan{engine->serialize()};
            // export engine for future uses
            // write engine to file
            write_buffer(enginePlan->data(), enginePlan->size(), enginePath);
            serializedModel->destroy();
        }
    }
    return engine;
}

void TensorrtAPI::set_config_settings(SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                         SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                         size_t maxWorkspace, unique_ptr<IInt8Calibrator>& calibrator,
                         unique_ptr<ChessBatchStream>& calibrationStream)
{
    config->setMaxWorkspaceSize(maxWorkspace);
    switch (precision) {
    case float32:
        // default: do nothing
        break;
    case float16:
        config->setFlag(BuilderFlag::kFP16);
        break;
    case int8:
        config->setFlag(BuilderFlag::kINT8);
        info_string("run INT8 quantization calibration");
        calibrationStream.reset(new ChessBatchStream(1, 104));
        calibrator.reset(new Int8EntropyCalibrator2<ChessBatchStream>(*calibrationStream.get(), 0, "model", "data"));
        config->setInt8Calibrator(calibrator.get());
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
        break;
    }
}

void TensorrtAPI::configure_network(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
{
    inputDims = network->getInput(0)->getDimensions();
    valueOutputDims = network->getOutput(0)->getDimensions();
    policyOutputDims = network->getOutput(1)->getDimensions();
    // add a softmax layer to the ONNX model
    ISoftMaxLayer* softmaxLayer = network->addSoftMax(*network->getOutput(1));
    // set the softmax axis to 1
    softmaxLayer->setAxes(1 << 1);

    // set precision of the first and last layers to float32
    fix_layer_precision(network->getLayer(0), nvinfer1::DataType::kFLOAT);
    fix_layer_precision(softmaxLayer, nvinfer1::DataType::kFLOAT);
    fix_layer_precision(network->getLayer(idxValueOutput), nvinfer1::DataType::kFLOAT);

    // set the softmax layer output as the new output
    network->unmarkOutput(*network->getOutput(1));
    network->markOutput(*softmaxLayer->getOutput(0));
    softmaxLayer->getOutput(0)->setName("policy_softmax");

    info_string("inputDims:", inputDims);
    info_string("valueOutputDims:", valueOutputDims);
    info_string("policyOutputDims:", policyOutputDims);
}

void write_buffer(void* buffer, size_t bufferSize, const string& filePath) {
    std::ofstream outputFile(filePath,std::ofstream::binary);
    outputFile.write((const char*) buffer, bufferSize);
    outputFile.close();
}

const char* read_buffer(const string& filePath, size_t& bufferSize) {
    std::ifstream inputFile (filePath,std::ifstream::binary);
    if (!inputFile) {
        info_string("no engine file found");
        return nullptr;
    }

    // get file size
    inputFile.seekg(0, inputFile.end);
    bufferSize = inputFile.tellg();
    inputFile.seekg(0);

    // allocate memory for the file content
    char* buffer = new char[bufferSize];

    // read content of the input file
    inputFile.read(buffer, bufferSize);
    if (!inputFile) {
        info_string("error reading file buffer:", filePath);
        return nullptr;
    }

    inputFile.close();
    return buffer;
}

void fix_layer_precision(ILayer *layer, nvinfer1::DataType dataType)
{
    layer->setPrecision(dataType);
    for (int idx = 0; idx < layer->getNbOutputs(); ++idx) {
        layer->setOutputType(idx, dataType);
    }
}

#endif
