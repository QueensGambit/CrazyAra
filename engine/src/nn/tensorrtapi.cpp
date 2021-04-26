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

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "EntropyCalibrator.h"
#include "stateobj.h"
#include "../util/communication.h"
#if !defined(MODE_POMMERMAN) && !defined(MODE_XIANGQI)
#include "environments/chess_related/chessbatchstream.h"
#endif

using namespace sample;

TensorrtAPI::TensorrtAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, const string& strPrecision):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true),
    precision(str_to_precision(strPrecision))
{
    // select the requested device
    cudaSetDevice(deviceID);
    // in ONNX, the model architecture and parameters are in the same file
    modelName = get_file_ending_with(modelDir, "-bsize-" + to_string(batchSize) + ".onnx");
    modelFilePath = modelDir + modelName;
    info_string("onnx file:", modelFilePath);
    trtFilePath = generate_trt_file_path(modelDir, batchSize, precision, deviceID);
    gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);

    load_model();
    init_nn_design();
    bind_executor();
}

TensorrtAPI::~TensorrtAPI()
{
    CHECK(cudaFree(deviceMemory[idxInput]));
    CHECK(cudaFree(deviceMemory[idxValueOutput]));
    CHECK(cudaFree(deviceMemory[idxPolicyOutput]));
    if (nnDesign.hasAuxiliaryOutputs) {
        CHECK(cudaFree(deviceMemory[idxAuxiliaryOutput]));
    }
    CHECK(cudaStreamDestroy(stream));
}

void TensorrtAPI::load_model()
{
    // load an engine from file or build an engine from the ONNX network
    engine = shared_ptr<nvinfer1::ICudaEngine>(get_cuda_engine(), samplesCommon::InferDeleter());
}

void TensorrtAPI::load_parameters()
{
    // do nothing
}

void TensorrtAPI:: init_nn_design()
{
    set_shape(nnDesign.inputShape, engine->getBindingDimensions(idxInput));
    set_shape(nnDesign.valueOutputShape, engine->getBindingDimensions(idxValueOutput));
    set_shape(nnDesign.policyOutputShape, engine->getBindingDimensions(idxPolicyOutput));
    nnDesign.hasAuxiliaryOutputs = engine->getNbBindings() > 3;
    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, engine->getBindingDimensions(idxAuxiliaryOutput));
    }
    nnDesign.isPolicyMap = unsigned(nnDesign.policyOutputShape.v[1]) != StateConstants::NB_LABELS();
}

void TensorrtAPI::bind_executor()
{
    // create an exectution context for applying inference
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    // create buffers object with respect to the engine and batch size
    CHECK(cudaStreamCreate(&stream));
    memorySizes[idxInput] = batchSize * get_nb_input_values_total() * sizeof(float);
    memorySizes[idxValueOutput] = batchSize * sizeof(float);
    memorySizes[idxPolicyOutput] = get_policy_output_length() * sizeof(float);
    if (nnDesign.hasAuxiliaryOutputs) {
        memorySizes[idxAuxiliaryOutput] = batchSize * get_nb_auxiliary_outputs() * sizeof (float);
        CHECK(cudaMalloc(&deviceMemory[idxAuxiliaryOutput], memorySizes[idxAuxiliaryOutput]));
    }
    CHECK(cudaMalloc(&deviceMemory[idxInput], memorySizes[idxInput]));
    CHECK(cudaMalloc(&deviceMemory[idxValueOutput], memorySizes[idxValueOutput]));
    CHECK(cudaMalloc(&deviceMemory[idxPolicyOutput], memorySizes[idxPolicyOutput]));
}

void TensorrtAPI::predict(float* inputPlanes, float* valueOutput, float* probOutputs, float* auxiliaryOutputs)
{
    // select the requested device
    cudaSetDevice(deviceID);
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
    if (has_auxiliary_outputs()) {
        CHECK(cudaMemcpyAsync(auxiliaryOutputs, deviceMemory[idxAuxiliaryOutput],
                              memorySizes[idxAuxiliaryOutput], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

ICudaEngine* TensorrtAPI::create_cuda_engine_from_onnx()
{
    info_string("Building TensorRT engine...");
    info_string("This may take a few minutes...");
    // create an engine builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    builder->setMaxBatchSize(int(batchSize));

    // create an ONNX network object
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    SampleUniquePtr<nvinfer1::IBuilderConfig> config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    unique_ptr<IInt8Calibrator> calibrator;
    unique_ptr<IBatchStream> calibrationStream;
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
    const char* buffer = read_buffer(trtFilePath, bufferSize);
    if (buffer) {
        info_string("deserialize engine:", trtFilePath);
        unique_ptr<IRuntime, samplesCommon::InferDeleter> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer, bufferSize, nullptr);
    }

    if (!engine) {
        // fallback: Create engine from scratch
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        engine = create_cuda_engine_from_onnx();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        info_elapsed_time("Elapsed time for building TensorRT engine:", begin, end);

        if (engine) {
            info_string("serialize engine:", trtFilePath);
            // serialized engines are not portable across platforms or TensorRT versions
            // engines are specific to the exact GPU model they were built on
            IHostMemory *serializedModel = engine->serialize();
            unique_ptr<IHostMemory, samplesCommon::InferDeleter> enginePlan{engine->serialize()};
            // export engine for future uses
            // write engine to file
            write_buffer(enginePlan->data(), enginePlan->size(), trtFilePath);
            serializedModel->destroy();
        }
    }
    return engine;
}

void TensorrtAPI::set_config_settings(SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                      SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                      size_t maxWorkspace, unique_ptr<IInt8Calibrator>& calibrator,
                                      unique_ptr<IBatchStream>& calibrationStream)
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
#ifdef MODE_CHESS
        calibrationStream.reset(new ChessBatchStream(1, 104));
#elif defined MODE_CRAZYHOUSE
        calibrationStream.reset(new ChessBatchStream(1, 232));
#endif
#if !defined(MODE_POMMERMAN) && !defined(MODE_OPEN_SPIEL) && !defined(MODE_XIANGQI)
        calibrator.reset(new Int8EntropyCalibrator2<ChessBatchStream>(*(dynamic_cast<ChessBatchStream*>(calibrationStream.get())), 0, "model", "data"));
#endif
        config->setInt8Calibrator(calibrator.get());
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
        break;
    }
}

void TensorrtAPI::configure_network(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
{
    // add a softmax layer to the ONNX model
    ISoftMaxLayer* softmaxLayer = network->addSoftMax(*network->getOutput(1));
    // set the softmax axis to 1
    softmaxLayer->setAxes(1 << 1);

    // set precision of the first and last layers to float32
    // 0 is the input layer, 1 the value output and 2 the policy output layer
//    fix_layer_precision(network->getLayer(0), nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(softmaxLayer, nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(network->getLayer(1), nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(network->getLayer(2), nvinfer1::DataType::kFLOAT);

    // set the softmax layer output as the new output
    network->unmarkOutput(*network->getOutput(nnDesign.policyOutputIdx));
    network->markOutput(*softmaxLayer->getOutput(0));
    softmaxLayer->getOutput(0)->setName("policy_softmax");
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

string generate_trt_file_path(const string &modelDirectory, unsigned int batchSize, Precision precision, int deviceID)
{
    return modelDirectory + "model-bsize" + to_string(batchSize) + "-" +
            precision_to_str(precision)+ "-" + to_string(deviceID) + ".trt";
}

Precision str_to_precision(const string &strPrecision)
{
    if (strPrecision == "float32" || strPrecision == "fp32") {
        return float32;
    }
    else if (strPrecision == "float16" || strPrecision == "fp16") {
        return float16;
    }
    else if (strPrecision == "int8") {
        return int8;
    }
    info_string("Fallback to float32. Invalid precision type given:", strPrecision);
    return  float32;
}

string precision_to_str(Precision precision)
{
    switch (precision) {
    case float32:
        return "fp32";
    case float16:
        return "fp16";
    case int8:
        return  "int8";
    }
    return "fp32";
}

void set_shape(nn_api::Shape &shape, const Dims &dims)
{
    shape.nbDims = dims.nbDims;
    for (int idx = 0; idx < shape.nbDims; ++idx) {
        shape.v[idx] = dims.d[idx];
    }
}

#endif
