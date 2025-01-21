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
#ifdef SF_DEPENDENCY
#include "environments/chess_related/chessbatchstream.h"
#endif

using namespace sample;

TensorrtAPI::TensorrtAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, const string& strPrecision):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true),
    idxInput(nnDesign.inputIdx),
    idxValueOutput(nnDesign.valueOutputIdx + nnDesign.nbInputs),
    idxPolicyOutput(nnDesign.policyOutputIdx + nnDesign.nbInputs),
    idxAuxiliaryOutput(nnDesign.auxiliaryOutputIdx + nnDesign.nbInputs),
    precision(str_to_precision(strPrecision)),
    generatedTrtFromONNX(false)
{
    // select the requested device
    cudaSetDevice(deviceID);
    // in ONNX, the model architecture and parameters are in the same file
    modelName = get_onnx_model_name(modelDir, batchSize);

    modelFilePath = modelDir + modelName;
    info_string("onnx file:", modelFilePath);
    trtFilePath = generate_trt_file_path(modelDir, batchSize, precision, deviceID);
    gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);

    initialize();
}

TensorrtAPI::~TensorrtAPI()
{
    CHECK(cudaFree(deviceMemory[idxInput]));
    CHECK(cudaFree(deviceMemory[idxValueOutput]));
    CHECK(cudaFree(deviceMemory[idxPolicyOutput]));
#ifdef DYNAMIC_NN_ARCH
    if (nnDesign.hasAuxiliaryOutputs) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
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

#ifndef TENSORRT10
bool TensorrtAPI::retrieve_indices_by_name(bool verbose)
{
    idxInput = engine->getBindingIndex(nnDesign.inputLayerName.c_str());
    if (idxInput == -1) {
        info_string_important("Layer name '" + nnDesign.inputLayerName + "' not found.");
        return false;
    }
    idxValueOutput = engine->getBindingIndex(nnDesign.valueOutputName.c_str());
    if (idxValueOutput == -1) {
        info_string_important("Layer name '" + nnDesign.valueOutputName + "' not found.");
        return false;
    }
    idxPolicyOutput = engine->getBindingIndex(nnDesign.policySoftmaxOutputName.c_str());
    if (idxPolicyOutput == -1) {
        info_string_important("Layer name '" + nnDesign.policySoftmaxOutputName + "' not found.");
        return false;
    }
    if (nnDesign.hasAuxiliaryOutputs) {
        idxAuxiliaryOutput = engine->getBindingIndex(nnDesign.auxiliaryOutputName.c_str());
        if (idxAuxiliaryOutput == -1) {
            info_string_important("Layer name '" + nnDesign.auxiliaryOutputName + "' not found.");
            return false;
        }
    }
    if (verbose) {
        info_string("Found 'idxInput' at index", idxInput);
        info_string("Found 'idxValueOutput' at index", idxValueOutput);
        info_string("Found 'idxPolicyOutput' at index", idxPolicyOutput);
        if (nnDesign.hasAuxiliaryOutputs) {
            info_string("Found 'idxAuxiliaryOutput' at index", idxAuxiliaryOutput);
        }
    }
    return true;
}
#endif

void TensorrtAPI::init_nn_design()
{
#ifndef TENSORRT10
    nnDesign.hasAuxiliaryOutputs = engine->getNbBindings() > 3;
    if (!retrieve_indices_by_name(generatedTrtFromONNX)) {
        info_string_important("Fallback to default indices.");
        idxInput = nnDesign.inputIdx;
        idxValueOutput = nnDesign.valueOutputIdx + nnDesign.nbInputs;
        idxPolicyOutput = nnDesign.policyOutputIdx + nnDesign.nbInputs;
        idxAuxiliaryOutput = nnDesign.auxiliaryOutputIdx + nnDesign.nbInputs;
    }
    set_shape(nnDesign.inputShape, engine->getBindingDimensions(idxInput));

    set_shape(nnDesign.valueOutputShape, engine->getBindingDimensions(idxValueOutput));
    set_shape(nnDesign.policyOutputShape, engine->getBindingDimensions(idxPolicyOutput));

    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, engine->getBindingDimensions(idxAuxiliaryOutput));
    }
#else
    set_shape(nnDesign.inputShape, engine->getTensorShape(nnDesign.inputLayerName.c_str()));
    set_shape(nnDesign.valueOutputShape, engine->getTensorShape(nnDesign.valueOutputName.c_str()));
    set_shape(nnDesign.policyOutputShape, engine->getTensorShape(nnDesign.policySoftmaxOutputName.c_str()));
    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, engine->getTensorShape(nnDesign.auxiliaryOutputName.c_str()));
    }
#endif
    // make sure that the first dimension is the batch size, otherwise '-1' could cause problems
    nnDesign.inputShape.v[0] = batchSize;
    nnDesign.isPolicyMap = unsigned(nnDesign.policyOutputShape.v[1]) != StateConstants::NB_LABELS();
}

void TensorrtAPI::bind_executor()
{
    // create an exectution context for applying inference
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    Dims inputDims;
    set_dims(inputDims, nnDesign.inputShape);
#ifdef TENSORRT10
    context->setInputShape(nnDesign.inputLayerName.c_str(), inputDims);
#else
    context->setBindingDimensions(0, inputDims);
#endif

    // create buffers object with respect to the engine and batch size
    CHECK(cudaStreamCreate(&stream));
#ifdef DYNAMIC_NN_ARCH
    memorySizes[idxInput] = batchSize * get_nb_input_values_total() * sizeof(float);
#else
    memorySizes[idxInput] = batchSize * StateConstants::NB_VALUES_TOTAL() * sizeof(float);
#endif
    memorySizes[idxValueOutput] = batchSize * sizeof(float);
    memorySizes[idxPolicyOutput] = batchSize * get_nb_policy_values() * sizeof(float);
#ifdef DYNAMIC_NN_ARCH
    if (nnDesign.hasAuxiliaryOutputs) {
        memorySizes[idxAuxiliaryOutput] = batchSize * get_nb_auxiliary_outputs() * sizeof (float);
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
        memorySizes[idxAuxiliaryOutput] = batchSize * StateConstants::NB_AUXILIARY_OUTPUTS() * sizeof (float);
#endif
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

#ifdef TENSORRT10
    context->setTensorAddress(nnDesign.inputLayerName.c_str(), deviceMemory[idxInput]);
    context->setTensorAddress(nnDesign.valueOutputName.c_str(), deviceMemory[idxValueOutput]);
    context->setTensorAddress(nnDesign.policySoftmaxOutputName.c_str(), deviceMemory[idxPolicyOutput]);
#ifdef DYNAMIC_NN_ARCH
    if (has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
        context->setTensorAddress(nnDesign.auxiliaryOutputName.c_str(), deviceMemory[idxAuxiliaryOutput]);
    }
#endif

    // run inference for given data
#ifdef TENSORRT10
    context->enqueueV3(stream);
#else
    context->enqueueV2(deviceMemory, stream, nullptr);
#endif

    // copy output from device back to host
    CHECK(cudaMemcpyAsync(valueOutput, deviceMemory[idxValueOutput],
                          memorySizes[idxValueOutput], cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(probOutputs, deviceMemory[idxPolicyOutput],
                          memorySizes[idxPolicyOutput], cudaMemcpyDeviceToHost, stream));
#ifdef DYNAMIC_NN_ARCH
    if (has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
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
    SampleUniquePtr<IBuilder> builder = SampleUniquePtr<IBuilder>(createInferBuilder(gLogger.getTRTLogger()));
#ifndef TENSORRT10
    builder->setMaxBatchSize(int(batchSize));
#endif

    // create an ONNX network object
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // conversion of ONNX model to TensorRT
    // parse the ONNX model file along with logger object for reporting info
    SampleUniquePtr<nvonnxparser::IParser> parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser->parseFromFile(modelFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "failed to parse onnx file");
        for (int32_t idx = 0; idx < parser->getNbErrors(); ++idx) {
            std::cout << parser->getError(idx)->desc() << std::endl;
        }
        exit(EXIT_FAILURE);
        return nullptr;
    }
    configure_network(network);
    
    SampleUniquePtr<nvinfer1::IBuilderConfig> config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    unique_ptr<IInt8Calibrator> calibrator;
    unique_ptr<IBatchStream> calibrationStream;
    set_config_settings(config, calibrator, calibrationStream);

    IOptimizationProfile* profile = builder->createOptimizationProfile();

    Dims inputDims = network->getInput(0)->getDimensions();
    inputDims.d[0] = batchSize;
    profile->setDimensions(nnDesign.inputLayerName.c_str(), OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(nnDesign.inputLayerName.c_str(), OptProfileSelector::kOPT, inputDims);
    profile->setDimensions(nnDesign.inputLayerName.c_str(), OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

#ifdef TENSORRT10
    nnDesign.hasAuxiliaryOutputs = network->getNbOutputs() > 2;
#endif

    // build an engine from the TensorRT network with a given configuration struct
#ifdef TENSORRT7
    return builder->buildEngineWithConfig(*network, *config);
#else
    SampleUniquePtr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    runtime = SampleUniquePtr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));

    // build an engine from the serialized model
    return runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());;
#endif
}

ICudaEngine* TensorrtAPI::get_cuda_engine() {
    ICudaEngine* engine{nullptr};

    // try to read an engine from file
    size_t bufferSize;
    const char* buffer = read_buffer(trtFilePath, bufferSize);
    if (buffer) {
        info_string("deserialize engine:", trtFilePath);
        runtime = unique_ptr<IRuntime, samplesCommon::InferDeleter>{createInferRuntime(gLogger)};
#ifdef TENSORRT7
        engine = runtime->deserializeCudaEngine(buffer, bufferSize, nullptr);
#else
        engine = runtime->deserializeCudaEngine(buffer, bufferSize);
#endif
    }

    if (!engine) {
        // fallback: Create engine from scratch
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        engine = create_cuda_engine_from_onnx();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        info_elapsed_time("Elapsed time for building TensorRT engine:", begin, end);
        generatedTrtFromONNX = true;

        if (engine) {
            info_string("serialize engine:", trtFilePath);
            // serialized engines are not portable across platforms or TensorRT versions
            // engines are specific to the exact GPU model they were built on
            unique_ptr<IHostMemory, samplesCommon::InferDeleter> enginePlan{engine->serialize()};
            // export engine for future uses
            // write engine to file
            write_buffer(enginePlan->data(), enginePlan->size(), trtFilePath);
        }
    }
    return engine;
}

void TensorrtAPI::set_config_settings(SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                      unique_ptr<IInt8Calibrator>& calibrator,
                                      unique_ptr<IBatchStream>& calibrationStream)
{
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
#if !defined(MODE_POMMERMAN) && !defined(MODE_OPEN_SPIEL) && !defined(MODE_XIANGQI) && !defined(MODE_STRATEGO) && !defined (MODE_BOARDGAMES)
        calibrator.reset(new Int8EntropyCalibrator2<ChessBatchStream>(*(dynamic_cast<ChessBatchStream*>(calibrationStream.get())), 0, "model", "data"));
#endif
        config->setInt8Calibrator(calibrator.get());
        // samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f); -> unavailable for TensorRT >= 8.2.0.6
        break;
    }
}

void TensorrtAPI::configure_network(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
{
    // add a softmax layer to the ONNX model
    int policyOutputIdx = -1;
    for (int idx = 0; idx < network->getNbOutputs(); ++idx) {
        if (string(network->getOutput(idx)->getName()) == nnDesign.policyOutputName) {
            policyOutputIdx = idx;
            break;
        }
    }
    if (policyOutputIdx == -1) {
        info_string("Did not find policy output with name '" + nnDesign.policyOutputName + "'");
        info_string("Setting policyOutputIdx to:", nnDesign.policyOutputIdx);
        policyOutputIdx = nnDesign.policyOutputIdx;
    }

    ISoftMaxLayer* softmaxLayer = network->addSoftMax(*network->getOutput(policyOutputIdx));
    // set the softmax axis to 1
    softmaxLayer->setAxes(1 << 1);

    // set precision of the first and last layers to float32
    // 0 is the input layer, 1 the value output and 2 the policy output layer
//    fix_layer_precision(network->getLayer(0), nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(softmaxLayer, nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(network->getLayer(1), nvinfer1::DataType::kFLOAT);
//    fix_layer_precision(network->getLayer(2), nvinfer1::DataType::kFLOAT);

    // set the softmax layer output as the new output
    network->unmarkOutput(*network->getOutput(policyOutputIdx));
    network->markOutput(*softmaxLayer->getOutput(0));
    softmaxLayer->getOutput(0)->setName(nnDesign.policySoftmaxOutputName.c_str());
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

void set_dims(Dims &dims, const nn_api::Shape &shape)
{
    dims.nbDims = shape.nbDims;
    for (int idx = 0; idx < shape.nbDims; ++idx) {
        dims.d[idx] = shape.v[idx];
    }
}

#endif
