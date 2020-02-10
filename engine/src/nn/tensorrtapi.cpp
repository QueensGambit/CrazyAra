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

TensorrtAPI::TensorrtAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, Precision precision):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true),
    precision(float32)
{
    // in ONNX, the model architecture and parameters are in the same file
    modelFilePath = "model.onnx";
    load_model();
    bind_executor();
}

void TensorrtAPI::load_model()
{
    // load an engine from file or build an engine from the ONNX network
    engine = shared_ptr<nvinfer1::ICudaEngine>(get_cuda_engine(), samplesCommon::InferDeleter());
}

void TensorrtAPI::bind_executor()
{
    // create an exectution context for applying inference
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
}

ICudaEngine* TensorrtAPI::create_cuda_engine_from_onnx() {

    // create an engine builder
    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(int(batchSize));

    // create an ONNX network object
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    inputDims = network->getInput(0)->getDimensions();
    outputDims = network->getOutput(0)->getDimensions();
    info_string("inputDims:", inputDims);
    info_string("outputDims:", outputDims);

    SampleUniquePtr<nvinfer1::IBuilderConfig> config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    set_config_settings(config, network, precision);

   // conversion of onnx model to tensorrt
   // parse the ONNX model file along with logger object for reporting info
   auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
   if (!parser->parseFromFile(modelFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
   {
         gLogger.log(nvinfer1::ILogger::Severity::kERROR, "failed to parse onnx file");
         exit(EXIT_FAILURE);
         return nullptr;
   }

   // show additional information about the network
   // parser->reportParsingInfo();

   // build an engine from the TensorRT network with a given configuration struct
   return builder->buildEngineWithConfig(*network, *config);
}

ICudaEngine* TensorrtAPI::get_cuda_engine() {

    // getBasename(modelFilePath)
    string modelBaseName = "model";
    string enginePath{modelBaseName + "_batch" + to_string(batchSize) + ".engine"};
    ICudaEngine* engine{nullptr};


//    TODO: Read engine from file
//    string buffer = readBuffer(enginePath);
//    if (buffer.size()) {
//        // try to deserialize engine
//        unique_ptr<IRuntime, Destroy> runtime{createInferRuntime(gLogger)};
//        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
//    }

    if (!engine) {
        // Fallback to creating engine from scratch
        engine = create_cuda_engine_from_onnx();

        if (engine) {
            // serialized engines are not portable across platforms or TensorRT versions
            // engines are specific to the exact GPU model they were built on
            IHostMemory *serializedModel = engine->serialize();
            //unique_ptr<IHostMemory, Destroy> enginePlan{engine->serialize()};
            // export engine for future uses
            //    TODO: Write engine to file
            // writeBuffer(enginePlan->data(), enginePlan->size(), enginePath);
            serializedModel->destroy();
        }
    }
    return engine;
}

void set_config_settings(SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                      Precision precision, size_t maxWorkspace)
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
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
        break;
    }
}

#endif
