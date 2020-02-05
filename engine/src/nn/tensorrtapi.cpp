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

TensorrtAPI::TensorrtAPI(int deviceID, unsigned int batchSize, const string &modelDirectory):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true)
{

    // CONVERSION OF ONNX MODEL TO TENSORRT
    // parse the ONNX model file along with logger object for reporting info
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(model_file, static_cast<int>(gLogger.getReportableSeverity())))
    {
          string msg("failed to parse onnx file");
          gLogger->log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
          exit(EXIT_FAILURE);
    }

    // show additional information about the network
    parser->reportParsingInfo();


    // BUILDING THE ENGINE
    // create an engine builder
    IBuilder* builder = createInferBuilder(gLogger);

    // build an engine from the TensorRT network
    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

    // RUN INFERENCE
    // create space for the action values
    IExecutionContext *context = engine->createExecutionContext();

    // extract the input and output buffer indices of the GPU memory
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
}

#endif
