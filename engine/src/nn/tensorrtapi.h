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
 * @file: tensorrtapi.h
 * Created on 05.02.2020
 * @author: queensgambit
 *
 * Interface for running inference with the ONNX-TensorRT GPU backend.
 * This class is built based on the MNIST TensorRT example:
 * https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleOnnxMNIST
 *
 * and the TensorRT inference documentation:
 * https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c
 */

#ifndef TENSORRTAPI_H
#define TENSORRTAPI_H

#ifdef TENSORRT
#include "neuralnetapi.h"

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

/**
 * @brief The TensorrtAPI class implements the usage of the ONNX-TensorRT back-end for GPUs.
 */
class TensorrtAPI : public NeuralNetAPI
{
private:
    // neural network parameters
    samplesCommon::OnnxParams params;

    // input and output dimension of the network
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;

    // tensorRT runtime engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    void load_model();
    void load_parameters();
    void bind_executor();

public:
    /**
     * @brief TensorrtAPI
     * @param deviceID Device ID to use for computation.
     * @param batchSize Constant batch size which is used for inference
     * @param modelDirectory Directory where the network architecture is stored (.json file) and
     * where parameters a.k.a weights of the neural are stored (.params file) are stored
     */
    TensorrtAPI(int deviceID, unsigned int batchSize, const string& modelDirectory);

    NDArray predict(float* inputPlanes, float& value);
    void predict(float* inputPlanes, NDArray& valueOutput, NDArray& probOutputs);
};

#endif

#endif // TENSORRTAPI_H
