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
 * References:
 * https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleOnnxMNIST
 * https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c
 * https://devblogs.nvidia.com/speed-up-inference-tensorrt/
 * https://github.com/NVIDIA/TensorRT/issues/322 (multi-gpu-support)
 * https://github.com/NVIDIA/TensorRT/blob/master/samples/opensource/sampleMovieLens/sampleMovieLens.cpp
 */

#ifndef TENSORRTAPI_H
#define TENSORRTAPI_H

#ifdef TENSORRT
#include "neuralnetapi.h"

#include "argsParser.h"
#include "buffers.h"
//#include "common.h"
//#include "logger.h"

#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>


//using namespace nvinfer1;
using namespace std;

enum Precision {
    float32,
    float16,
    int8
};

template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

/**
 * @brief The TensorrtAPI class implements the usage of the ONNX-TensorRT back-end for GPUs.
 */
class TensorrtAPI : public NeuralNetAPI
{
private:
    // input and output layer names
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;

    // input and output dimension of the network
    Precision precision;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims valueOutputDims;
    nvinfer1::Dims policyOutputDims;

    // tensorRT runtime engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;
    std::shared_ptr<samplesCommon::BufferManager> buffers;

    void load_model();
    void load_parameters();
    void bind_executor();

    void check_if_policy_map();

    /**
     * @brief createCudaEngineFromONNX Creates a new cuda engine from a onnx model architecture
     * @return ICudaEngine*
     */
    ICudaEngine* create_cuda_engine_from_onnx();

    /**
     * @brief get_cuda_engine Gets a the cuda engine by either loading a pre-existing tensorRT-engine from file or
     * otherwise creating and exporting a new tensorRT engine
     * @return ICudaEngine*
     */
    ICudaEngine* get_cuda_engine();

public:
    /**
     * @brief TensorrtAPI
     * @param deviceID Device ID to use for computation.
     * @param batchSize Constant batch size which is used for inference
     * @param modelDirectory Directory where the network architecture is stored (.json file) and
     * where parameters a.k.a weights of the neural are stored (.params file) are stored
     * @param precision Inference precision type. Available options: float32, float16, int8 (float32 is default).
     */
    TensorrtAPI(int deviceID, unsigned int batchSize, const string& modelDirectory, Precision precision = float32);

    void predict(float* inputPlanes, float* valueOutput, float* probOutputs);
};

/**
 * @brief set_config_settings Sets the configuration object which will be later used to build the engine
 * @param config Configuration object
 * @param precision Inference precision (e.g. float32, float16, int8)
 * @param maxWorkspace Maximum allowable GPU work space for TensorRT tactic selection (e.g. 16_MiB, 1_GiB)
 */
void set_config_settings(SampleUniquePtr<nvinfer1::IBuilderConfig> &config, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                         Precision precision, size_t maxWorkspace=1_GiB);

#endif

#endif // TENSORRTAPI_H
