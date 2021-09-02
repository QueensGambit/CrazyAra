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
 * @file: openvinoapi.cpp
 * Created on 29.08.2021
 * @author: queensgambit
 */

#ifdef OPENVINO
#include "openvinoapi.h"
#include "stateobj.h"

OpenVinoAPI::OpenVinoAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, size_t threadsNNInference):
    NeuralNetAPI("gpu", deviceID, batchSize, modelDirectory, true),
    rawInputData(nullptr),
    threadsNNInference(threadsNNInference)
{
    modelName = get_file_ending_with(modelDir, "-bsize-" + to_string(batchSize) + ".onnx");
    modelFilePath = modelDir + "/" + modelName;
    initialize();
}

void OpenVinoAPI::set_nn_value_policy_shape()
{
    if (outputInfo.find(nnDesign.policyOutputName) == outputInfo.end() || outputInfo.find(nnDesign.valueOutputName) == outputInfo.end()) {
        info_string_important(nnDesign.policyOutputName, " or ", nnDesign.valueOutputName, "not found. Fallback to default indices.");
        nnDesign.valueOutputName = outputInfo.rbegin()->first;
        nnDesign.policyOutputName = outputInfo.begin()->first;
    }
    set_shape(nnDesign.policyOutputShape, outputInfo.at(nnDesign.policyOutputName).get()->getDims());
    set_shape(nnDesign.valueOutputShape, outputInfo.at(nnDesign.valueOutputName).get()->getDims());
}

void OpenVinoAPI::init_nn_design()
{
    set_shape(nnDesign.inputShape, network.getInputShapes().at("data"));
    set_nn_value_policy_shape();

    nnDesign.hasAuxiliaryOutputs = outputInfo.size() > 2;
    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, outputInfo.at(nnDesign.auxiliaryOutputName).get()->getDims());
    }
    nnDesign.isPolicyMap = uint(nnDesign.policyOutputShape.v[1]) != (StateConstants::NB_LABELS());
    policyOutputLength = nnDesign.policyOutputShape.v[1] * batchSize;
    nbNNInputValues = nnDesign.inputShape.flatten();
}

void OpenVinoAPI::load_model()
{
    // load the model architecture
    network = core.ReadNetwork(modelFilePath);

    // get information about all topology inputs
    inputInfo = network.getInputsInfo();
    // get information about all topology outputs
    outputInfo = network.getOutputsInfo();

    // set precision and layout for input data
    inputInfo.at(nnDesign.inputLayerName)->setLayout(InferenceEngine::Layout::NCHW);
    inputInfo.at(nnDesign.inputLayerName)->setPrecision(InferenceEngine::Precision::FP32);

    // set precision for outputs
    for (auto output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
    }
}

void OpenVinoAPI::load_parameters()
{
    // load the model to the device
    std::map<std::string, std::string> config = {
        { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(threadsNNInference).c_str() }
    };
    executableNetwork = core.LoadNetwork(network, "CPU", config);
}

void OpenVinoAPI::bind_executor()
{
    // create an infer request
    inferRequest = executableNetwork.CreateInferRequest();

    // allocate required objects before the actual inference
    inputBlob = inferRequest.GetBlob(nnDesign.inputLayerName);
    inputBlob->allocate();
    rawInputData = inputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    inferRequest.SetBlob(nnDesign.inputLayerName, inputBlob);
}

void OpenVinoAPI::predict(float *inputPlanes, float *valueOutput, float *probOutputs, float *auxiliaryOutputs)
{
    // copy over the input planes into the raw data cotainer
    std::copy(inputPlanes, inputPlanes + batchSize * get_nb_input_values_total(), rawInputData);

    // run the request synchronously
    inferRequest.Infer();

    // process outputs
    outputBlobValue = inferRequest.GetBlob(nnDesign.valueOutputName);
    outputBlobPolicy = inferRequest.GetBlob(nnDesign.policyOutputName);

    auto const memLockerValue = outputBlobValue->cbuffer(); // use const memory locker
    auto const memLockerPolicy = outputBlobPolicy->cbuffer(); // use const memory locker

    const float* outputBufferValue = memLockerValue.as<const float *>();
    const float* outputBufferPolicy = memLockerPolicy.as<const float *>();

    // copy the outputs to the given pointers
    std::copy(outputBufferValue, outputBufferValue + batchSize, valueOutput);
    std::copy(outputBufferPolicy, outputBufferPolicy + get_policy_output_length(), probOutputs);

    for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        apply_softmax(probOutputs + batchIdx * nnDesign.policyOutputShape.v[1], nnDesign.policyOutputShape.v[1]);
    }
}

void set_shape(nn_api::Shape& shape, const InferenceEngine::SizeVector& sizeVector)
{
    shape.nbDims = sizeVector.size();
    for (int idx = 0; idx < shape.nbDims; ++idx) {
        shape.v[idx] = sizeVector[idx];
    }
}
#endif
