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
 * @file: torchapi.cpp
 * Created on 23.09.2020
 * @author: queensgambit
 */

#ifdef TORCH
#include "torchapi.h"
#include "stateobj.h"

TorchAPI::TorchAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string &modelDirectory):
    NeuralNetAPI(ctx, deviceID, miniBatchSize, modelDirectory, false),
    device(torch::kCPU)
{
    modelFilePath = modelDir + "model-bsize-" + to_string(batchSize) + ".pt";
    if (ctx == "cpu" || ctx == "CPU") {
        device = torch::kCPU;
    } else if (ctx == "gpu" || ctx == "GPU") {
        device = torch::Device(torch::kCUDA, deviceID);
    } else {
        throw "unsupported context " + ctx + " given";
    }
    load_model();
    check_if_policy_map();
    bind_executor();
}

void TorchAPI::predict(float *inputPlanes, float *valueOutput, float *probOutputs, float *auxiliaryOutputs)
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs = {torch::from_blob(inputPlanes, {batchSize, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH()}, device)};

    // Execute the model and turn its output into a tensor.
    auto output = module.forward(inputs).toList();

    const float* torchValuePt = output.get(0).toTensor().data_ptr<float>();
    std::copy(torchValuePt, torchValuePt+batchSize, valueOutput);
    const float* torchPolicyPt = torch::softmax(output.get(1).toTensor(), 1).data_ptr<float>();
    std::copy(torchPolicyPt, torchPolicyPt+policyOutputLength, probOutputs);
    if (StateConstants::NB_AUXILIARY_OUTPUTS() != 0) {
        const float* torchAuxiliaryPt = output.get(2).toTensor().data_ptr<float>();
        std::copy(torchAuxiliaryPt, torchAuxiliaryPt+StateConstants::NB_AUXILIARY_OUTPUTS()*batchSize, auxiliaryOutputs);
    }
}

void TorchAPI::load_model()
{
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(modelFilePath, device);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model: " <<  modelFilePath << std::endl;
    }
}

void TorchAPI::load_parameters()
{
    // pass
}

void TorchAPI::bind_executor()
{
    // pass
}

void TorchAPI::check_if_policy_map()
{
    float* inputPlanes = new float[batchSize*StateConstants::NB_VALUES_TOTAL()];

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs = {torch::from_blob(inputPlanes, {batchSize, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH()}, device)};

    auto output = module.forward(inputs).toList();
    auto probOutputs = output.get(1).toTensor();

    isPolicyMap = probOutputs.size(1) != StateConstants::NB_LABELS();
    info_string("isPolicyMap:", isPolicyMap);
    if (isPolicyMap) {
        policyOutputLength = StateConstants::NB_LABELS_POLICY_MAP() * batchSize;
    }
}
#endif
