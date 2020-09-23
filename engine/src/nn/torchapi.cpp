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
#include "constants.h"

TorchAPI::TorchAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string &modelDirectory):
    NeuralNetAPI(ctx, deviceID, miniBatchSize, modelDirectory, false)
{
    modelFilePath = modelDir + "model-bsize-" + to_string(batchSize) + ".pt";

    load_model();
    check_if_policy_map();
    bind_executor();
}

void TorchAPI::predict(float *inputPlanes, float *valueOutput, float *probOutputs)
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs = {torch::from_blob(inputPlanes, {batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH})};

    // Execute the model and turn its output into a tensor.
    auto output = module.forward(inputs).toList();
    valueOutput = output.get(1).toTensor().data_ptr<float>();
    probOutputs = output.get(1).toTensor().data_ptr<float>();
}

void TorchAPI::load_model()
{
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(modelFilePath);
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
    // TODO
}
#endif
