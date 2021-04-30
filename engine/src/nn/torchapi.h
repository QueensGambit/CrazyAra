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
 * @file: torchapi.h
 * Created on 23.09.2020
 * @author: queensgambit
 *
 * This file contains wrappers for handling the neural network.
 * Parts of the code are based on the TorchScript Model in C++ tutorial:
 * https://pytorch.org/tutorials/advanced/cpp_export.html
 */

#ifdef TORCH
#ifndef TORCHAPI_H
#define TORCHAPI_H

#include "neuralnetapi.h"
#include <torch/script.h>

/**
 * @brief The TorchAPI class implements access to the Lib Torch-C++ back-end for running inference on CPU and GPU for torchscript models.
 */
class TorchAPI : public NeuralNetAPI
{
private:
    torch::jit::script::Module module;
    torch::Device device;
public:
    TorchAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string& modelDirectory);

    // NeuralNetAPI interface
    void predict(float *inputPlanes, float *valueOutput, float *probOutputs, float *auxiliaryOutputs) override;

protected:
    void load_model() override;
    void load_parameters() override;
    void bind_executor() override;
    void init_nn_design() override;
};

/**
 * @brief set_shape Converter function from at::Tensor& tensor to nn_api::Shape
 * @param shape Shape object to be set
 * @param tensor Target object
 */
void set_shape(nn_api::Shape& shape, const at::Tensor& tensor);

/**
 * @brief set_shape Converter function from at::IntArrayRef to nn_api::Shape
 * @param shape Shape object to be set
 * @param sizes Target object
 */
void set_shape(nn_api::Shape& shape, const at::IntArrayRef& sizes);


#endif // TORCHAPI_H
#endif
