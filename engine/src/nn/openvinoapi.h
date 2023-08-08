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
 * @file: openvinoapi.h
 * Created on 29.08.2021
 * @author: queensgambit
 *
 * This file describes the Intel OpenVino interface for CrazyAra networks.
 * More information about OpenVino can be found at:
 * https://github.com/openvinotoolkit/openvino
 * https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html
 */

#ifndef OPENVINOAPI_H
#define OPENVINOAPI_H

#ifdef OPENVINO
#include "neuralnetapi.h"

#include <ie_core.hpp>
#include "openvino/openvino.hpp"


/**
 * @brief The OpenVinoAPI class provides a compatible interface to use CrazyAra networks in the ONNX format using the OpenVino API.
 */
class OpenVinoAPI : public NeuralNetAPI
{
private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
    ov::InferRequest inferRequest;

    ov::Tensor inputTensor;
    float* rawInputData;
    size_t threadsNNInference;
public:
    OpenVinoAPI(int deviceID, unsigned int batchSize, const string &modelDirectory, size_t threadsNNInference);

    // NeuralNetAPI interface
private:
    void init_nn_design() override;
    void load_model() override;
    void load_parameters() override;
    void bind_executor() override;

    // helper methods
    void set_nn_value_policy_shape();
public:
    void predict(float *inputPlanes, float *valueOutput, float *probOutputs, float *auxiliaryOutputs) override;
};

/**
 * @brief set_shape Converter function from InferenceEngine::SizeVector to nn_api::Shape
 * @param shape Shape object to be set
 * @param dims Target object
 */
void set_shape(nn_api::Shape& shape, const InferenceEngine::SizeVector& sizeVector);

#endif

#endif // OPENVINOAPI_H
