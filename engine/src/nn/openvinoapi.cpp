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
    NeuralNetAPI("cpu", deviceID, batchSize, modelDirectory, true),
    rawInputData(nullptr),
    threadsNNInference(threadsNNInference)
{
    modelName = get_onnx_model_name(modelDir, batchSize);
    modelFilePath = modelDir + "/" + modelName;
    initialize();
}

void OpenVinoAPI::set_nn_value_policy_shape()
{
    set_shape(nnDesign.policyOutputShape, model->get_output_shape(nnDesign.policyOutputIdx));
    set_shape(nnDesign.valueOutputShape, model->get_output_shape(nnDesign.valueOutputIdx));
}

void OpenVinoAPI::init_nn_design()
{
    set_shape(nnDesign.inputShape, model->input().get_shape());
    set_nn_value_policy_shape();

    nnDesign.hasAuxiliaryOutputs = model->get_output_size() > 2;

    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, model->get_output_shape(nnDesign.auxiliaryOutputIdx));
    }
    nnDesign.isPolicyMap = uint(nnDesign.policyOutputShape.v[1]) != (StateConstants::NB_LABELS());
    nbPolicyValues = nnDesign.policyOutputShape.v[1];
    nbNNInputValues = nnDesign.inputShape.flatten();
}

void OpenVinoAPI::load_model()
{
    // load the model architecture
    model = core.read_model(modelFilePath);
    // set the batch size
    if (model->is_dynamic()) {
        model->get_parameters()[nnDesign.inputIdx]->set_layout("NCHW");
        ov::set_batch(model, batchSize);
    }
}

void OpenVinoAPI::load_parameters()
{
    // load the model to the device
    compiledModel = core.compile_model(model, "CPU", ov::inference_num_threads(threadsNNInference));
}

void OpenVinoAPI::bind_executor()
{
    // create an infer request
    inferRequest = compiledModel.create_infer_request();

    // allocate required objects before the actual inference
    ov::element::Type inputType = ov::element::f32;
    ov::Shape inputShape = {batchSize, unsigned(nnDesign.inputShape.v[1]),
                            unsigned(nnDesign.inputShape.v[2]),
                            unsigned(nnDesign.inputShape.v[3])};

    inputTensor = ov::Tensor(inputType, inputShape);
    rawInputData = (float*)inputTensor.data();
    inferRequest.set_input_tensor(inputTensor);
}

void OpenVinoAPI::predict(float *inputPlanes, float *valueOutput, float *probOutputs, float *auxiliaryOutputs)
{
    // copy over the input planes into the raw data cotainer
    std::copy(inputPlanes, inputPlanes + batchSize * get_nb_input_values_total(), rawInputData);

    // run the request synchronously
    inferRequest.infer();

    // process outputs
    const ov::Tensor& outputTensorValue = inferRequest.get_output_tensor(nnDesign.valueOutputIdx);
    const ov::Tensor& outputTensorPolicy = inferRequest.get_output_tensor(nnDesign.policyOutputIdx);

    const float* outputBufferValue = (const float*)outputTensorValue.data();
    const float* outputBufferPolicy = (const float*)outputTensorPolicy.data();

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
