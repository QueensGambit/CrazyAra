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
 * @file: mxnetapi.cpp
 * Created on 05.02.2020
 * @author: queensgambit
 */

#include "mxnetapi.h"

#ifdef MXNET
#include "../util/communication.h"
#include "stateobj.h"


MXNetAPI::MXNetAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string& modelDirectory, bool tensorRT) :
    NeuralNetAPI(ctx, deviceID, miniBatchSize, modelDirectory, tensorRT),
    inputShape(Shape(miniBatchSize, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH()))
{
    modelFilePath = modelDir + get_file_ending_with(modelDir, ".json");
    parameterFilePath = get_file_ending_with(modelDir, ".params");;
    modelName = parameterFilePath.substr(0, parameterFilePath.length()-string(".params").length());
    parameterFilePath = modelDir + parameterFilePath;

    info_string("json file:", modelFilePath);

    if (ctx == "cpu" || ctx == "CPU") {
        globalCtx = Context::cpu();
    } else if (ctx == "gpu" || ctx == "GPU") {
        globalCtx = Context::gpu(deviceID);
    } else {
        throw "unsupported context " + ctx + " given";
    }

    load_model();
    load_parameters();
    bind_executor();
    check_if_policy_map();
}

MXNetAPI::~MXNetAPI()
{
    delete executor;
}

void MXNetAPI::load_model()
{
    if (!file_exists(modelFilePath)) {
        info_string("Model file  does not exist", modelFilePath);
        throw runtime_error("Model file does not exist");
    }
    info_string("Loading the model from", modelFilePath);
    net = Symbol::Load(modelFilePath);
    if (enableTensorrt) {
      #ifdef TENSORRT
      net = net.GetBackendSymbol("TensorRT");
      #endif
    }
}

void MXNetAPI::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *argParamInTargetContext,
    std::map<std::string, NDArray> *auxParamInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    std::string type = pair.first.substr(0, 4);
    std::string name = pair.first.substr(4);
    if (type == "arg:") {
      (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
    } else if (type == "aux:") {
      (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
    }
  }
}

void MXNetAPI::ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *paramMapInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    (*paramMapInTargetContext)[pair.first] = pair.second.Copy(targetContext);
  }
}

void MXNetAPI::load_parameters() {
    if (!file_exists(parameterFilePath)) {
        info_string("Parameter file does not exist:", parameterFilePath);
        throw runtime_error("Model parameters does not exist");
    }
    info_string("Loading the model parameters from:", parameterFilePath);
    map<string, NDArray> parameters;
    NDArray::Load(parameterFilePath, 0, &parameters);

    if (enableTensorrt) {
      #ifdef TENSORRT
      std::map<std::string, NDArray> intermediate_args_map;
      std::map<std::string, NDArray> intermediate_aux_map;
      SplitParamMap(parameters, &intermediate_args_map, &intermediate_aux_map, Context::cpu());
      contrib::InitTensorRTParams(net, &intermediate_args_map, &intermediate_aux_map);
      ConvertParamMapToTargetContext(intermediate_args_map, &argsMap, globalCtx);
      ConvertParamMapToTargetContext(intermediate_aux_map, &auxMap, globalCtx);
      #endif
    } else {
      SplitParamMap(parameters, &argsMap, &auxMap, globalCtx);
    }

    // WaitAll is needed when data is copied between GPU and the main memory
    NDArray::WaitAll();
}

void MXNetAPI::bind_executor()
{
    // Create an executor after binding the model to input parameters.
    argsMap["data"] = NDArray(inputShape, globalCtx, false);
    /* new */
    vector<NDArray> argArrays;
    vector<NDArray> gradArrays;
    vector<OpReqType> gradReqs;
    vector<NDArray> auxArrays;
    Shape value_label_shape(inputShape[0]);
    Shape policy_label_shape(inputShape[0]);

    argsMap["value_label"] = NDArray(value_label_shape, globalCtx, false);
    argsMap["policy_label"] = NDArray(policy_label_shape, globalCtx, false);

    net.InferExecutorArrays(globalCtx, &argArrays, &gradArrays, &gradReqs,
                            &auxArrays, argsMap, map<string, NDArray>(),
                            map<string, OpReqType>(), auxMap);
    for (size_t i = 0; i < gradReqs.size(); ++i) {
        gradReqs[i] = kNullOp;
    }

    executor = new Executor(net, globalCtx, argArrays, gradArrays, gradReqs, auxArrays);
    info_string("Bind successfull!");
}

void MXNetAPI::check_if_policy_map()
{
    float* inputPlanes = new float[batchSize*StateConstants::NB_VALUES_TOTAL()];
    fill(inputPlanes, inputPlanes+batchSize*StateConstants::NB_VALUES_TOTAL(), 0.0f);

    float value;
    NDArray probOutputs = predict(inputPlanes, value);
    isPolicyMap = probOutputs.GetShape()[1] != size_t(StateConstants::NB_LABELS());
    info_string("isPolicyMap:", isPolicyMap);
    if (isPolicyMap) {
        policyOutputLength = StateConstants::NB_LABELS_POLICY_MAP() * batchSize;
    }
    delete[] inputPlanes;
}

NDArray MXNetAPI::predict(float* inputPlanes, float& value)
{
    executor->arg_dict()["data"].SyncCopyFromCPU(inputPlanes, StateConstants::NB_VALUES_TOTAL() * batchSize);

    // Run the forward pass.
    executor->Forward(false);

    auto valueOutput = executor->outputs[0].Copy(Context::cpu());
    auto probOutputs = executor->outputs[1].Copy(Context::cpu());

    // Assign the value output to the return paramter
    valueOutput.WaitToRead();
    value = valueOutput.At(0, 0);

    probOutputs.WaitToRead();

    auto predicted = probOutputs.ArgmaxChannel();
    predicted.WaitToRead();

    return probOutputs;
}

void MXNetAPI::predict(float *inputPlanes, float* valueOutput, float* probOutputs, float* auxiliaryOutputs)
{
    executor->arg_dict()["data"].SyncCopyFromCPU(inputPlanes, StateConstants::NB_VALUES_TOTAL() * batchSize);

    // Run the forward pass.
    executor->Forward(false);

    executor->outputs[0].SyncCopyToCPU(valueOutput, batchSize);
    executor->outputs[1].SyncCopyToCPU(probOutputs, policyOutputLength);
    if (StateConstants::NB_AUXILIARY_OUTPUTS() != 0) {
        executor->outputs[2].SyncCopyToCPU(auxiliaryOutputs, StateConstants::NB_AUXILIARY_OUTPUTS()*batchSize);
    }
}

#endif
