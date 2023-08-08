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


MXNetAPI::MXNetAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string& modelDirectory, const string& strPrecision, bool tensorRT) :
    NeuralNetAPI(ctx, deviceID, miniBatchSize, modelDirectory, tensorRT),
    inputShape(Shape(miniBatchSize, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH()))
{
    fill_model_paths(strPrecision);
    info_string("json file:", modelFilePath);

    if (ctx == "cpu" || ctx == "CPU") {
        globalCtx = Context::cpu();
    } else if (ctx == "gpu" || ctx == "GPU") {
        globalCtx = Context::gpu(deviceID);
    } else {
        throw "unsupported context " + ctx + " given";
    }
    custom_initialize();
}

MXNetAPI::~MXNetAPI()
{
    delete executor;
}

void MXNetAPI::custom_initialize()
{
    load_model();
    load_parameters();
    bind_executor();
    initialize_nn_design();
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

void MXNetAPI::init_nn_design()
{
    set_shape(nnDesign.inputShape, inputShape);
    set_shape(nnDesign.policyOutputShape, executor->outputs[nnDesign.policyOutputIdx].GetShape());
    set_shape(nnDesign.valueOutputShape, executor->outputs[nnDesign.valueOutputIdx].GetShape());
    nnDesign.hasAuxiliaryOutputs = executor->outputs.size() > 2;
    if (nnDesign.hasAuxiliaryOutputs) {
        set_shape(nnDesign.auxiliaryOutputShape, executor->outputs[nnDesign.auxiliaryOutputIdx].GetShape());
    }

    float* inputPlanes = new float[batchSize*StateConstants::NB_VALUES_TOTAL()];
    fill(inputPlanes, inputPlanes+batchSize*StateConstants::NB_VALUES_TOTAL(), 0.0f);

    float value;
    NDArray probOutputs = predict(inputPlanes, value);
    nnDesign.isPolicyMap = probOutputs.GetShape()[1] != size_t(StateConstants::NB_LABELS());
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
    executor->outputs[1].SyncCopyToCPU(probOutputs, get_policy_output_length());
#ifdef DYNAMIC_NN_ARCH
    if (has_auxiliary_outputs()) {
        executor->outputs[2].SyncCopyToCPU(auxiliaryOutputs, get_nb_auxiliary_outputs()*batchSize);
    }
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS() != 0) {
         executor->outputs[2].SyncCopyToCPU(auxiliaryOutputs, StateConstants::NB_AUXILIARY_OUTPUTS()*batchSize);
    }
#endif
}

void set_shape(nn_api::Shape &shape, const std::vector<mx_uint> &mxnetShape)
{
    shape.nbDims = mxnetShape.size();
    for (uint idx = 0; idx < mxnetShape.size(); ++idx) {
        shape.v[idx] = mxnetShape[idx];
    }
}

void set_shape(nn_api::Shape &shape, const Shape &mxnetShape)
{
    shape.nbDims = mxnetShape.ndim();
    for (uint idx = 0; idx < mxnetShape.ndim(); ++idx) {
        shape.v[idx] = mxnetShape[idx];
    }
}

void MXNetAPI::fill_model_paths(const string& strPrecision)
{
    vector<string> files = get_directory_files(modelDir);
    if (strPrecision == "int8") {
        const vector<string>& int8Files = get_items_by_elment(files, "int8", true);
        if (int8Files.size() < 2) {  // we need at least to files
            info_string("No int8 model weights were found in directory " + modelDir);
            info_string("Falling back to float32 weights.");
        }
        else {
            files = int8Files;
        }
    }
    if (strPrecision == "float32") {
        files = get_items_by_elment(files, "int8", false);
    }
    const string fileSuffixModel = ".json";
    modelFilePath = get_string_ending_with(files, fileSuffixModel);
    if (modelFilePath == "") {
        throw invalid_argument( "The given directory at " + modelDir + " doesn't contain a file ending with " + fileSuffixModel);
    }
    modelFilePath = modelDir + modelFilePath;

    const string fileSuffixParams = ".params";
    parameterFilePath = get_string_ending_with(files, fileSuffixParams);
    if (parameterFilePath == "") {
        throw invalid_argument( "The given directory at " + modelDir + " doesn't contain a file ending with " + fileSuffixParams);
    }

    modelName = parameterFilePath.substr(0, parameterFilePath.length()-fileSuffixParams.length());
    parameterFilePath = modelDir + parameterFilePath;
}

#endif
