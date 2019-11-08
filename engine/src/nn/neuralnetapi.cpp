/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
 * @file: neuralnetapi.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "neuralnetapi.h"
#include <dirent.h>
#include <exception>
#include <string>
#include "../domain/crazyhouse/constants.h"

// http://www.codebind.com/cpp-tutorial/cpp-program-list-files-directory-windows-linux/
namespace {
vector<string> get_directory_files(const string& dir) {
    vector<string> files;
    shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;
    if (!directory_ptr) {
        cout << "info string Error opening : " << strerror(errno) << " " << dir << endl;
        return files;
    }

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        files.push_back(string(dirent_ptr->d_name));
    }
    return files;
}
}  // namespace

NeuralNetAPI::NeuralNetAPI(const string& ctx, int deviceID, unsigned int batchSize, const string& modelDirectory, bool enableTensorrt):
    batchSize(batchSize),
    enableTensorrt(enableTensorrt)
{
    if (ctx == "cpu" || ctx == "CPU") {
        globalCtx = Context::cpu();
    } else if (ctx == "gpu" || ctx == "GPU") {
        globalCtx = Context::gpu(deviceID);
    } else {
        throw "unsupported context " + ctx + " given";
    }
    deviceName = ctx + string("_") + to_string(deviceID);

    string jsonFilePath;
    string paramterFilePath;

    const vector<string>& files = get_directory_files(modelDirectory);
    for (const string& file : files) {
        size_t pos_json = file.find(".json");
        size_t pos_params = file.find(".params");
        if (pos_json != string::npos) {
            jsonFilePath = modelDirectory + file;
        }
        else if (pos_params != string::npos) {
            paramterFilePath = modelDirectory + file;
            modelName = file.substr(0, file.length()-string(".params").length());
        }
    }
    if (jsonFilePath == "" || paramterFilePath == "") {
        throw invalid_argument( "The given directory at " + modelDirectory
                                     + " doesn't contain a .json and a .params file.");
    }
	cout << "info string json file: " << jsonFilePath << endl;

    inputShape = Shape(batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH);
    load_model(jsonFilePath);
    load_parameters(paramterFilePath);
    bind_executor();
    check_if_policy_map();
}

NeuralNetAPI::~NeuralNetAPI()
{
    delete executor;
}

bool NeuralNetAPI::is_policy_map() const
{
    return isPolicyMap;
}

string NeuralNetAPI::get_model_name() const
{
    return modelName;
}

string NeuralNetAPI::get_device_name() const
{
    return deviceName;
}

bool NeuralNetAPI::file_exists(const string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void NeuralNetAPI::load_model(const string &jsonFilePath)
{
    if (!file_exists(jsonFilePath)) {
		cout << "info string Model file " << jsonFilePath << " does not exist";
        throw runtime_error("Model file does not exist");
    }
	cout << "info string Loading the model from " << jsonFilePath << endl;
    net = Symbol::Load(jsonFilePath);
    if (enableTensorrt) {
      #ifdef TENSORRT
      net = net.GetBackendSymbol("TensorRT");
      #endif
    }
}

void NeuralNetAPI::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
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

void NeuralNetAPI::ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *paramMapInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    (*paramMapInTargetContext)[pair.first] = pair.second.Copy(targetContext);
  }
}

void NeuralNetAPI::load_parameters(const string& paramterFilePath) {
    if (!file_exists(paramterFilePath)) {
        cout << "info string Parameter file " << paramterFilePath << " does not exist";
        throw runtime_error("Model parameters does not exist");
    }
	cout << "info string Loading the model parameters from " << paramterFilePath << endl;
    map<string, NDArray> parameters;
    NDArray::Load(paramterFilePath, 0, &parameters);

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

void NeuralNetAPI::bind_executor()
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
	cout << "info string Bind successfull!" << endl;
}

void NeuralNetAPI::check_if_policy_map()
{
    float* inputPlanes = new float[batchSize*NB_VALUES_TOTAL];
    fill(inputPlanes, inputPlanes+batchSize*NB_VALUES_TOTAL, 0.0f);

    float value;
    NDArray probOutputs = predict(inputPlanes, value);
    isPolicyMap = probOutputs.GetShape()[1] != NB_LABELS;
    cout << "info string isPolicyMap: " << isPolicyMap << endl;
    delete[] inputPlanes;
}

NDArray NeuralNetAPI::predict(float *inputPlanes, float &value)
{
    executor->arg_dict()["data"].SyncCopyFromCPU(inputPlanes, NB_VALUES_TOTAL * batchSize);

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

void NeuralNetAPI::predict(float *inputPlanes, NDArray &valueOutput, NDArray &probOutputs)
{
    executor->arg_dict()["data"].SyncCopyFromCPU(inputPlanes, NB_VALUES_TOTAL * batchSize);

    // Run the forward pass.
    executor->Forward(false);

    valueOutput = executor->outputs[0].Copy(Context::cpu());
    probOutputs = executor->outputs[1].Copy(Context::cpu());

    // Assign the value output to the return parameter
    valueOutput.WaitToRead();
    probOutputs.WaitToRead();
}
