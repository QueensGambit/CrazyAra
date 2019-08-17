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
		cout << "info string Error opening : " << strerror(errno) << dir << endl;
        return files;
    }

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        files.push_back(string(dirent_ptr->d_name));
    }
    return files;
}
}  // namespace

NeuralNetAPI::NeuralNetAPI(const string& ctx, unsigned int batchSize, const string& modelDirectory):
    batchSize(batchSize)
{
    if (ctx == "cpu" || ctx == "CPU") {
        global_ctx = Context::cpu();
    } else if (ctx == "gpu" || ctx == "GPU") {
        global_ctx = Context::gpu();
    } else {
        throw "unsupported context " + ctx + " given";
    }

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
        }
    }
    if (jsonFilePath == "" || paramterFilePath == "") {
        throw invalid_argument( "The given directory at " + modelDirectory
                                     + " doesn't contain a .json and a .params file.");
    }

	cout << "info string json file: " << jsonFilePath << endl;

    input_shape =  Shape(batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH);

    load_model(jsonFilePath);
    load_parameters(paramterFilePath);
    bind_executor();
    infer_select_policy_from_planes();
}

bool NeuralNetAPI::getSelectPolicyFromPlane() const
{
    return selectPolicyFromPlane;
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
}

void NeuralNetAPI::load_parameters(const string& paramterFilePath) {
    if (!file_exists(paramterFilePath)) {
        cout << "info string Parameter file " << paramterFilePath << " does not exist";
        throw runtime_error("Model parameters does not exist");
    }
	cout << "info string Loading the model parameters from " << paramterFilePath << endl;
    map<string, NDArray> parameters;
    NDArray::Load(paramterFilePath, 0, &parameters);
    for (const auto &k : parameters) {
        if (k.first.substr(0, 4) == "aux:") {
            auto name = k.first.substr(4, k.first.size() - 4);
            aux_map[name] = k.second.Copy(global_ctx);
        }
        if (k.first.substr(0, 4) == "arg:") {
            auto name = k.first.substr(4, k.first.size() - 4);
            args_map[name] = k.second.Copy(global_ctx);
        }
    }
    // WaitAll is need when we copy data between GPU and the main memory
    NDArray::WaitAll();
}

void NeuralNetAPI::bind_executor()
{
    // Create an executor after binding the model to input parameters.
    args_map["data"] = NDArray(input_shape, global_ctx, false);
    /* new */
    vector<NDArray> arg_arrays;
    vector<NDArray> grad_arrays;
    vector<OpReqType> grad_reqs;
    vector<NDArray> aux_arrays;
    Shape value_label_shape(input_shape[0]);
    Shape policy_label_shape(input_shape[0]);

    args_map["value_label"] = NDArray(value_label_shape, global_ctx, false);
    args_map["policy_label"] = NDArray(policy_label_shape, global_ctx, false);

    net.InferExecutorArrays(global_ctx, &arg_arrays, &grad_arrays, &grad_reqs,
                            &aux_arrays, args_map, map<string, NDArray>(),
                            map<string, OpReqType>(), aux_map);
    for (size_t i = 0; i < grad_reqs.size(); ++i) {
        grad_reqs[i] = kNullOp;
    }

    executor = new Executor(net, global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
	cout << "info string Bind successfull!" << endl;
}

void NeuralNetAPI::infer_select_policy_from_planes()
{
    float* inputPlanes = new float[batchSize*NB_VALUES_TOTAL];
    fill(inputPlanes, inputPlanes+batchSize*NB_VALUES_TOTAL, 0.0f);

    float value;
    NDArray probOutputs = predict(inputPlanes, value);
    selectPolicyFromPlane = probOutputs.GetShape()[1] != NB_LABELS;
	cout << "info string selectPolicyFromPlane: " << selectPolicyFromPlane << endl;
	delete inputPlanes;
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

    // Assign the value output to the return paramter
    valueOutput.WaitToRead();
    probOutputs.WaitToRead();
}
