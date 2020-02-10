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
        info_string("Error opening :", strerror(errno));
        info_string(dir);
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
    deviceName = ctx + string("_") + to_string(deviceID);

    const vector<string>& files = get_directory_files(modelDirectory);
    for (const string& file : files) {
        size_t pos_json = file.find(".json");
        size_t pos_params = file.find(".params");
        if (pos_json != string::npos) {
            modelFilePath = modelDirectory + file;
        }
        else if (pos_params != string::npos) {
            paramterFilePath = modelDirectory + file;
            modelName = file.substr(0, file.length()-string(".params").length());
        }
    }
    if (modelFilePath == "" || paramterFilePath == "") {
        throw invalid_argument( "The given directory at " + modelDirectory
                                     + " doesn't contain a .json and a .params file.");
    }
    info_string("json file:", modelFilePath);
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

bool NeuralNetAPI::file_exists(const string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void NeuralNetAPI::check_if_policy_map()
{
    float* inputPlanes = new float[batchSize*NB_VALUES_TOTAL];
    fill(inputPlanes, inputPlanes+batchSize*NB_VALUES_TOTAL, 0.0f);

    float value;
    NDArray probOutputs = predict(inputPlanes, value);
    isPolicyMap = probOutputs.GetShape()[1] != NB_LABELS;
    info_string("isPolicyMap:", isPolicyMap);
    delete[] inputPlanes;
}
