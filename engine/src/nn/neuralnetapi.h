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
 * @file: neuralnetapi.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * This file defines a general interface for implementing a neural network back-end.
 */

#ifndef NEURALNETAPI_H
#define NEURALNETAPI_H

#include <iostream>
#include <sys/stat.h>
#include <mutex>
#include <vector>
#include <memory>
#include <dirent.h>
#include <cstring>
#include "../util/communication.h"


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

/**
 * @brief The NeuralNetAPI class is an abstract class for accessing a neural network back-end and to run inference
 */
class NeuralNetAPI
{
protected:
    std::vector<std::string> outputLabels;
    int deviceID;
    unsigned int batchSize;
    // vector length for the policy output as returned by the neural network respecting the batch size
    unsigned int policyOutputLength;
    bool isPolicyMap;
    bool enableTensorrt;
    // defines the name for the model based on the loaded .params file
    string modelName;
    // defines the device (e.g. GPU or CPU) and its respective deviceID
    string deviceName;

    // file names for the loaded model and its parameters
    string modelDir;
    string modelFilePath;
    string paramterFilePath;

public:
    /**
     * @brief NeuralNetAPI
     * @param ctx Computation contex either "cpu" or "gpu"
     * @param deviceID Device ID to use for computation. Only used for gpu context.
     * @param batchSize Constant batch size which is used for inference
     * @param modelDirectory Directory where the network architecture is stored (.json file) and
     * where parameters a.k.a weights of the neural are stored (.params file) are stored
     */
    NeuralNetAPI(const string& ctx, int deviceID, unsigned int batchSize, const string& modelDirectory, bool enableTensorrt);

    /**
     * @brief is_policy_map Returns true if the policy outputs is defined in policy map representation else false
     * @return bool
     */
    bool is_policy_map() const;

    /**
     * @brief get_model_name Returns the name of the model based on the loaded parameter/weight file
     * @return string
     */
    string get_model_name() const;

    /**
     * @brief get_device_name Returns the device name (e.g. gpu_0, or cpu_0)
     * @return string
     */
    string get_device_name() const;

    /**
     * @brief predict Runs a prediction on the given inputPlanes and returns the policy vector in form of a NDArray and the value as a float number
     * @param inputPlanes Pointer to the input planes of a single board position
     * @param value Value prediction for the board by the neural network
     * @param probOutputs Policy array of the raw network output (including illegal moves). It's assumend that the memory has already been allocated.
     */
    virtual void predict(float* inputPlanes, float* valueOutput, float* probOutputs) = 0;

    unsigned int get_policy_output_length() const;

    unsigned int get_batch_size() const;

protected:
    /**
     * @brief FileExists Function to check if a file exists in a given path
     * @param name Filepath
     * @return True if exists else false
     */
    bool file_exists(const std::string& name);

    /**
     * @brief load_model Loads the model architecture definition from a json file
     */
    virtual void load_model() = 0;

    /**
     * @brief load_parameters Loads the parameters a.k.a weights of the model given a parameter path
     */
    virtual void load_parameters() = 0;

    /**
     * @brief bind_executor Binds the executor object to the neural network
     */
    virtual void bind_executor() = 0;

    /**
     * @brief infer_select_policy_from_planes Checks if the loaded model encodes the policy as planes
     * and sets the selectPolicyFromPlane boolean accordingly
     */
    virtual void check_if_policy_map() = 0;
};

/**
 * @brief parse_directory Checks if the directory is empty and appends a "/" if necessary
 * @param directory Directory path which can both be relative or absolute
 * @return string with "/" as suffix
 */
string parse_directory(const string& directory);

#endif // NEURALNETAPI_H
