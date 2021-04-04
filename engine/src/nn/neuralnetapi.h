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

// https://stackoverflow.com/questions/20446201/how-to-check-if-string-ends-with-txt/20446257
bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

/**
 * @brief get_string_ending_with Returns the first string of a list of strings ending with the given suffix
 * @param stringVector Vector of strings
 * @param suffix Suffix which must be at the end of the file
 * @return The filename of found file excluding the directory and "" and invalid_argument if no file was found
 */
string get_string_ending_with(const vector<string>& stringVector, const string& suffix);

/**
 * @brief get_items_with_elment Returns a vector of all elements of stringVector which contain element
 * @param stringVector Vector of strings
 * @param targetString String which searched for in all string vector items
 * @param shouldContain Boolean indicating if you want to get a vector where each item contains the targetString or
 * a vector where each item does not contain the targetString
 * @return new vector
 */
vector<string> get_items_by_elment(const vector<string>& stringVector, const string& targetString, bool shouldContain);


template <typename T>
/**
 * @brief assert_condition Wrapper for an assert statement that is also validate in release mode
 * @param value Given value
 * @param target Target value
 * @param valueStr Value description
 * @param targetStr Target description
 * @return True, if the assert statement is correct, else false
 */
bool assert_condition(const T& value, const T& target, const string& valueStr, const string& targetStr) {
    if (value != target) {
        std::cerr << valueStr << " != " << targetStr << ": " << value << " != " << target << endl;
        throw valueStr + string(" != ") + targetStr;
        return false;
    }
    return true;
}

namespace nn_api {
/**
 * @brief The Shape struct is a basic shape container object.
 */
struct Shape {
    int nbDims = -1;  // uninitialized
    int v[8];         // shape dimensions

};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

/**
 * @brief The NeuralNetDesign struct stores information about the neural network design.
 * It is supposed to be loaded dynamically from a neural network architecture file via the method `NeuralNetAPI->init_nn_design()`.
 */
struct NeuralNetDesign {
    bool isPolicyMap = false;
    bool hasAuxiliaryOutputs = false;
    const int valueOutputIdx = 0;
    const int policyOutputIdx = 1;
    const int auxiliaryOutputIdx = 2;
    Shape inputShape;
    Shape valueOutputShape;
    Shape policyOutputShape;
    Shape auxiliaryOutputShape;
    /**
     * @brief print Prints the outputs shapes using info_string(...)
     */
    void print() const;
};
}

/**
 * @brief The NeuralNetAPI class is an abstract class for accessing a neural network back-end and to run inference
 */
class NeuralNetAPI
{
protected:
    std::vector<std::string> outputLabels;
    int deviceID;
    unsigned int batchSize;
    bool enableTensorrt;
    // defines the name for the model based on the loaded .params file
    string modelName;
    // defines the device (e.g. GPU or CPU) and its respective deviceID
    string deviceName;

    // file names for the loaded model and its parameters
    string modelDir;
    string modelFilePath;
    string parameterFilePath;

    nn_api::NeuralNetDesign nnDesign;
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
     * @param auxiliaryOutputs Array of optional auxiliary outputs
     */
    virtual void predict(float* inputPlanes, float* valueOutput, float* probOutputs, float* auxiliaryOutputs) = 0;

    /**
     * @brief is_neural_network_valid Runs validation checks of the neural network architecture by comparing input and output shape of the loaded graph to the pre-defined constants.
     * @return True, if neural network is valid else false.
     */
    void validate_neural_network();

    /**
     * @brief get_policy_output_length Returns vector length for the policy output as returned by the neural network respecting the batch size
     * @return Vector length
     */
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
     * @brief init_nn_design Infers the input and output shapes of the loaded neural network architectures and
     * initializes the struct nnDesign.
     * and sets the selectPolicyFromPlane boolean accordingly
     */
    virtual void init_nn_design() = 0;
};

/**
 * @brief parse_directory Checks if the directory is empty and appends a "/" if necessary
 * @param directory Directory path which can both be relative or absolute
 * @return string with "/" as suffix
 */
string parse_directory(const string& directory);

#endif // NEURALNETAPI_H
