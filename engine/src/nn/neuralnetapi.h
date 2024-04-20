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
#include "neuralnetdesign.h"
#include "version.h"
#include "../stateobj.h"

// http://www.codebind.com/cpp-tutorial/cpp-program-list-files-directory-windows-linux/
namespace {
vector<string> get_directory_files(const string& dir) {
    vector<string> files;
    shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;
    if (!directory_ptr) {
        info_string_important("Error opening :", dir, "(" + string(strerror(errno)) + ")");
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

/**
 * @brief get_file_ending_with Returns the first file of a directory ending with the given suffix
 * @param dir Directory where to look for the file
 * @param suffix Suffix which must be at the end of the file
 * @return The filename of found file excluding the directory and "" and invalid_argument if no file was found
 */
string get_file_ending_with(const string& dir, const string& suffix);

/**
 * @brief read_version_from_string Returns the corresponding version for a given model file name.
 * The version identifier is expected to come after the substring "-v" in the format "-v<Major>.<Minor>", e.g. "-v1.2.onnx".
 * If the information is missing or parsing failed, make_version<0,0,0>() will be returned.
 * Versioning patch information is always set to 0.
 * The version information is used to decide between different input representations for the neural network.
 * @param modelFileName
 * @return Version information
 */
Version read_version_from_string(const string& modelFileName);

/**
 * @brief read_game_phase_from_string Returns the GamePhase a given model directory belongs to based on its last character
 * e.g. "/model/ClassicAra/chess/separated_learning/phase0" indicates that the model in this directory belongs to phase 0
 * @param modelDir Model directory
 * @return GamePhase
 */
GamePhase read_game_phase_from_string(const string& modelDir);


template <typename T>
/**
 * @brief check_condition Wrapper for a condition that is also validate in release mode
 * @param value Given value
 * @param target Target value
 * @param valueStr Value description
 * @param targetStr Target description
 * @return True, if the assert statement is correct, else false
 */
bool check_condition(const T& value, const T& target, const string& valueStr, const string& targetStr) {
    if (value != target) {
        info_string(valueStr + " !=", targetStr + ":");
        info_string("expected:", target);
        info_string("given:", value);
        return false;
    }
    return true;
}


/**
 * @brief get_onnx_model_name Returns the model name in the given model directory based on the given batch size.
 * If no file is found, it looks for an onnx file with dynamic shape support.
 * If this is satisfied neither, an exception is thrown.
 * @param modelDir Model directory
 * @param batchSize Batch size
 * @return model directory as string
 */
string get_onnx_model_name(const string& modelDir, int batchSize);


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
    uint_fast32_t nbNNInputValues;
    uint_fast32_t nbNNAuxiliaryOutputs;
    uint_fast32_t nbPolicyValues;

    Version version;
    GamePhase gamePhase;
private:
    /**
     * @brief init_nn_design Infers the input and output shapes of the loaded neural network architectures and
     * initializes the struct nnDesign.
     * and sets the selectPolicyFromPlane boolean accordingly
     */
    virtual void init_nn_design() = 0;

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
    * @brief get_game_phase Returns the game phase of this NeuralNetAPI
    * @return GamePhase
    */
    GamePhase get_game_phase() const;

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
     * @brief get_policy_output_length Returns the number of policy values for a single batch
     * @return Vector length
     */
    inline uint_fast32_t get_nb_policy_values() const {
        return nbPolicyValues;
    }

    /**
     * @brief get_policy_output_length Returns vector length for the policy output as returned by the neural network respecting the batch size
     * @return Vector length
     */
    inline uint_fast32_t get_policy_output_length() const {
        return get_nb_policy_values() * batchSize;
    }

    /**
     * @brief get_nb_input_values_total Returns the total number of input values for a single batch
     * @return uint
     */
    inline uint_fast32_t get_nb_input_values_total() const {
        return nbNNInputValues;
    }

    /**
     * @brief get_nb_auxiliary_outputs Returns the total number of auxiliary outputs for a single batch infered form the nnDesign
     * @return uint
     */
    inline uint_fast32_t get_nb_auxiliary_outputs() const {
        return nbNNAuxiliaryOutputs;
    }

    /**
     * @brief has_auxiliary_outputs Returns nnDesign.hasAuxiliaryOutputs
     * @return bool
     */
    inline bool has_auxiliary_outputs() const {
        return nnDesign.hasAuxiliaryOutputs;
    }

    /**
     * @brief get_version Returns the loaded version of the neural network.
     * @return Version
     */
    inline Version get_version() const {
        return version;
    }

    unsigned int get_batch_size() const;

    /**
     * @brief initialize Initializes the neural net api using the template method pattern
     */
    void initialize();
protected:
    /**
     * @brief FileExists Function to check if a file exists in a given path
     * @param name Filepath
     * @return True if exists else false
     */
    bool file_exists(const std::string& name);

    /**
     * @brief initialize_nn_design Template method pattern which calls init_nn_design() and does post processing
     */
    void initialize_nn_design();
};

/**
 * @brief parse_directory Checks if the directory is empty and appends a "/" if necessary
 * @param directory Directory path which can both be relative or absolute
 * @return string with "/" as suffix
 */
string parse_directory(const string& directory);

/**
 * @brief apply_softmax Applies the softmax activation on a given data array.
 * This method is based on an implementation by "SlayStudy":
 * https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/
 * @param input Data array
 * @param size Length on how many values to apply softmax
 */
void apply_softmax(float* input, size_t size);


#endif // NEURALNETAPI_H
