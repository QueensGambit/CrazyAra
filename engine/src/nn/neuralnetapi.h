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
// TODO: Remove MXNet dependency for interface class
//#include "mxnet-cpp/MxNetCpp.h"
#include "../util/communication.h"

//using namespace mxnet::cpp;
using namespace std;

/**
 * @brief The NeuralNetAPI class is an abstract class for accessing a neural network back-end and to run inference
 */
class NeuralNetAPI
{
protected:
    std::vector<std::string> outputLabels;
    unsigned int batchSize;
    bool isPolicyMap;
    bool enableTensorrt;
    // defines the name for the model based on the loaded .params file
    string modelName;
    // defines the device (e.g. GPU or CPU) and its respective deviceID
    string deviceName;

    // file names for the loaded model and its parameters
    string modelFilePath;
    string paramterFilePath;

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
};

#endif // NEURALNETAPI_H
