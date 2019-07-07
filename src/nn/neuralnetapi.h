/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: neuralnetapi.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * This file contains wrappers for handling the neural network.
 */

#ifndef NEURALNETAPI_H
#define NEURALNETAPI_H

#include <iostream>
using namespace std;
#include <sys/stat.h>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

class NeuralNetAPI
{
private:
    std::mutex mtx;
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::vector<std::string> output_labels;
    Symbol net;
    Executor *executor;
    Shape input_shape;
    Context global_ctx = Context::cpu();

    /**
     * @brief FileExists Function to check if a file exists in a given path
     * @param name Filepath
     * @return True if exists else false
     */
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    /**
     * @brief loadModel Loads the model architecture definition from a json file
     * @param model_json_file JSON-Path to the json file
     */
    void loadModel(const std::string& jsonFilePath);

    /**
     * @brief loadParameters Loads the parameters / weights of the model given a parameter path
     * @param model_parameters_file Parameter file path
     */
    void loadParameters(const std::string& paramterFilePath);

    /**
     * @brief bindExecutor Binds the executor object to the neural network
     */
    void bindExecutor(); //Shape &input_shape, Executor* executor);

public:
    NeuralNetAPI(string ctx, unsigned int batchSize, bool selectPolicyFromPlanes, string modelArchitectureDir, string modelWeightsDir);
    NDArray predict(float *inputPlanes, float &value);
    void predict(float *input_planes, NDArray &valueOutput, NDArray &probOutputs);
};

#endif // NEURALNETAPI_H
