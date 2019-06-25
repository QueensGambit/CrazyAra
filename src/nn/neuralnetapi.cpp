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
 * @file: neuralnetapi.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "neuralnetapi.h"
#include "../domain/crazyhouse/constants.h"

void NeuralNetAPI::loadModel(const string &jsonFilePath)
{
    if (!FileExists(jsonFilePath)) {
      LG << "Model file " << jsonFilePath << " does not exist";
      throw std::runtime_error("Model file does not exist");
    }
    LG << "Loading the model from " << jsonFilePath << std::endl;
    net = Symbol::Load(jsonFilePath);
}

void NeuralNetAPI::loadParameters(const std::string& paramterFilePath) {
  if (!FileExists(paramterFilePath)) {
    LG << "Parameter file " << paramterFilePath << " does not exist";
    throw std::runtime_error("Model parameters does not exist");
  }
  LG << "Loading the model parameters from " << paramterFilePath << std::endl;
  std::map<std::string, NDArray> parameters;
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

void NeuralNetAPI::bindExecutor() //Shape *input_shape_single, Executor* executor_single)
{
    // Create an executor after binding the model to input parameters.
    args_map["data"] = NDArray(input_shape, global_ctx, false);
    /* new */
    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> grad_arrays;
    std::vector<OpReqType> grad_reqs;
    std::vector<NDArray> aux_arrays;

    net.InferExecutorArrays(global_ctx, &arg_arrays, &grad_arrays, &grad_reqs,
                            &aux_arrays, args_map, std::map<std::string, NDArray>(),
                            std::map<std::string, OpReqType>(), aux_map);
    for (size_t i = 0; i < grad_reqs.size(); ++i) {
        grad_reqs[i] = kNullOp;
    }
    executor = net.Bind(global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays,
                                         std::map<std::string, Context>(), nullptr);
//    executor = net.SimpleBind(global_ctx, args_map, std::map<std::string, NDArray>(),
//                              std::map<std::string, OpReqType>(), aux_map);

    /*end new */
    LG << ">>>> Bind successfull! >>>>>>";
}

NeuralNetAPI::NeuralNetAPI(string ctx, unsigned int batchSize, bool selectPolicyFromPlanes, string modelArchitectureDir, string modelWeightsDir)
{
    const string prefix = "/home/queensgambit/Programming/Deep_Learning/CrazyAra_Fish/";
//    const string prefix = "/home/queensgambit/Programming/Deep_Learning/models/risev2/";

    const string jsonFilePath = prefix + "symbol/model-1.32689-0.566-symbol.json"; //model-1.19246-0.603-symbol.json";
//    const string jsonFilePath = prefix + "symbol/model-1.19246-0.603-symbol.json";

    const string paramterFilePath = prefix + "params/model-1.32689-0.566-0011.params"; //model-1.19246-0.603-0223.params";
//    const string paramterFilePath = prefix + "params/model-1.19246-0.603-0223.params";

    input_shape =  Shape(batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH);
//    input_shape =  Shape(batchSize, NB_CHANNELS_FULL, BOARD_HEIGHT, BOARD_WIDTH);

    loadModel(jsonFilePath);
    loadParameters(paramterFilePath);
    bindExecutor(); //&input_shape_single, executor_single);
//    bindExecutor(&input_shape, executor);
}

NDArray NeuralNetAPI::predict(float *inputPlanes, float &value)
{
    // populates v vector data in a matrix of 1 row and 4 columns
     NDArray image_data {inputPlanes, input_shape, global_ctx};

//    std::cout << "image data" << image_data << std::endl;
    image_data.CopyTo(&(executor->arg_dict()["data"]));

    // Run the forward pass.
    executor->Forward(false);

    // The output is available in executor->outputs.
//    auto valueArray = executor->outputs[0].Copy(Context::cpu());
//    auto probArray = executor->outputs[1].Copy(Context::cpu());

    auto valueOutput = executor->outputs[0].Copy(Context::cpu());
    auto probOutputs = executor->outputs[1].Copy(Context::cpu());

    // Assign the value output to the return paramter
    valueOutput.WaitToRead();
    cout << "valueOutput.shape" << valueOutput.Size() << " " << valueOutput << endl;
    value = valueOutput.At(0, 0);
    cout << "value " << value << endl;

    probOutputs.WaitToRead();

    auto predicted = probOutputs.ArgmaxChannel();
    cout << "probOutputs" << probOutputs << endl;
    predicted.WaitToRead();
    cout << "predicted" << predicted << endl;
    int best_idx = predicted.At(0); //, 0);
    cout << "best_idx: " << best_idx << endl;


//    const float *data = probArray.GetData();
    // TODO: Find a faster way for copying this
//    prob_vec = Eigen::VectorXf(data, NB_LABELS);
//    Eigen::Map<Eigen::VectorXf> prob_vec(data, NB_LABELS);

//    std::vector<float> a = {1, 2, 3, 4};
//    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(a.data(), a.size());

//    prob_vec= Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(a.data(), a.size());

//    for (int i = 0; i < NB_LABELS; ++i) {
//        policyProb[i] =  probArray.At(0, i);
//    }

    //https://stackoverflow.com/questions/45328600/passing-values-of-vectors-to-eigen-library-format
    // https://stackoverflow.com/questions/17036818/initialise-eigenvector-with-stdvector
//auto b = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(a.data(), NB_LABELS);

//    float best_accuracy = probArray.At(0, best_idx);
//    std::cout << "best_accuracy" << best_accuracy << std::endl;

//    cout << "best_idx: " << best_idx << endl;

    return probOutputs; //best_idx;
}

void NeuralNetAPI::predict(float *inputPlanes, NDArray &valueOutput, NDArray &probOutputs)
{
    // populates v vector data in a matrix of 1 row and 4 columns
     NDArray image_data {inputPlanes, input_shape, global_ctx};

//    std::cout << "image data" << image_data << std::endl;
    image_data.CopyTo(&(executor->arg_dict()["data"]));

    // Run the forward pass.
    executor->Forward(false);

    // The output is available in executor->outputs.
//    auto valueArray = executor->outputs[0].Copy(Context::cpu());
//    auto probArray = executor->outputs[1].Copy(Context::cpu());

    valueOutput = executor->outputs[0].Copy(Context::cpu());
    probOutputs = executor->outputs[1].Copy(Context::cpu());

    // Assign the value output to the return paramter
    valueOutput.WaitToRead();
//    value = valueArray.At(0, 0);

    probOutputs.WaitToRead();

//    const float *data = probArray.GetData();
    // TODO: Find a faster way for copying this
//    prob_vec = Eigen::VectorXf(data, NB_LABELS);
//    Eigen::Map<Eigen::VectorXf> prob_vec(data, NB_LABELS);

//    std::vector<float> a = {1, 2, 3, 4};
//    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(a.data(), a.size());

//    prob_vec= Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(a.data(), a.size());

//    for (int i = 0; i < NB_LABELS; ++i) {
//        policyProb[i] =  probArray.At(0, i);
//    }

    //https://stackoverflow.com/questions/45328600/passing-values-of-vectors-to-eigen-library-format
    // https://stackoverflow.com/questions/17036818/initialise-eigenvector-with-stdvector
//auto b = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(a.data(), NB_LABELS);

//    float best_accuracy = probArray.At(0, best_idx);
//    std::cout << "best_accuracy" << best_accuracy << std::endl;

//    cout << "best_idx: " << best_idx << endl;

    //    return probArray; //best_idx;
}

