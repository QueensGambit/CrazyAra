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
#include "yaml-cpp/yaml.h"
#include <dirent.h>
#include <exception>
#include <string>

// http://www.codebind.com/cpp-tutorial/cpp-program-list-files-directory-windows-linux/
namespace {
std::vector<std::string> GetDirectoryFiles(const std::string& dir) {
  std::vector<std::string> files;
  std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
  struct dirent *dirent_ptr;
  if (!directory_ptr) {
    std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
    return files;
  }

  while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
    files.push_back(std::string(dirent_ptr->d_name));
  }
  return files;
}
}  // namespace

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
    Shape value_label_shape(input_shape[0]);
    Shape policy_label_shape(input_shape[0]);

    args_map["value_label"] = NDArray(value_label_shape, global_ctx, false);
    args_map["policy_label"] = NDArray(policy_label_shape, global_ctx, false);

    net.InferExecutorArrays(global_ctx, &arg_arrays, &grad_arrays, &grad_reqs,
                            &aux_arrays, args_map, std::map<std::string, NDArray>(),
                            std::map<std::string, OpReqType>(), aux_map);
    for (size_t i = 0; i < grad_reqs.size(); ++i) {
        grad_reqs[i] = kNullOp;
    }
//    executor = net.Bind(global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays,
//                                         std::map<std::string, Context>(), nullptr);
    /*end new */
    executor = new Executor(net, global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
//        executor = net.SimpleBind(global_ctx, args_map, std::map<std::string, NDArray>(),
//                                  std::map<std::string, OpReqType>(), aux_map);
    LG << ">>>> Bind successfull! >>>>>>";
}

NeuralNetAPI::NeuralNetAPI(string ctx, unsigned int batchSize, bool selectPolicyFromPlanes, string modelArchitectureDir, string modelWeightsDir)
{
    if (ctx == "cpu") {
        global_ctx = Context::cpu();
    } else if (ctx == "gpu") {
        global_ctx = Context::gpu();
    } else {
        throw "unsupported context " + ctx + " given";
    }

    YAML::Node config = YAML::LoadFile("config.yaml");
    const std::string prefix = config["model_directory"].as<std::string>();

    string jsonFilePath;
    string paramterFilePath;

    const auto& files = GetDirectoryFiles(prefix);
    for (const auto& file : files) {
      cout << file << std::endl;

      size_t pos_json = file.find(".json");
      size_t pos_params = file.find(".params");
      if (pos_json != string::npos) {
          jsonFilePath = prefix + file;
      }
      else if (pos_params != string::npos) {
        paramterFilePath = prefix + file;
      }
    }
    if (jsonFilePath == "" || paramterFilePath == "") {
      throw std::invalid_argument( "The given directory at " + prefix
                                   + " doesn't containa .json and a .parmas file.");
    }

    cout << "json file: " << jsonFilePath << endl;

//    const string prefix = "/media/queensgambit/5C483A84483A5CC8/Deep_Learning/data/stockfish/Crazyhouse/model/";
//                           "/home/queensgambit/Programming/Deep_Learning/CrazyAra_Fish/";
//    const string prefix = "/home/queensgambit/Programming/Deep_Learning/models/risev2/";

//    const string jsonFilePath = prefix + "symbol/model-1.32689-0.566-symbol.json"; //model-1.19246-0.603-symbol.json";
//    const string jsonFilePath = prefix + "symbol/model-1.19246-0.603-symbol.json";

//    const string paramterFilePath = prefix + "params/model-1.32689-0.566-0011.params"; //model-1.19246-0.603-0223.params";
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
    int best_idx = predicted.At(0, 0); //, 0);
    cout << "best_idx: " << best_idx << endl;

    return probOutputs; //best_idx;
}

void NeuralNetAPI::predict(float *inputPlanes, NDArray &valueOutput, NDArray &probOutputs)
{

    NDArray image_data {inputPlanes, input_shape, global_ctx};
//    mtx.lock();
    image_data.CopyTo(&(executor->arg_dict()["data"]));  // crash occurs here!

    // Run the forward pass.
    executor->Forward(false);

    valueOutput = executor->outputs[0].Copy(Context::cpu());
    probOutputs = executor->outputs[1].Copy(Context::cpu());

    // Assign the value output to the return paramter
    valueOutput.WaitToRead();
    probOutputs.WaitToRead();
//    mtx.unlock();

}

