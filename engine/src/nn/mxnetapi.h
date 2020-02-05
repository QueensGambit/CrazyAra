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
 * @file: mxnetapi.h
 * Created on 05.02.2020
 * @author: queensgambit
 *
 * This file contains wrappers for handling the neural network.
 * Parts of the code are based on the MXNet C++ inference tutorial:
 * https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example/inference
 */

#ifndef MXNETAPI_H
#define MXNETAPI_H

#include "neuralnetapi.h"

/**
 * @brief The MXNetAPI class implements access to the MXNET-C++ back-end for running inference on CPU and GPU.
 */
class MXNetAPI : public NeuralNetAPI
{
private:
    std::mutex mtx;
    std::map<std::string, NDArray> argsMap;
    std::map<std::string, NDArray> auxMap;
    Symbol net;
    Executor *executor;
    Shape inputShape;
    Context globalCtx = Context::cpu();

    void load_model();
    void load_parameters();
    void bind_executor();

    /**
     * @brief infer_select_policy_from_planes Checks if the loaded model encodes the policy as planes
     * and sets the selectPolicyFromPlane boolean accordingly
     */
    void check_if_policy_map();

    /**
     * @brief SplitParamMap Splits loaded param map into arg parm and aux param with target context
     * @param paramMap Parameter map
     * @param argParamInTargetContext Output intermediate parameter map
     * @param auxParamInTargetContext Output intermediate auxiliary map
     * @param targetContext Computation context e.g. Context::cpu(), Context::gpu()
     */
    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *argParamInTargetContext,
        std::map<std::string, NDArray> *auxParamInTargetContext,
        Context targetContext);

    /**
     * @brief ConvertParamMapToTargetContext Copies the param map into the target context
     * @param paramMap Parameter map
     * @param paramMapInTargetContext Output parameter map
     * @param targetContext Computation context e.g. Context::cpu(), Context::gpu()
     */
    void ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *paramMapInTargetContext,
        Context targetContext);

public:
    MXNetAPI(const string& ctx, int deviceID, unsigned int miniBatchSize, const string& modelDirectory, bool tensorRT);
    ~MXNetAPI();

    NDArray predict(float* inputPlanes, float& value);
    void predict(float* inputPlanes, NDArray& valueOutput, NDArray& probOutputs);
};

#endif // MXNETAPI_H
