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
 * @file: neuralnetapiuser.h
 * Created on 06.10.2020
 * @author: queensgambit
 */

#include "neuralnetapiuser.h"
#include "stateobj.h"
#ifdef TENSORRT
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"
#endif

NeuralNetAPIUser::NeuralNetAPIUser(NeuralNetAPI* netGating, const vector<unique_ptr<NeuralNetAPI>>& netsNew) :
    netGating(netGating),
    auxiliaryOutputs(nullptr)
{
    for (size_t idx = 0; idx < netsNew.size(); idx++) {
        nets.push_back(netsNew[idx].get());
    }
    numPhases = nets.size();
    for (unsigned int i = 0; i < numPhases; i++)
    {
        GamePhase phaseOfNetI = nets[i]->get_game_phase();
        assert(phaseOfNetI < numPhases); // no net should have a phase greater or equal to the total amount of nets (assumes that only phases from 0 to numPhases -1 are possible)
        assert(phaseToNetsIndex.count(phaseOfNetI) == 0); // no net should have the same phase as another net
        phaseToNetsIndex[phaseOfNetI] = i;
    }
    
    // allocate memory for all predictions and results
#ifdef TENSORRT
#ifdef DYNAMIC_NN_ARCH
    CHECK(cudaMallocHost((void**) &inputPlanes, nets.front()->get_batch_size() * nets.front()->get_nb_input_values_total() * sizeof(float)));
#else
     CHECK(cudaMallocHost((void**) &inputPlanes, nets.front()->get_batch_size() * StateConstants::NB_VALUES_TOTAL() * sizeof(float)));
#endif
    CHECK(cudaMallocHost((void**) &phaseOutputs, nets.front()->get_batch_size() * numPhases * sizeof(float)));
    CHECK(cudaMallocHost((void**) &valueOutputs, nets.front()->get_batch_size() * sizeof(float)));
    CHECK(cudaMallocHost((void**) &probOutputs, nets.front()->get_batch_size() * nets.front()->get_nb_policy_values() * sizeof(float)));
    if (nets.front()->has_auxiliary_outputs()) {
        CHECK(cudaMallocHost((void**) &auxiliaryOutputs, nets.front()->get_batch_size() * nets.front()->get_nb_auxiliary_outputs() * sizeof(float)));
    }
#else
    phaseOutputs = new float[nets.front()->get_batch_size() * numPhases];
    inputPlanes = new float[nets.front()->get_batch_size() * nets.front()->get_nb_input_values_total()];
    valueOutputs = new float[nets.front()->get_batch_size()];
    probOutputs = new float[nets.front()->get_batch_size() * nets.front()->get_nb_policy_values()];
#ifdef DYNAMIC_NN_ARCH
    if (nets.front()->has_auxiliary_outputs()) {
        auxiliaryOutputs = new float[nets.front()->get_batch_size() * nets.front()->get_nb_auxiliary_outputs()];
    }
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
         auxiliaryOutputs = new float[nets.front()->get_batch_size() * StateConstants::NB_AUXILIARY_OUTPUTS()];
    }
#endif
#endif
}

NeuralNetAPIUser::~NeuralNetAPIUser()
{
#ifdef TENSORRT
    CHECK(cudaFreeHost(inputPlanes));
    CHECK(cudaFreeHost(phaseOutputs));
    CHECK(cudaFreeHost(valueOutputs));
    CHECK(cudaFreeHost(probOutputs));
#ifdef DYNAMIC_NN_ARCH
    if (nets.front()->has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
        CHECK(cudaFreeHost(auxiliaryOutputs));
    }
#else
    delete [] inputPlanes;
    delete [] phaseOutputs;
    delete [] valueOutputs;
    delete [] probOutputs;
#ifdef DYNAMIC_NN_ARCH
    if (nets.front()->has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
        delete [] auxiliaryOutputs;
    }
#endif
}

void NeuralNetAPIUser::run_inference(uint_fast16_t iterations)
{
    for (uint_fast16_t it = 0; it < iterations; ++it) {
        nets.front()->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
    }
}

unsigned int NeuralNetAPIUser::get_num_phases() const
{
    return numPhases;
}

void NeuralNetAPIUser::predict(bool useGatingNetwork, size_t majorityNNIndex, size_t batchSize)
{
    if (useGatingNetwork) {
        // query the gating network to check how to combine the network outputs (only the phase output will be written here)
        netGating->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
        blaze::DynamicMatrix<float> phaseOutputsGatingNet(batchSize, 3, phaseOutputs);
        // cout << "phaseOutputs:" << phaseOutputs[0] << " "<< phaseOutputs[1] << " " << phaseOutputs[2] << " ";
        nets[phaseToNetsIndex[0]]->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
        blaze::DynamicVector<float> valueOutputsNet0(batchSize, valueOutputs);
        blaze::DynamicMatrix<float> probOutputsNet0(batchSize, nets.front()->get_nb_policy_values(), probOutputs);
        nets[phaseToNetsIndex[1]]->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
        blaze::DynamicVector<float> valueOutputsNet1(batchSize, valueOutputs);
        blaze::DynamicMatrix<float> probOutputsNet1(batchSize, nets.front()->get_nb_policy_values(), probOutputs);
        nets[phaseToNetsIndex[2]]->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
        blaze::DynamicVector<float> valueOutputsNet2(batchSize, valueOutputs);
        blaze::DynamicMatrix<float> probOutputsNet2(batchSize, nets.front()->get_nb_policy_values(), probOutputs);
        // combine the outputs of multiple neural networks and copy back the data
        blaze::DynamicVector<float> combinedValueOutputs(batchSize);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            valueOutputs[batchIdx] = phaseOutputsGatingNet.at(batchIdx, 0) * valueOutputsNet0.at(batchIdx) + phaseOutputsGatingNet.at(batchIdx, 1) * valueOutputsNet1.at(batchIdx) + phaseOutputsGatingNet.at(batchIdx, 2) * valueOutputsNet2.at(batchIdx);
        }
        blaze::DynamicMatrix<float> combinedProbOutputs(batchSize, nets.front()->get_nb_policy_values());
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            for (int colIdx = 0; colIdx < nets.front()->get_nb_policy_values(); ++colIdx) {
                probOutputs[batchIdx * nets.front()->get_nb_policy_values() + colIdx] = phaseOutputsGatingNet.at(batchIdx, 0) * probOutputsNet0.at(batchIdx, colIdx) + phaseOutputsGatingNet.at(batchIdx, 1) * probOutputsNet1.at(batchIdx, colIdx) + phaseOutputsGatingNet.at(batchIdx, 2) * probOutputsNet2.at(batchIdx, colIdx);
            }
        }
    }

    if (!useGatingNetwork) {
        netGating->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
        blaze::DynamicMatrix<float> phaseOutputsGatingNet(batchSize, 3, phaseOutputs);
        cout << "phaseOutputs:" << phaseOutputs[0] << " "<< phaseOutputs[1] << " " << phaseOutputs[2] << " ";
        // query the network that corresponds to the majority phase
        nets[majorityNNIndex]->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs, phaseOutputs);
    }
}

