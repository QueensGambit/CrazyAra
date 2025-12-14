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
 *
 * This file defines an abstract for a class which uses a NeuralNetAPI.
 */

#ifndef NEURALNETAPIUSER_H
#define NEURALNETAPIUSER_H

#include "neuralnetapi.h"

/**
 * @brief The NeuralNetAPIUser class is a utility class which handles memory allocation and de-allocation.
 * The results of NN-inference are stored in valueOutputs and probOutputs.
 */
class NeuralNetAPIUser
{
public:
    vector<NeuralNetAPI*> nets; // vector of net objects
    unsigned int numPhases;
    std::map<GamePhase, int> phaseToNetsIndex;  // maps a GamePhase to the index of the net that should be used

    // inputPlanes stores the plane representation of all newly expanded nodes of a single mini-batch
    float* inputPlanes;
    // stores the corresponding value-Outputs and probability-Outputs of the nodes stored in the vector "newNodes"
    // sufficient memory according to the batch-size will be allocated in the constructor
    float* valueOutputs;
    float* probOutputs;
    float* auxiliaryOutputs;

public:
    NeuralNetAPIUser(const vector<unique_ptr<NeuralNetAPI>>& netsNew);
    NeuralNetAPIUser(const vector<NeuralNetAPI*> netsNew);
    ~NeuralNetAPIUser();
    NeuralNetAPIUser(NeuralNetAPIUser&) = delete;

    /**
     * @brief run_inference Runs inference of the allocated input planes for a given number of iterations.
     * Output of the inference is ignored.
     * @param iterations Number of iterations to run
     */
    void run_inference(uint_fast16_t iterations);

    /**
     * @brief get_num_phases Returns the number of phases
     * @return numPhases
     */
    unsigned int get_num_phases() const;

    /**
     * @brief init_members Initialize member and allocates memory.
     */
    void init_members();
};

#endif // NEURALNETAPIUSER_H
