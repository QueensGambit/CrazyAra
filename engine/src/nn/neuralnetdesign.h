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
 * @file: neuralnetdesign.h
 * Created on 26.06.2021
 * @author: queensgambit
 *
 * The NeuralNetDesign struct stores information about the neural network design.
 * It is supposed to be loaded dynamically from a neural network architecture file via the method `NeuralNetAPI->init_nn_design()`.
 */

#ifndef NEURALNETDESIGN_H
#define NEURALNETDESIGN_H

#include <string>
using namespace std;


namespace nn_api {
/**
 * @brief The Shape struct is a basic shape container object.
 */
struct Shape {
    int nbDims = -1;  // uninitialized
    int v[8];         // shape dimensions

    /**
     * @brief flatten Returns the flattened shape dimension
     * @return -1 if not initialized else product of all dimensions
     */
    int flatten() const;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

/**
 * @brief The NeuralNetDesign struct stores information about the neural network design.
 * It is supposed to be loaded dynamically from a neural network architecture file via the method `NeuralNetAPI->init_nn_design()`.
 */
struct NeuralNetDesign {
    bool isPolicyMap = false;
    bool hasAuxiliaryOutputs = false;
    const int nbInputs = 1;
    const string inputLayerName = "data";
    string policyOutputName = "policy_out";  // may be adjusted using "policyOutputIdx" if not found
    const string policySoftmaxOutputName = "policy_softmax";
    string valueOutputName = "value_out";  // may be adjusted using "valueOutputIdx" if not found
    const string auxiliaryOutputName = "auxiliary_out";
    const int inputIdx = 0;
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


#endif // NEURALNETDESIGN_H
