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
 * @file: neuralnetdesign.cpp
 * Created on 26.06.2021
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "neuralnetdesign.h"
#include <sstream>
#include "util/communication.h"

using namespace nn_api;

void nn_api::NeuralNetDesign::print() const
{
   std::stringstream ssInputDims;
   ssInputDims << inputShape;
   std::stringstream ssValueOutputDims;
   ssValueOutputDims << valueOutputShape;
   std::stringstream ssPolicyOutputDims;
   ssPolicyOutputDims << policyOutputShape;

   info_string("inputDims:", ssInputDims.str());
   info_string("valueOutputDims:", ssValueOutputDims.str());
   info_string("policyOutputDims:", ssPolicyOutputDims.str());
   if (hasAuxiliaryOutputs) {
       std::stringstream ssAuxiliaryOutputDims;
       ssAuxiliaryOutputDims << auxiliaryOutputShape;
       info_string("auxiliaryOutputDims:", ssAuxiliaryOutputDims.str());
       return;
   }
   info_string("No auxiliary outputs detected.");
}

int nn_api::Shape::flatten() const
{
    if (nbDims == -1) {
        return -1;
    }
    int product = 1;
    for (int idx = 0; idx < nbDims; ++idx) {
        product *= v[idx];
    }
    return product;
}
