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
 * @file: neuralnetapi.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "neuralnetapi.h"
#include <string>
#include "../stateobj.h"


string get_string_ending_with(const vector<string>& stringVector, const string& suffix) {
    for (const string& curString : stringVector) {
        if (has_suffix(curString, suffix)) {
            return curString;
        }
    }
    return "";
}

vector<string> get_items_by_elment(const vector<string> &stringVector, const string &targetString, bool shouldContain)
{
    vector<string> returnVector;
    for (const string& curString : stringVector) {
        if ((curString.find(targetString) == std::string::npos) == !shouldContain) {
            returnVector.emplace_back(curString);
        }
    }
    return returnVector;
}


unsigned int NeuralNetAPI::get_batch_size() const
{
    return batchSize;
}

NeuralNetAPI::NeuralNetAPI(const string& ctx, int deviceID, unsigned int batchSize, const string& modelDirectory, bool enableTensorrt):
    deviceID(deviceID),
    batchSize(batchSize),
    enableTensorrt(enableTensorrt),
    modelName("")
{
    modelDir = parse_directory(modelDirectory);
    deviceName = ctx + string("_") + to_string(deviceID);
}

bool NeuralNetAPI::is_policy_map() const
{
    return nnDesign.isPolicyMap;
}

string NeuralNetAPI::get_model_name() const
{
    return modelName;
}

string NeuralNetAPI::get_device_name() const
{
    return deviceName;
}

void NeuralNetAPI::validate_neural_network()
{
    nnDesign.print();
    assert_condition(nnDesign.policyOutputShape.nbDims, 2, "policyOutputShape.nbDims", "2");
    assert_condition(nnDesign.valueOutputShape.nbDims, 2, "valueOutputShape.nbDims", "2");
    assert_condition(nnDesign.valueOutputShape.v[1], 1, "valueOutputShape.v[1]", "1");
    assert_condition(nnDesign.inputShape.nbDims, 4, "inputShape.nbDims", "4");
    assert_condition(unsigned(nnDesign.inputShape.v[1]), StateConstants::NB_CHANNELS_TOTAL(), "inputShape.v[1]", "StateConstants::NB_CHANNELS_TOTAL()");
    assert_condition(unsigned(nnDesign.inputShape.v[2]), StateConstants::BOARD_HEIGHT(), "inputShape.v[2]", "StateConstants::BOARD_HEIGHT()");
    assert_condition(unsigned(nnDesign.inputShape.v[3]), StateConstants::BOARD_WIDTH(), "inputShape.v[3]", "StateConstants::BOARD_WIDTH()");
    if (nnDesign.isPolicyMap) {
        assert_condition(unsigned(nnDesign.policyOutputShape.v[1]), StateConstants::NB_LABELS_POLICY_MAP(), "neuralNetDesign.policyOutputShape.v[1]", "StateConstants::NB_LABELS_POLICY_MAP()");
    } else {
        assert_condition(unsigned(nnDesign.policyOutputShape.v[1]), StateConstants::NB_LABELS(), "neuralNetDesign.policyOutputShape.v[1]", "StateConstants::NB_LABELS()");
    }
    if (nnDesign.hasAuxiliaryOutputs) {
        assert_condition(unsigned(nnDesign.auxiliaryOutputShape.v[1]), StateConstants::NB_AUXILIARY_OUTPUTS(), "auxiliaryOutputDims.v[1]", "StateConstants::NB_AUXILIARY_OUTPUTS()");
    }
    else if (StateConstants::NB_AUXILIARY_OUTPUTS() != 0) {
        std::cerr << "StateConstants::NB_AUXILIARY_OUTPUTS(): " << StateConstants::NB_AUXILIARY_OUTPUTS() << endl;
        throw "No auxiliary outputs detected but auxiliary output was expected.";
    }
}

unsigned int NeuralNetAPI::get_policy_output_length() const
{
    return nnDesign.policyOutputShape.v[1] * batchSize;
}

bool NeuralNetAPI::file_exists(const string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

string parse_directory(const string& directory)
{
    if (directory.length() == 0) {
        throw invalid_argument("The given directory must not be empty.");
    }
    if (directory[directory.length()-1] != '/') {
        return directory + "/";
    }
    return directory;
}

ostream& nn_api::operator<<(ostream &os, const nn_api::Shape &shape)
{
    os << "(";
    for (int idx = 0; idx < shape.nbDims-1; ++idx) {
        os << shape.v[idx] << ", ";
    }
    if (shape.nbDims > 0) {
        os << shape.v[shape.nbDims-1];
    }
    os << ")";
    return os;
}

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
