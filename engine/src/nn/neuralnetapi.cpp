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
#include <regex>


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

string get_file_ending_with(const string& dir, const string& suffix) {
    const vector<string> files = get_directory_files(dir);
    const string retString = get_string_ending_with(files, suffix);
    return retString;
}

string get_onnx_model_name(const string& modelDir, int batchSize)
{
    string modelName = get_file_ending_with(modelDir, "-bsize-" + to_string(batchSize) + ".onnx");

    if (modelName  == "") {
        // check for onnx with dynamic shape
        modelName = get_file_ending_with(modelDir, ".onnx");
        if (modelName == "") {
            throw invalid_argument( "The given directory at " + modelDir + " doesn't contain a file ending with " + ".onnx");
        }
        if (modelName.find("-bsize-") != string::npos) {
            throw invalid_argument( "The given directory at " + modelDir + " should either contain an onnx file supporting the current batch size"
                                                                           " or an onnx file with dynamic shape support.");
        }
    }
    return modelName;
}


unsigned int NeuralNetAPI::get_batch_size() const
{
    return batchSize;
}

void NeuralNetAPI::initialize_nn_design()
{
    init_nn_design();
    nbNNInputValues = nnDesign.inputShape.flatten() / batchSize;
    nbNNAuxiliaryOutputs = nnDesign.auxiliaryOutputShape.flatten() / batchSize;
    nbPolicyValues = nnDesign.policyOutputShape.v[1];
    version = read_version_from_string(modelName);
    gamePhase = read_game_phase_from_string(modelDir);
    info_string("Input representation: ", version_to_string(version));
    info_string("Game Phase: ", std::to_string(gamePhase));
}

void NeuralNetAPI::initialize()
{
    load_model();
    initialize_nn_design();
    load_parameters();
    bind_executor();
}

NeuralNetAPI::NeuralNetAPI(const string& ctx, int deviceID, unsigned int batchSize, const string& modelDirectory, bool enableTensorrt):
    deviceID(deviceID),
    batchSize(batchSize),
    enableTensorrt(enableTensorrt),
    modelName(""),
    nbNNInputValues(0),  // will be set dynamically in initialize_nn_design()
    nbNNAuxiliaryOutputs(0),  // will be set dynamically in initialize_nn_design()
    nbPolicyValues(0),  // will be set dynamically in initialize_nn_design()
    version(make_version<0,0,0>()),
    gamePhase(0)
{
    modelDir = parse_directory(modelDirectory);
    deviceName = ctx + string("_") + to_string(deviceID);
}

bool NeuralNetAPI::is_policy_map() const
{
    return nnDesign.isPolicyMap;
}


GamePhase NeuralNetAPI::get_game_phase() const
{
    return gamePhase;
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
    
    check_condition(nnDesign.policyOutputShape.nbDims, 2, "policyOutputShape.nbDims", "2");
    check_condition(nnDesign.valueOutputShape.nbDims, 2, "valueOutputShape.nbDims", "2");
    check_condition(nnDesign.valueOutputShape.v[1], 1, "valueOutputShape.v[1]", "1");
    check_condition(nnDesign.inputShape.nbDims, 4, "inputShape.nbDims", "4");
    check_condition(unsigned(nnDesign.inputShape.v[0]), batchSize, "inputShape.v[0]", "batchSize");
    check_condition(unsigned(nnDesign.inputShape.v[1]), StateConstants::NB_CHANNELS_TOTAL(), "inputShape.v[1]", "StateConstants::NB_CHANNELS_TOTAL()");
    check_condition(unsigned(nnDesign.inputShape.v[2]), StateConstants::BOARD_HEIGHT(), "inputShape.v[2]", "StateConstants::BOARD_HEIGHT()");
    check_condition(unsigned(nnDesign.inputShape.v[3]), StateConstants::BOARD_WIDTH(), "inputShape.v[3]", "StateConstants::BOARD_WIDTH()");
    if (nnDesign.isPolicyMap) {
        check_condition(unsigned(nnDesign.policyOutputShape.v[1]), StateConstants::NB_LABELS_POLICY_MAP(), "neuralNetDesign.policyOutputShape.v[1]", "StateConstants::NB_LABELS_POLICY_MAP()");
    } else {
        check_condition(unsigned(nnDesign.policyOutputShape.v[1]), StateConstants::NB_LABELS(), "neuralNetDesign.policyOutputShape.v[1]", "StateConstants::NB_LABELS()");
    }
    if (nnDesign.hasAuxiliaryOutputs) {
        check_condition(unsigned(nnDesign.auxiliaryOutputShape.v[1]), StateConstants::NB_AUXILIARY_OUTPUTS(), "auxiliaryOutputDims.v[1]", "StateConstants::NB_AUXILIARY_OUTPUTS()");
    }
    else if (StateConstants::NB_AUXILIARY_OUTPUTS() != 0) {
        info_string("No auxiliary outputs detected but auxiliary output was expected.");
        info_string("StateConstants::NB_AUXILIARY_OUTPUTS():", StateConstants::NB_AUXILIARY_OUTPUTS());
    }
    
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

Version read_version_from_string(const string &modelFileName)
{
    // pattern to detect "-v<major>.<minor>"
    const string prefix = "-v";
    const string pattern = "(" + prefix + ")[0-9]+.[0-9]+";

    // regex expression for pattern to be searched
    regex regexp(pattern);

    // flag type for determining the matching behavior (in this case on string objects)
    smatch matches;

    // regex_search that searches pattern regexp in the string
    regex_search(modelFileName, matches, regexp);

    if (matches.size() > 0) {
        for (auto match : matches) {
            if (match.length() > 3) {
                const string content = match;
                const size_t pointPos = content.find(".");
                try {
                const string versionMajor = content.substr(prefix.size(), pointPos-prefix.size());  // skip "-v"
                const string versionMinor = content.substr(pointPos+1);     // skip "."
                    return make_version(std::stoi(versionMajor), std::stoi(versionMinor), 0);
                } catch (const exception& e) {
                    info_string(e.what());
                    break;
                }
            }
        }
    }
    // unsuccessfull
    return make_version<0,0,0>();
}

GamePhase read_game_phase_from_string(const string& modelDir)
{
    // use last char of modelDir and convert to int by subtracting '0'
    // assume phase 0 if last character is not a digit
    char phaseChar = modelDir[modelDir.length() - 2];
    if (!std::isdigit(phaseChar)) {
        return GamePhase(0);
    }
    int gamePhase = phaseChar - '0';
    return GamePhase(gamePhase);
}

void apply_softmax(float* input, size_t size) {

    size_t idx;
    double maximum = -INFINITY;
    for (idx = 0; idx < size; ++idx) {
        if (maximum < input[idx]) {
            maximum = input[idx];
        }
    }

    double sum = 0.0;
    for (idx = 0; idx < size; ++idx) {
        sum += exp(input[idx] - maximum);
    }

    double constant = maximum + log(sum);
    for (idx = 0; idx < size; ++idx) {
        input[idx] = exp(input[idx] - constant);
    }
}
