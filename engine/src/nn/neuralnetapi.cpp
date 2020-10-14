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


string get_file_ending_with(const string& dir, const string& suffix) {
    const vector<string>& files = get_directory_files(dir);
    for (const string& file : files) {
        if (has_suffix(file, suffix)) {
            return file;
        }
    }
    throw invalid_argument( "The given directory at " + dir + " doesn't contain a file ending with " + suffix);
    return "";
}


unsigned int NeuralNetAPI::get_batch_size() const
{
    return batchSize;
}

NeuralNetAPI::NeuralNetAPI(const string& ctx, int deviceID, unsigned int batchSize, const string& modelDirectory, bool enableTensorrt):
    deviceID(deviceID),
    batchSize(batchSize),
    policyOutputLength(StateConstants::NB_LABELS() * batchSize),
    enableTensorrt(enableTensorrt)
{
    modelDir = parse_directory(modelDirectory);
    deviceName = ctx + string("_") + to_string(deviceID);
}

bool NeuralNetAPI::is_policy_map() const
{
    return isPolicyMap;
}

string NeuralNetAPI::get_model_name() const
{
    return modelName;
}

string NeuralNetAPI::get_device_name() const
{
    return deviceName;
}

unsigned int NeuralNetAPI::get_policy_output_length() const
{
    return policyOutputLength;
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
