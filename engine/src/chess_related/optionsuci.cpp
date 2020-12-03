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
 * @file: optionsuci.cpp
 * Created on 13.07.2019
 * @author: queensgambit
 */

#include "optionsuci.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;

void on_logger(const Option& o) {
    start_logger(o);
}

void OptionsUCI::init(OptionsMap &o)
{
#ifdef MODE_CRAZYHOUSE
    o["UCI_Variant"]                   << Option("crazyhouse", {"crazyhouse"});
#elif defined MODE_LICHESS
    o["UCI_Variant"]                   << Option(availableVariants.front().c_str(), availableVariants);
#else  // MODE = MODE_CHESS
    o["UCI_Variant"]                   << Option("chess", {"chess"});
#endif
    o["MultiPV"]                       << Option(1, 1, 99999);
    o["Search_Type"]                   << Option("mcts", {"mcts"});
    o["Context"]                       << Option("gpu", {"cpu", "gpu"});
    o["First_Device_ID"]               << Option(0, 0, 99999);
    o["Last_Device_ID"]                << Option(0, 0, 99999);
#ifdef USE_RL
    o["Batch_Size"]                    << Option(8, 1, 8192);
#else
    o["Batch_Size"]                    << Option(16, 1, 8192);
#endif
    o["Threads"]                       << Option(2, 1, 512);
    o["Centi_CPuct_Init"]              << Option(250, 1, 99999);
    o["CPuct_Base"]                    << Option(19652, 1, 99999);
#ifdef USE_RL
    o["Centi_Dirichlet_Epsilon"]       << Option(25, 0, 99999);
#else
    o["Centi_Dirichlet_Epsilon"]       << Option(0, 0, 99999);
#endif
    o["Centi_Dirichlet_Alpha"]         << Option(20, 1, 99999);
//    o["Centi_U_Init"]                  << Option(100, 0, 100);         currently disabled
//    o["Centi_U_Min"]                   << Option(100, 0, 100);         currently disabled
//    o["U_Base"]                        << Option(1965, 0, 99999);      currently disabled
    o["Centi_U_Init_Divisor"]          << Option(100, 1, 99999);
    o["Centi_Q_Value_Weight"]          << Option(0, 0, 99999);
    o["Centi_Q_Thresh_Init"]           << Option(50, 0, 100);
    o["Centi_Q_Thresh_Max"]            << Option(90, 0, 100);
    o["Q_Thresh_Base"]                 << Option(1965, 0, 99999);
    o["Max_Search_Depth"]              << Option(99, 1, 99999);
#ifdef USE_RL
    o["Centi_Temperature"]             << Option(80, 0, 99999);
#else
    o["Centi_Temperature"]             << Option(170, 0, 99999);
#endif
    o["Temperature_Moves"]             << Option(0, 0, 99999);
#ifdef USE_RL
    o["Centi_Quantile_Clipping"]       << Option(0, 0, 100);
#else
    o["Centi_Quantile_Clipping"]       << Option(25, 0, 100);
#endif
    o["Centi_Temperature_Decay"]       << Option(92, 0, 100);
    o["Centi_Node_Temperature"]        << Option(170, 1, 99999);
    o["Centi_Virtual_Loss"]            << Option(100, 0, 99999);
#ifdef USE_RL
    o["Nodes"]                         << Option(800, 0, 99999999);
#else
    o["Nodes"]                         << Option(0, 0, 99999999);
#endif
    o["Allow_Early_Stopping"]          << Option(true);
    o["Use_Raw_Network"]               << Option(false);
//    o["Enhance_Checks"]                << Option(false);         currently disabled
//    o["Enhance_Captures"]              << Option(false);         currently disabled
    o["Use_Transposition_Table"]       << Option(true);
#ifdef TENSORRT
    o["Use_TensorRT"]                  << Option(true);
    o["Precision"]                     << Option("float16", {"float32", "float16", "int8"});
#endif
#ifdef MODE_CRAZYHOUSE
    o["Model_Directory"]               << Option("model");
#else
    o["Model_Directory"]               << Option("model");
#endif
#ifdef USE_RL
    o["Model_Directory_Contender"]     << Option("model_contender/");
    o["Selfplay_Number_Chunks"]        << Option(640, 1, 99999);
    o["Selfplay_Chunk_Size"]           << Option(128, 1, 99999);
    o["Centi_Raw_Prob_Temperature"]    << Option(25, 0, 100);
    o["Milli_Policy_Clip_Thresh"]      << Option(0, 0, 100);
    o["MeanInitPly"]                   << Option(15, 0, 99999);
    o["MaxInitPly"]                    << Option(30, 0, 99999);
    o["Quick_Nodes"]                   << Option(100, 0, 99999);
    o["Centi_Quick_Probability"]       << Option(0, 0, 100);
    o["Centi_Quick_Q_Value_Weight"]    << Option(70, 0, 99999);
    o["Centi_Quick_Dirichlet_Epsilon"] << Option(0, 0, 99999);
    o["Centi_Node_Random_Factor"]      << Option(10, 0, 100);
    o["Centi_Resign_Probability"]      << Option(90, 0, 100);
    o["Centi_Resign_Threshold"]        << Option(-90, -100, 100);
    o["Reuse_Tree"]                    << Option(false);
#endif
    o["Move_Overhead"]                 << Option(50, 0, 5000);
    o["Centi_Random_Move_Factor"]      << Option(0, 0, 99);
    o["SyzygyPath"]                    << Option("<empty>");
    o["Log_File"]                      << Option("", on_logger);
    o["Use_NPS_Time_Manager"]          << Option(true);
    o["Use_Advantage"]                 << Option(false);
#ifdef SUPPORT960
    o["UCI_Chess960"]                  << Option(true);
#endif
    o["Random_Playout"]                << Option(true);
    o["Fixed_Movetime"]                << Option(0, 0, 99999999);
    o["Reuse_Tree"]                    << Option(true);
}

void OptionsUCI::setoption(istringstream &is)
{

    string token, name, value;
    is >> token; // Consume "name" token

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (Options.find(name) != Options.end()) {
        Options[name] = value;
        cout << "info string Updated option " << name << " to " << value << endl;
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name == "uci_variant") {
            Variant variant = UCI::variant_from_name(value);
            cout << "info string variant " << (string)Options["UCI_Variant"] << " startpos " << StartFENs[variant] << endl;
        }
    }
    else {
        cout << "info string Given option " << name << " does not exist " << endl;
    }
}
