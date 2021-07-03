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
#ifdef MODE_XIANGQI
    #include "variant.h"
#endif
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstring>
#include "customlogger.h"
#include "syzygy/tbprobe.h"
#include "../util/communication.h"
#include "../nn/neuralnetapi.h"

using namespace std;

void on_logger(const Option& o) {
    CustomLogger::start(o, ifstream::app);
}

// method is based on 3rdparty/Stockfish/uci.cpp
#ifndef MODE_XIANGQI
void on_tb_path(const Option& o) {
    Tablebases::init(UCI::variant_from_name(Options["UCI_Variant"]), Options["SyzygyPath"]);
}
#endif

void OptionsUCI::init(OptionsMap &o)
{
    o["Allow_Early_Stopping"]          << Option(true);
#ifdef USE_RL
    o["Batch_Size"]                    << Option(8, 1, 8192);
#else
    o["Batch_Size"]                    << Option(16, 1, 8192);
#endif
    o["Centi_CPuct_Init"]              << Option(250, 1, 99999);
#ifdef USE_RL
    o["Centi_Dirichlet_Epsilon"]       << Option(25, 0, 99999);
#else
    o["Centi_Dirichlet_Epsilon"]       << Option(0, 0, 99999);
#endif
    o["Centi_Dirichlet_Alpha"]         << Option(20, 1, 99999);
    o["Centi_Epsilon_Checks"]          << Option(1, 0, 100);
    o["Centi_Epsilon_Greedy"]          << Option(5, 0, 100);
//    o["Centi_U_Init"]                  << Option(100, 0, 100);         currently disabled
//    o["Centi_U_Min"]                   << Option(100, 0, 100);         currently disabled
//    o["U_Base"]                        << Option(1965, 0, 99999);      currently disabled
    o["Centi_Node_Temperature"]        << Option(170, 1, 99999);
    o["Centi_Q_Value_Weight"]          << Option(100, 0, 99999);
    o["Centi_Q_Veto_Delta"]            << Option(40, 0, 99999);
#ifdef USE_RL
    o["Centi_Quantile_Clipping"]       << Option(0, 0, 100);
#else
    o["Centi_Quantile_Clipping"]       << Option(25, 0, 100);
#endif
    o["Centi_Random_Move_Factor"]      << Option(0, 0, 99);
#ifdef USE_RL
    o["Centi_Temperature"]             << Option(80, 0, 99999);
#else
    o["Centi_Temperature"]             << Option(170, 0, 99999);
#endif
    o["Centi_Temperature_Decay"]       << Option(92, 0, 100);
    o["Centi_U_Init_Divisor"]          << Option(100, 1, 99999);
    o["Centi_Virtual_Loss"]            << Option(100, 0, 99999);
#if defined(MXNET) && defined(TENSORRT)
    o["Context"]                       << Option("gpu", {"cpu", "gpu"});
#elif defined (TORCH)
    o["Context"]                       << Option("gpu", {"cpu", "gpu"});
#elif defined (TENSORRT)
    o["Context"]                       << Option("gpu", {"gpu"});
#else
    o["Context"]                       << Option("cpu");
#endif
    o["CPuct_Base"]                    << Option(19652, 1, 99999);
//    o["Enhance_Captures"]              << Option(false);         currently disabled
    o["First_Device_ID"]               << Option(0, 0, 99999);
    o["Fixed_Movetime"]                << Option(0, 0, 99999999);
    o["Last_Device_ID"]                << Option(0, 0, 99999);
    o["Log_File"]                      << Option("", on_logger);
    o["MCTS_Solver"]                   << Option(true);
#ifdef MODE_CRAZYHOUSE
    o["Model_Directory"]               << Option("model/crazyhouse");
#elif defined MODE_LICHESS
    o["Model_Directory"]               << Option((string("model/") + get_first_variant_with_model()).c_str());
#elif defined MODE_CHESS
    o["Model_Directory"]               << Option("model/chess");
#elif defined MODE_XIANGQI
    o["Model_Directory"]               << Option("model/xiangqi");
#else
    o["Model_Directory"]               << Option("model");
#endif
    o["Move_Overhead"]                 << Option(20, 0, 5000);
    o["MultiPV"]                       << Option(1, 1, 99999);
#ifdef USE_RL
    o["Nodes"]                         << Option(800, 0, 99999999);
#else
    o["Nodes"]                         << Option(0, 0, 99999999);
    o["Nodes_Limit"]                   << Option(0, 0, 999999999);
#endif
#ifdef TENSORRT
    o["Precision"]                     << Option("float16", {"float32", "float16", "int8"});
#else
    o["Precision"]                     << Option("int8", {"float32", "int8"});
#endif
#ifdef USE_RL
    o["Reuse_Tree"]                    << Option(false);
#else
    o["Reuse_Tree"]                    << Option(true);
#endif
#ifdef USE_RL
    o["Temperature_Moves"]             << Option(15, 0, 99999);
#else
    o["Temperature_Moves"]             << Option(0, 0, 99999);
#endif
    o["Use_NPS_Time_Manager"]          << Option(true);
#ifdef TENSORRT
    o["Use_TensorRT"]                  << Option(true);
#endif
#ifdef SUPPORT960
    o["UCI_Chess960"]                  << Option(false);
#endif
    o["Search_Type"]                   << Option("mcgs", {"mcgs", "mcts"});
#ifdef USE_RL
    o["Simulations"]                   << Option(3200, 0, 99999999);
#else
    o["Simulations"]                   << Option(0, 0, 99999999);
#endif
#ifndef MODE_XIANGQI
    o["SyzygyPath"]                    << Option("<empty>", on_tb_path);
#endif
    o["Threads"]                       << Option(2, 1, 512);
    o["Timeout_MS"]                    << Option(0, 0, 99999999);
#ifdef MODE_CRAZYHOUSE
      // we repeat "crazyhouse" in the list because of problem in XBoard/Winboard #23
    o["UCI_Variant"]                   << Option("crazyhouse", {"crazyhouse", "crazyhouse"});
#elif defined MODE_LICHESS
    o["UCI_Variant"]                   << Option(get_first_variant_with_model().c_str(), availableVariants);
#elif defined MODE_XIANGQI
    o["UCI_Variant"]                   << Option("xiangqi", {"xiangqi", "xiangqi"});
#else  // MODE = MODE_CHESS
    o["UCI_Variant"]                   << Option("chess", {"chess", "chess"});
#endif
    o["Use_Raw_Network"]               << Option(false);
    // additional UCI-Options for RL only
#ifdef USE_RL
    o["Centi_Node_Random_Factor"]      << Option(10, 0, 100);
    o["Centi_Quick_Dirichlet_Epsilon"] << Option(0, 0, 99999);
    o["Centi_Quick_Probability"]       << Option(0, 0, 100);
    o["Centi_Quick_Q_Value_Weight"]    << Option(70, 0, 99999);
    o["Centi_Raw_Prob_Temperature"]    << Option(25, 0, 100);
    o["Centi_Resign_Probability"]      << Option(90, 0, 100);
    o["Centi_Resign_Threshold"]        << Option(-90, -100, 100);
    o["MaxInitPly"]                    << Option(30, 0, 99999);
    o["MeanInitPly"]                   << Option(15, 0, 99999);
#ifdef LICHESS_MODE
    o["Model_Directory_Contender"]     << Option((string("model_contender/") + get_first_variant_with_model()).c_str());
#elif defined MODE_CHESS
    o["Model_Directory_Contender"]     << Option("model_contender/chess");
#elif defined MODE_XIANGQI
    o["Model_Directory_Contender"]     << Option("model_contender/xiangqi");
#else
    o["Model_Directory_Contender"]     << Option("model_contender/");
#endif
    o["Selfplay_Number_Chunks"]        << Option(640, 1, 99999);
    o["Selfplay_Chunk_Size"]           << Option(128, 1, 99999);
    o["Milli_Policy_Clip_Thresh"]      << Option(0, 0, 100);
    o["Quick_Nodes"]                   << Option(100, 0, 99999);
#endif
}

void OptionsUCI::setoption(istringstream &is, Variant& variant, StateObj& state)
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
        const string givenName = name;
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
#ifdef MODE_LICHESS
        if (name == "model_directory") {
            if (value.find((string)Options["UCI_Variant"]) == std::string::npos) {
                info_string_important("The Model_Directory must have the active UCI_Variant", string("'")+(string)Options["UCI_Variant"]+string("'"), "in its filepath");
                return;
            }
        }
#endif
        Options[name] = value;
        if (name != "uci_variant") {
            info_string_important("Updated option", givenName, "to", value);
        } else {
#ifdef XIANGQI
            // Workaround. Fairy-Stockfish does not use an enum for variants
            info_string_important("variant Xiangqi startpos rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1");
#else
            bool is960 = false;
            string uciVariant = check_uci_variant_input(value, &is960);
            std::transform(uciVariant.begin(), uciVariant.end(), uciVariant.begin(), ::tolower);
            Options["UCI_Variant"] << Option(uciVariant.c_str());
            info_string_important("Updated option", givenName, "to", uciVariant);
#ifdef SUPPORT960
            if (Options["UCI_Chess960"] != is960) {
                Options["UCI_Chess960"] << Option(is960);
                info_string("Updated option UCI_Chess960 to", (string)Options["UCI_Chess960"]);
            }
#endif // SUPPORT960
            variant = UCI::variant_from_name(uciVariant);
            state.init(variant, is960);

            string suffix_960 = (is960) ? "960" : "";
            Options["Model_Directory"] << Option(("model/" + (string)Options["UCI_Variant"] + suffix_960).c_str());
            Options["Model_Directory_Contender"] << Option(("model_contender/" + (string)Options["UCI_Variant"] + suffix_960).c_str());
            info_string_important("variant", (string)Options["UCI_Variant"] + suffix_960, "startpos", state.fen());
#endif
        }
    }
    else {
        info_string_important("Given option", name, "does not exist");
    }
}

string OptionsUCI::check_uci_variant_input(const string &value, bool *is960) {
    // default value of is960 == false
#ifdef SUPPORT960
    // we only want 960 for chess atm
    if (value == "fischerandom" || value == "chess960"
        || (((value == "chess") || (value == "standard")) && Options["UCI_Chess960"])) {
        *is960 = true;
        return "chess";
    }
#endif // SUPPORT960
    if (value == "standard") {
       return "chess";
    }
#ifdef MODE_LICHESS
    if (value == "giveaway" || value == "losers") {
        return "antichess";
    }
#endif // MODE_LICHESS
    // MODE_CRAZYHOUSE or others (keep value as is)
    return value;
}

const string OptionsUCI::get_first_variant_with_model()
{
    vector<string> dirs = get_directory_files("model/");
    for(string dir : dirs) {
        if (std::find(availableVariants.begin(), availableVariants.end(), dir) != availableVariants.end()) {
            const vector <string> files = get_directory_files("model/" + dir);
            if ("" != get_string_ending_with(files, "-bsize-1.onnx")) {
                return dir;
            }
        }
    }
    return availableVariants.front();
}

void OptionsUCI::init_new_search(SearchLimits& searchLimits, OptionsMap &options)
{
    searchLimits.reset();
    searchLimits.startTime = now();
    searchLimits.moveOverhead = TimePoint(options["Move_Overhead"]);
    searchLimits.nodes = options["Nodes"];
    searchLimits.nodesLimit = options["Nodes_Limit"];
    searchLimits.movetime = options["Fixed_Movetime"];
    searchLimits.simulations = options["Simulations"];
}
