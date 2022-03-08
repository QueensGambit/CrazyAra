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
#ifdef SF_DEPENDENCY
#include "uci.h"
#include "syzygy/tbprobe.h"
#endif
#include "../util/communication.h"
#include "../nn/neuralnetapi.h"

using namespace std;

void on_logger(const CUSTOM_UCI::Option& o) {
    CustomLogger::start(o, ifstream::app);
}

// method is based on 3rdparty/Stockfish/misc.cpp
inline TimePoint current_time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

// method is based on 3rdparty/Stockfish/uci.cpp
#ifdef SF_DEPENDENCY
void on_tb_path(const CUSTOM_UCI::Option& o) {
    Tablebases::init(UCI::variant_from_name(CustomOptions["UCI_Variant"]), CustomOptions["SyzygyPath"]);
}
#endif

void OptionsUCI::init(OptionsMap &o)
{
    o["Allow_Early_Stopping"]          << CUSTOM_UCI::Option(true);
#ifdef USE_RL
    o["Batch_Size"]                    << CUSTOM_UCI::Option(8, 1, 8192);
#else
#ifdef OPENVINO
    o["Batch_Size"]                    << CUSTOM_UCI::Option(16, 1, 8192);
#else
#ifdef MODE_CHESS
    o["Batch_Size"]                    << CUSTOM_UCI::Option(64, 1, 8192);
#else
    o["Batch_Size"]                    << CUSTOM_UCI::Option(16, 1, 8192);
#endif
#endif
#endif
    o["Child_Threads"]                 << CUSTOM_UCI::Option(4, 1, 512);
    o["Centi_CPuct_Init"]              << CUSTOM_UCI::Option(250, 1, 99999);
#ifdef USE_RL
    o["Centi_Dirichlet_Epsilon"]       << CUSTOM_UCI::Option(25, 0, 99999);
#else
    o["Centi_Dirichlet_Epsilon"]       << CUSTOM_UCI::Option(0, 0, 99999);
#endif
    o["Centi_Dirichlet_Alpha"]         << CUSTOM_UCI::Option(20, 1, 99999);
    o["Centi_Epsilon_Checks"]          << CUSTOM_UCI::Option(1, 0, 100);
    o["Centi_Epsilon_Greedy"]          << CUSTOM_UCI::Option(5, 0, 100);
//    o["Centi_U_Init"]                  << CUSTOM_UCI::Option(100, 0, 100);         currently disabled
//    o["Centi_U_Min"]                   << CUSTOM_UCI::Option(100, 0, 100);         currently disabled
//    o["U_Base"]                        << CUSTOM_UCI::Option(1965, 0, 99999);      currently disabled
    o["Centi_Node_Temperature"]        << CUSTOM_UCI::Option(170, 1, 99999);
    o["Centi_Q_Value_Weight"]          << CUSTOM_UCI::Option(100, 0, 99999);
    o["Centi_Q_Veto_Delta"]            << CUSTOM_UCI::Option(40, 0, 99999);
#ifdef USE_RL
    o["Centi_Quantile_Clipping"]       << CUSTOM_UCI::Option(0, 0, 100);
#else
    o["Centi_Quantile_Clipping"]       << CUSTOM_UCI::Option(25, 0, 100);
#endif
    o["Centi_Random_Move_Factor"]      << CUSTOM_UCI::Option(0, 0, 99);
#ifdef USE_RL
    o["Centi_Temperature"]             << CUSTOM_UCI::Option(80, 0, 99999);
#else
    o["Centi_Temperature"]             << CUSTOM_UCI::Option(170, 0, 99999);
#endif
    o["Centi_Temperature_Decay"]       << CUSTOM_UCI::Option(92, 0, 100);
    o["Centi_U_Init_Divisor"]          << CUSTOM_UCI::Option(100, 1, 99999);
    o["Centi_Virtual_Loss"]            << CUSTOM_UCI::Option(100, 0, 99999);
#if defined(MXNET) && defined(TENSORRT)
    o["Context"]                       << CUSTOM_UCI::Option("gpu", {"cpu", "gpu"});
#elif defined (TORCH)
    o["Context"]                       << CUSTOM_UCI::Option("gpu", {"cpu", "gpu"});
#elif defined (TENSORRT)
    o["Context"]                       << CUSTOM_UCI::Option("gpu", {"gpu"});
#else
    o["Context"]                       << CUSTOM_UCI::Option("cpu");
#endif
    o["CPuct_Base"]                    << CUSTOM_UCI::Option(19652, 1, 99999);
//    o["Enhance_Captures"]              << CUSTOM_UCI::Option(false);         currently disabled
    o["First_Device_ID"]               << CUSTOM_UCI::Option(0, 0, 99999);
    o["Fixed_Movetime"]                << CUSTOM_UCI::Option(0, 0, 99999999);
    o["Last_Device_ID"]                << CUSTOM_UCI::Option(0, 0, 99999);
    o["Log_File"]                      << CUSTOM_UCI::Option("", on_logger);
    o["MCTS_Solver"]                   << CUSTOM_UCI::Option(true);
#ifdef MODE_LICHESS
    o["Model_Directory"]               << CUSTOM_UCI::Option((string("model/") + engineName + "/" + get_first_variant_with_model()).c_str());
#else
    o["Model_Directory"]               << CUSTOM_UCI::Option(string("model/" + engineName + "/" + StateConstants::DEFAULT_UCI_VARIANT()).c_str());
#endif
    o["Move_Overhead"]                 << CUSTOM_UCI::Option(20, 0, 5000);
    o["MultiPV"]                       << CUSTOM_UCI::Option(1, 1, 99999);
#ifdef USE_RL
    o["Nodes"]                         << CUSTOM_UCI::Option(800, 0, 99999999);
#else
    o["Nodes"]                         << CUSTOM_UCI::Option(0, 0, 99999999);
    o["Nodes_Limit"]                   << CUSTOM_UCI::Option(0, 0, 999999999);
#endif
#ifdef TENSORRT
    o["Precision"]                     << CUSTOM_UCI::Option("float16", {"float32", "float16", "int8"});
#else
    o["Precision"]                     << CUSTOM_UCI::Option("float32", {"float32", "int8"});
#endif
#ifdef USE_RL
    o["Reuse_Tree"]                    << CUSTOM_UCI::Option(false);
#else
    o["Reuse_Tree"]                    << CUSTOM_UCI::Option(true);
#endif
#ifdef USE_RL
    o["Temperature_Moves"]             << CUSTOM_UCI::Option(15, 0, 99999);
#else
    o["Temperature_Moves"]             << CUSTOM_UCI::Option(0, 0, 99999);
#endif
    o["Use_NPS_Time_Manager"]          << CUSTOM_UCI::Option(true);
#ifdef TENSORRT
    o["Use_TensorRT"]                  << CUSTOM_UCI::Option(true);
#endif
#ifdef SUPPORT960
    o["UCI_Chess960"]                  << CUSTOM_UCI::Option(false);
#endif
    o["Search_Type"]                   << CUSTOM_UCI::Option("mcgs", {"mcgs", "mcts"});
#ifdef USE_RL
    o["Simulations"]                   << CUSTOM_UCI::Option(3200, 0, 99999999);
#else
    o["Simulations"]                   << CUSTOM_UCI::Option(0, 0, 99999999);
#endif
#ifdef MODE_STRATEGO
   o["Centi_Temperature"]              << CUSTOM_UCI::Option(99999, 0, 99999);
   o["Centi_Temperature_Decay"]        << CUSTOM_UCI::Option(100, 0, 100);
   o["Temperature_Moves"]              << CUSTOM_UCI::Option(0, 0, 99999);
#endif
#ifdef SF_DEPENDENCY
    o["SyzygyPath"]                    << CUSTOM_UCI::Option("<empty>", on_tb_path);
#endif
    o["Threads"]                       << CUSTOM_UCI::Option(2, 1, 512);
#ifdef OPENVINO
    o["Threads_NN_Inference"]          << CUSTOM_UCI::Option(8, 1, 512);
#endif
    o["Timeout_MS"]                    << CUSTOM_UCI::Option(0, 0, 99999999);
#ifdef MODE_LICHESS
    o["UCI_Variant"]                   << CUSTOM_UCI::Option(get_first_variant_with_model().c_str(), StateConstants::available_variants());
#else
    // we repeat e.g. "crazyhouse" in the list because of problem in XBoard/Winboard CrazyAra#23
    o["UCI_Variant"]                   << CUSTOM_UCI::Option(StateConstants::DEFAULT_UCI_VARIANT().c_str(), {StateConstants::DEFAULT_UCI_VARIANT().c_str(), StateConstants::DEFAULT_UCI_VARIANT().c_str()});
#endif
    o["Use_Raw_Network"]               << CUSTOM_UCI::Option(false);
    // additional UCI-Options for RL only
#ifdef USE_RL
    o["Centi_Node_Random_Factor"]      << CUSTOM_UCI::Option(10, 0, 100);
    o["Centi_Quick_Dirichlet_Epsilon"] << CUSTOM_UCI::Option(0, 0, 99999);
    o["Centi_Quick_Probability"]       << CUSTOM_UCI::Option(0, 0, 100);
    o["Centi_Quick_Q_Value_Weight"]    << CUSTOM_UCI::Option(70, 0, 99999);
    o["Centi_Raw_Prob_Temperature"]    << CUSTOM_UCI::Option(25, 0, 100);
    o["Centi_Resign_Probability"]      << CUSTOM_UCI::Option(90, 0, 100);
    o["Centi_Resign_Threshold"]        << CUSTOM_UCI::Option(-90, -100, 100);
    o["MaxInitPly"]                    << CUSTOM_UCI::Option(30, 0, 99999);
    o["MeanInitPly"]                   << CUSTOM_UCI::Option(15, 0, 99999);
#ifdef MODE_LICHESS
    o["Model_Directory_Contender"]     << CUSTOM_UCI::Option((string("model_contender/" + engineName + "/") + get_first_variant_with_model()).c_str());
#else
    o["Model_Directory_Contender"]     << CUSTOM_UCI::Option(string("model_contender/" + engineName + "/" + StateConstants::DEFAULT_UCI_VARIANT()).c_str());
#endif
    o["Selfplay_Number_Chunks"]        << CUSTOM_UCI::Option(640, 1, 99999);
    o["Selfplay_Chunk_Size"]           << CUSTOM_UCI::Option(128, 1, 99999);
    o["Milli_Policy_Clip_Thresh"]      << CUSTOM_UCI::Option(0, 0, 100);
    o["Quick_Nodes"]                   << CUSTOM_UCI::Option(100, 0, 99999);
#endif
}

void OptionsUCI::setoption(istringstream &is, int& variant, StateObj& state)
{

    string token, name, value;
    is >> token; // Consume "name" token

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (CustomOptions.find(name) != CustomOptions.end()) {
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
        CustomOptions[name] = value;
        if (name != "uci_variant" && name != "uci_chess960") {
            info_string_important("Updated option", givenName, "to", value);
        } else {
#ifdef XIANGQI
            if (name == "uci_variant") {
                // Workaround. Fairy-Stockfish does not use an enum for variants
                info_string_important("variant Xiangqi startpos rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1");
            }
#else
            bool is960 = false;
            string uciVariant = CustomOptions["UCI_Variant"];
            if (name == "uci_variant") {
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                uciVariant = check_uci_variant_input(value, &is960);
                CustomOptions["UCI_Variant"] << CUSTOM_UCI::Option(uciVariant.c_str());
                info_string_important("Updated option", givenName, "to", uciVariant);
#ifdef SUPPORT960
                if (Options["UCI_Chess960"] != is960) {
                    Options["UCI_Chess960"] << CUSTOM_UCI::Option(is960);
                    info_string("Updated option UCI_Chess960 to", (string)Options["UCI_Chess960"]);
                }
#endif // SUPPORT960
            } else { // name == "uci_chess960"
                info_string_important("Updated option", givenName, "to", value);
                is960 = CustomOptions["UCI_Chess960"];
            }
            variant = StateConstants::variant_to_int(uciVariant);
            state.init(variant, is960);

            string suffix_960 = (is960) ? "960" : "";
#ifdef MODE_LICHESS
            Options["Model_Directory"] << CUSTOM_UCI::Option(("model/" + engineName + "/" + (string)Options["UCI_Variant"] + suffix_960).c_str());
            Options["Model_Directory_Contender"] << CUSTOM_UCI::Option(("model_contender/" + engineName + "/" + (string)Options["UCI_Variant"] + suffix_960).c_str());
#endif
            info_string_important("variant", (string)CustomOptions["UCI_Variant"] + suffix_960, "startpos", state.fen());
#endif // not XIANGQI
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
    if (value == "threecheck") {
        return "3check";
    }
#endif // MODE_LICHESS
    // MODE_CRAZYHOUSE or others (keep value as is)
    return value;
}

const string OptionsUCI::get_first_variant_with_model()
{
    vector<string> dirs = get_directory_files("model/" + engineName + "/");
    const static vector<string> availableVariants = StateConstants::available_variants();
    for(string dir : dirs) {
        if (std::find(availableVariants.begin(), availableVariants.end(), dir) != availableVariants.end()) {
            const vector <string> files = get_directory_files("model/" + engineName + "/" + dir);
            if ("" != get_string_ending_with(files, ".onnx")) {
                return dir;
            }
        }
    }
    return StateConstants::DEFAULT_UCI_VARIANT();
}

void OptionsUCI::init_new_search(SearchLimits& searchLimits, OptionsMap &options)
{
    searchLimits.reset();
    searchLimits.startTime = current_time();
    searchLimits.moveOverhead = TimePoint(options["Move_Overhead"]);
    searchLimits.nodes = options["Nodes"];
    searchLimits.nodesLimit = options["Nodes_Limit"];
    searchLimits.movetime = options["Fixed_Movetime"];
    searchLimits.simulations = options["Simulations"];
}
