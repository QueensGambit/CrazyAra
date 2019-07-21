/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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

void OptionsUCI::init(OptionsMap &o)
{
         // at most 2^32 clusters.
         constexpr int MaxHashMB = Is64Bit ? 131072 : 2048;

         o["UCI_Variant"]              << Option(availableVariants.front().c_str(), availableVariants);
         o["Search_Type"]              << Option("MCTS", {"MCTS"});
         o["Context"]                  << Option("CPU", {"CPU", "GPU"});
         o["Use_Raw_Network"]          << Option(false);
         o["Threads"]                  << Option(2, 1, 512);
         o["Batch_Size"]               << Option(8, 1, 8192);
         o["Playouts"]                 << Option(99999, 1, 99999);
         o["Centi_CPUCT"]              << Option(250, 1, 99999);
         o["Centi_Dirichlet_Epsilon"]  << Option(25, 1, 99999);
         o["Centi_Dirichlet_Epsilon"]  << Option(20, 1, 99999);
         o["Centi_U-Init-Divisor"]     << Option(100, 1, 99999);
         o["Max_Search_Depth"]         << Option(99, 1, 99999);
         o["Centi_Temperature"]        << Option(7, 0, 99999);
         o["Temperature_Moves"]        << Option(0, 0, 99999);
         o["Virtual_Loss"]             << Option(0, 3, 99999);
         o["Centi_Q-Value_Weight"]     << Option(70, 0, 99999);
         o["Enhance_Checks"]           << Option(true);
         o["Enhance_Captures"]         << Option(true);
         o["Use_Transposition_Table"]  << Option(true);
         o["Model_Architecture_Dir"]   << Option("default");
//         o["Debug_Log_File"]           << Option("", activate_logger);
         o["Contempt"]                 << Option(24, -100, 100);
         o["Hash"]                     << Option(16, 1, MaxHashMB);
         o["Move_Overhead"]            << Option(50, 0, 5000);
         o["Minimum_Thinking_Time"]    << Option(20, 0, 5000);
         o["UCI_Chess960"]             << Option(false);
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

  if (Options.count(name))
  {
      Options[name] = value;
      std::transform(name.begin(), name.end(), name.begin(), ::tolower);
      if (name == "uci_variant") {
          Variant variant = UCI::variant_from_name(value);
          sync_cout << "info string variant " << (string)Options["UCI_Variant"] << " startpos " << StartFENs[variant] << sync_endl;
//          Tablebases::init(variant, Options["SyzygyPath"]);
      }
  }
  else
      sync_cout << "No such option: " << name << sync_endl;
}
