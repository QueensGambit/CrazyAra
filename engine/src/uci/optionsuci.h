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
 * @file: optionsuci.h
 * Created on 13.07.2019
 * @author: queensgambit
 *
 * UCI Option definition and initialization with default values.
 */

#ifndef OPTIONSUCI_H
#define OPTIONSUCI_H

#ifndef SF_DEPENDENCY
#include "customuci.h"
#else
#include "uci.h"
#endif
#include "stateobj.h"
#include "agents/config/searchlimits.h"

#ifndef SF_DEPENDENCY
using namespace CUSTOM_UCI;
#else
using namespace UCI;
#endif

namespace OptionsUCI {

    /**
     * @brief init Defines and initiatlizes the UCI options
     * @param o Alias to the option map which will get initialized
     */
    void init(OptionsMap& o);

    /**
     * @brief setoption Sets a given option value to the Options map.
     * Method is based on 3rdparty/Stockfish/uci.cpp
     * @param is Stringstream
     * @param variant Active variant
     * @param state Active state object
     */
    void setoption(istringstream& is, int& variant, StateObj& state);


    /**
     * @brief check_uci_variant_input Gets a uci variant and translates it if necessary.
     * This way we can support multiple names for the same variants.
     * @param value UCI Variant name
     * @param is960 Bool pointer, which shall be false when passed
     * @return string The uci variant string that we use internally to represent the variant.
     */
    string check_uci_variant_input(const string &value, bool *is960);

    /**
     * @brief get_first_variant_with_model Return the name of the first variant in the vector
     * of available variants that has a model saved in 'model/<variant_name>'.
     * @return string UCI variant name.
     */
    const string get_first_variant_with_model();

    /**
     * @brief init_new_search Initializes the struct according to the given OptionsMap for a new search
     * @param searchLimit search limits struct to be changed
     * @param options UCI Options struct (won't be changed)
     */
    void init_new_search(SearchLimits& searchLimits, OptionsMap &options);
}

#endif // OPTIONSUCI_H
