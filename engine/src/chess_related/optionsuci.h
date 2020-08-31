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

#include "uci.h"
#include "misc.h"
#include "variants.h"

using namespace UCI;

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
     */
    void setoption(istringstream& is);

}

#endif // OPTIONSUCI_H
