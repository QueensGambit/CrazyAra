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
 * @file: constants.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include <stdlib.h>
#include "constants.h"
#include <iostream>

std::string mirror_move(std::string moveUCI) {

    // first copy the original move
    std::string moveMirrored = std::string(moveUCI);

    // replace the rank with the mirrored rank
    for (unsigned int idx = 0; idx < moveUCI.length(); ++idx) {
        if (isdigit(moveUCI[idx])) {
            int rank = moveUCI[idx] - '0';
            int rank_mirrored = 8 - rank + 1;
            moveMirrored[idx] = char(rank_mirrored + '0');
        }
    }
    return moveMirrored;
}
