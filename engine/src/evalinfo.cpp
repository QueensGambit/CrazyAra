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
 * @file: evalinfo.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "evalinfo.h"
#include "uci.h"

std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo)
{
    os << "info score cp " << evalInfo.centipawns
       << " depth " << evalInfo.depth
       << " nodes " << evalInfo.nodes
       << " time " << evalInfo.elapsedTimeMS
       << " nps " << evalInfo.nps
       << " pv";
    for (Move move: evalInfo.pv) {
        os << " " << UCI::move(move, evalInfo.isChess960);
    }
    return os;
}
