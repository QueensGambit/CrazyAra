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
 * @file: searchlimits.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "searchlimits.h"
#include "state.h"
#include "constants.h"

std::ostream &operator<<(std::ostream &os, const SearchLimits &searchLimits)
{
    os << " wtime " << searchLimits.time[FIRST_PLAYER_IDX]
          << " btime " << searchLimits.time[FIRST_PLAYER_IDX+1]
             << " winc "  << searchLimits.inc[FIRST_PLAYER_IDX]
                << " binc "  << searchLimits.inc[FIRST_PLAYER_IDX+1]
                   << "movestogo " << searchLimits.movestogo;
    return os;
}

SearchLimits::SearchLimits()
{
    reset();
}

void SearchLimits::reset()
{
    movetime = 0;
    nodes = 0;
    nodesLimit = 0;
    simulations = 0;
    movestogo = 0;
    depth = 0;
    minMovetime = 0;
    npmsec = 0;
    startTime = 0;
    moveOverhead = 0;
    infinite = false;
    ponder = false;
    time[FIRST_PLAYER_IDX] = 0;
    time[FIRST_PLAYER_IDX+1] = 0;
    inc[FIRST_PLAYER_IDX] = 0;
    inc[FIRST_PLAYER_IDX+1] = 0;
}

int SearchLimits::get_safe_remaining_time(SideToMove sideToMove) const
{
    return max(time[sideToMove] - moveOverhead * TIME_BUFFER_FACTOR, 1);
}

bool is_game_sceneario(const SearchLimits* searchLimits)
{
    return searchLimits->movestogo != 0 || searchLimits->time[FIRST_PLAYER_IDX] != 0 || searchLimits->time[FIRST_PLAYER_IDX+1] != 0;
}
