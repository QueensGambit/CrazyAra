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
 * @file: agent.cpp
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "agent.h"
#include <chrono>
#include "misc.h"
#include "uci.h"

Agent::Agent(float temperature, unsigned int temperature_moves, bool verbose)
{
    this->temperature = temperature;
    this->temperature_moves = temperature_moves;
    this->verbose = verbose;
}

void Agent::perform_action(Board *pos, SearchLimits* searchLimits, EvalInfo& evalInfo)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    this->searchLimits = searchLimits;
    this->evalute_board_state(pos, evalInfo);
    sync_cout << "end time" << sync_endl;
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    evalInfo.elapsedTimeMS = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    evalInfo.nps = int(((evalInfo.nodes-evalInfo.nodesPreSearch) / (evalInfo.elapsedTimeMS / 1000.0f)) + 0.5f);
    evalInfo.bestMove = evalInfo.pv[0];
    sync_cout << evalInfo << sync_endl;
    sync_cout << "bestmove " << UCI::move(evalInfo.bestMove, pos->is_chess960()) << sync_endl;
}
