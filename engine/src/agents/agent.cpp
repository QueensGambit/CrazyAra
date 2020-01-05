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
 * @file: agent.cpp
 * Created on 17.06.2019
 * @author: queensgambit
 */

#include <iostream>
#include <chrono>

#include "agent.h"
#include "misc.h"
#include "uci.h"
#include "../util/communication.h"
#include "../util/blazeutil.h"

using namespace std;


void Agent::set_best_move(EvalInfo &evalInfo, size_t moveCounter)
{
    if (moveCounter < playSettings->temperatureMoves && playSettings->initTemperature > 0.01) {
        info_string("Sample move");
        DynamicVector<double> policyProbSmall = evalInfo.childNumberVisits / sum(evalInfo.childNumberVisits);
        apply_temperature(policyProbSmall, get_current_temperature(*playSettings, moveCounter));
        size_t moveIdx = random_choice(policyProbSmall);
        evalInfo.bestMove = evalInfo.legalMoves[moveIdx];
    }
    else {
        evalInfo.bestMove = evalInfo.pv[0];
    }
}

Agent::Agent(PlaySettings* playSettings, bool verbose):
    playSettings(playSettings), verbose(verbose)
{
}

void Agent::perform_action(Board *pos, SearchLimits* searchLimits, EvalInfo& evalInfo)
{
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    this->searchLimits = searchLimits;
    this->evaluate_board_state(pos, evalInfo);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    evalInfo.elapsedTimeMS = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    evalInfo.nps = int(((evalInfo.nodes-evalInfo.nodesPreSearch) / (evalInfo.elapsedTimeMS / 1000.0f)) + 0.5f);
    set_best_move(evalInfo, pos->total_move_cout());
    info_score(evalInfo);
    info_string(pos->fen());
    info_bestmove(UCI::move(evalInfo.bestMove, pos->is_chess960()));
}

