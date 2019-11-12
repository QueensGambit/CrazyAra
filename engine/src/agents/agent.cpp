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
 */

#include <iostream>
#include <chrono>

#include "agent.h"
#include "misc.h"
#include "uci.h"
#include "../util/communication.h"

using namespace std;

size_t Agent::pick_move_idx(DynamicVector<double>& policyProbSmall)
{
    double* prob = policyProbSmall.data();
    default_random_engine eng{static_cast<long unsigned int>(time(0))};
    discrete_distribution<> d(prob, prob+policyProbSmall.size());
    return size_t(d(eng));
}

void Agent::apply_temperature_to_policy(DynamicVector<double> &policyProbSmall)
{
    // apply exponential scaling
    policyProbSmall = pow(policyProbSmall, 1.0f / temperature);
    // re-normalize the values to probabilities again
    policyProbSmall /= sum(policyProbSmall);
}

void Agent::set_best_move(EvalInfo &evalInfo, size_t moveCounter)
{
    if (moveCounter <= temperatureMoves && temperature > 0.01f) {
        info_string("Sample move");
        DynamicVector<double> policyProbSmall = evalInfo.childNumberVisits / sum(evalInfo.childNumberVisits);
        apply_temperature_to_policy(policyProbSmall);
        size_t moveIdx = pick_move_idx(policyProbSmall);
        evalInfo.bestMove = evalInfo.legalMoves[moveIdx];
    }
    else {
        evalInfo.bestMove = evalInfo.pv[0];
    }
}

Agent::Agent(float temperature, unsigned int temperature_moves, bool verbose)
{
    this->temperature = temperature;
    this->temperatureMoves = temperature_moves;
    this->verbose = verbose;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gen.seed(seed);
    mt19937 gen(rd());
}

void Agent::perform_action(Board *pos, SearchLimits* searchLimits, EvalInfo& evalInfo)
{
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    this->searchLimits = searchLimits;
    this->evalute_board_state(pos, evalInfo);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    evalInfo.elapsedTimeMS = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    evalInfo.nps = int(((evalInfo.nodes-evalInfo.nodesPreSearch) / (evalInfo.elapsedTimeMS / 1000.0f)) + 0.5f);
    set_best_move(evalInfo, pos->total_move_cout());
    info_score(evalInfo);
    info_string(pos->fen());
    info_bestmove(UCI::move(evalInfo.bestMove, pos->is_chess960()));
}

