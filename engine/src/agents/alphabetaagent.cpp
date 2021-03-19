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
 * @file: alphabetaagent.cpp
 * Created on 14.03.2021
 * @author: queensgambit
 */

#include "alphabetaagent.h"
#include "util/blazeutil.h"

AlphaBetaAgent::AlphaBetaAgent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose):
    Agent(net, playSettings, verbose),
    isRunning(false)
{

}

void AlphaBetaAgent::evaluate_board_state()
{
    evalInfo->legalMoves = state->legal_actions();
    evalInfo->init_vectors_for_multi_pv(1UL);

    searchLimits->depth = 7;  // TODO make this customizeable
    evalInfo->nodes = 0;

    ActionTrajectory pline;
    isRunning = true;
    float bestValue = negamax(state, searchLimits->depth, &pline);
    evalInfo->pv[0].clear();
    for (int idx = 0; idx < pline.nbMoves; ++idx) {
        evalInfo->pv[0].emplace_back(pline.moves[idx]);
    }
    evalInfo->bestMove = evalInfo->pv[0][0];
    evalInfo->centipawns[0] = value_to_centipawn(bestValue);
    evalInfo->movesToMate[0] = 0;
    evalInfo->depth = pline.nbMoves;
    evalInfo->selDepth = 1;
    evalInfo->tbHits = 0;
    evalInfo->isChess960 = state->is_chess960();
}

void AlphaBetaAgent::stop()
{
    isRunning = false;
}

void AlphaBetaAgent::apply_move_to_tree(Action move, bool ownMove)
{

}

size_t get_quantile_idx(DynamicVector<float>& policy, vector<size_t> ordering, float quantile, size_t maxCnt)
{
    float sum = 0;
    size_t idxCnt = 0;

    for (size_t idx : ordering) {
        ++idxCnt;
        sum += policy[idx];
        if (sum > quantile) {
            return idx;
        }
        if (idxCnt > maxCnt) {
            return idx;
        }
    }
    return ordering.back();
}

float AlphaBetaAgent::negamax(StateObj* state, int depth, ActionTrajectory* pline, float alpha, float beta, SideToMove color, bool allMoves, bool inCheck)
{
    float customTerminalValue;
    TerminalType terminal = state->is_terminal(state->legal_actions().size(), inCheck, customTerminalValue);
    if (terminal != TERMINAL_NONE) {
        return terminal_to_float(terminal);
    }

    state->get_state_planes(true, inputPlanes);
    net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
    ++evalInfo->nodes;
    ActionTrajectory line;
    if (depth == 0) {
        pline->nbMoves = 0;
        return valueOutputs[0];  // the value is always returned in the view of the current player
    }

    float bestValue = -__FLT_MAX__;  // initialization

    std::vector<Action> legalActions = state->legal_actions();

    DynamicVector<float> policyProbSmall(legalActions.size());
    get_probs_of_move_list(0, probOutputs, legalActions, state->side_to_move(),
                           !net->is_policy_map(), policyProbSmall, net->is_policy_map());

    auto p = sort_permutation(policyProbSmall, std::greater<float>());
    size_t stopIdx = get_quantile_idx(policyProbSmall, p, 0.95, 7);  // TODO make this customizeable

    size_t idx = 0;
    for (size_t mvIdx : p) {  // each child of position
        ++idx;
        Action action = legalActions[mvIdx];
        unique_ptr<StateObj> stateChild = unique_ptr<StateObj>(state->clone());
        bool inCheck = stateChild->gives_check(action);
        stateChild->do_action(action);
        float value = -negamax(stateChild.get(), depth - 1, &line, -beta, -alpha, -color, allMoves, inCheck);
        if (value > bestValue) {
            bestValue = value;
        }

        if (value > alpha) {
            alpha = value;
            pline->moves[0] = action;
            memcpy(pline->moves + 1, line.moves, line.nbMoves * sizeof(Action));
            pline->nbMoves = line.nbMoves + 1;
        }
        if (alpha >= beta) {
            break;
        }
        if (mvIdx == stopIdx) {
            break;
        }
    }
    return bestValue;
}
