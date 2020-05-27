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
 * @file: evalinfo.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "evalinfo.h"
#include "uci.h"

std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo)
{
    const size_t elapsedTimeMS = evalInfo.calculate_elapsed_time_ms();

    if (evalInfo.movesToMate == 0) {
       os << "cp " << evalInfo.centipawns;
    }
    else {
       os << "mate " << evalInfo.movesToMate;
    }
    os << " depth " << evalInfo.depth
       << " seldepth " << evalInfo.selDepth
       << " nodes " << evalInfo.nodes
       << " time " << elapsedTimeMS
       << " nps " << evalInfo.calculate_nps(elapsedTimeMS)
       << " tbhits " << evalInfo.tbHits
       << " pv";
    for (Move move: evalInfo.pv) {
        os << " " << UCI::move(move, evalInfo.isChess960);
    }
    return os;
}

size_t EvalInfo::calculate_elapsed_time_ms() const
{
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

size_t EvalInfo::calculate_nps(size_t elapsedTimeMS) const
{
    // avoid division by 0
    if (elapsedTimeMS == 0) {
        elapsedTimeMS = 1;
    }
    return int(((nodes-nodesPreSearch) / (elapsedTimeMS / 1000.0f)) + 0.5f);
}

size_t EvalInfo::calculate_nps() const
{
    return calculate_nps(calculate_elapsed_time_ms());
}

// https://helloacm.com/how-to-implement-the-sgn-function-in-c/
template <class T>
inline int
sgn(T v) {
    return (v > T(0)) - (v < T(0));
}

int value_to_centipawn(float value)
{
    if (std::abs(value) >= 1) {
        // return a constant if the given value is 1 (otherwise log will result in infinity)
        return sgn(value) * 9999;
    }
    // use logarithmic scaling with basis 1.1 as a pseudo centipawn conversion
    return int(-(sgn(value) * std::log(1.0f - std::abs(value)) / std::log(1.2f)) * 100.0f);
}

void update_eval_info(EvalInfo& evalInfo, Node* rootNode, size_t tbHits, size_t selDepth)
{
    evalInfo.childNumberVisits = rootNode->get_child_number_visits();
    evalInfo.policyProbSmall.resize(rootNode->get_number_child_nodes());
    size_t bestMoveIdx;
    rootNode->get_mcts_policy(evalInfo.policyProbSmall, bestMoveIdx);
    evalInfo.legalMoves = rootNode->get_legal_moves();
    rootNode->get_principal_variation(evalInfo.pv);
    evalInfo.depth = evalInfo.pv.size();
    evalInfo.selDepth = selDepth;
    // return mate score for known wins and losses
    if (rootNode->get_node_type() == SOLVED_WIN) {
        // always round up the ply counter
        evalInfo.movesToMate = evalInfo.pv.size() / 2 + evalInfo.pv.size() % 2;
    }
    else if (rootNode->get_node_type() == SOLVED_LOSS) {
        // always round up the ply counter
        evalInfo.movesToMate = -evalInfo.pv.size() / 2 + evalInfo.pv.size() % 2;
    }
    else {
        evalInfo.movesToMate = 0;
        evalInfo.bestMoveQ = rootNode->get_q_value(bestMoveIdx);
        evalInfo.centipawns = value_to_centipawn(evalInfo.bestMoveQ);
    }
    evalInfo.nodes = get_node_count(rootNode);
    evalInfo.tbHits = tbHits;
}
