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
 * @file: evalinfo.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Stores the evaluation output for a given board position.
 */

#ifndef EVALINFO_H
#define EVALINFO_H
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "types.h"
#include <blaze/Math.h>
#include "constants.h"
#include "node.h"

using blaze::HybridVector;
using blaze::DynamicVector;

struct EvalInfo
{
    chrono::steady_clock::time_point start;
    chrono::steady_clock::time_point end;
    float bestMoveQ;
    std::vector<Move> legalMoves;
    DynamicVector<float> policyProbSmall;
    DynamicVector<float> childNumberVisits;
    int centipawns;
    size_t depth;
    size_t nodes;
    size_t nodesPreSearch;
    bool isChess960;
    std::vector<Move> pv;
    Move bestMove;
    int movesToMate;
    size_t tbHits;

    size_t calculate_elapsed_time_ms() const;
    int calculate_nps(size_t elapsedTimeMS) const;
    int calculate_nps() const;
};

/**
 * @brief value_to_centipawn Converts a value in A0-notation to roughly a centi-pawn loss
 * @param value floating value from [-1.,1.]
 * @return Returns centipawn conversion for value
 */
int value_to_centipawn(float value);

/**
 * @brief update_eval_info Updates the evaluation information based on the current search tree state
 * @param evalInfo Evaluation infomration struct
 * @param rootNode Root node of the search tree
 */
void update_eval_info(EvalInfo& evalInfo, Node* rootNode, size_t tbHits);

extern std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo);

#endif // EVALINFO_H
