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

#include <blaze/Math.h>
#include "node.h"

using blaze::HybridVector;
using blaze::DynamicVector;

struct EvalInfo
{
    chrono::steady_clock::time_point start;
    chrono::steady_clock::time_point end;
    std::vector<float> bestMoveQ;
    std::vector<Action> legalMoves;
    DynamicVector<double> policyProbSmall;
    DynamicVector<double> childNumberVisits;
    DynamicVector<float> qValues;
    std::vector<int> centipawns;
    size_t depth;
    size_t selDepth;
    uint_fast32_t nodes;
    size_t nodesPreSearch;
    bool isChess960;
    std::vector<std::vector<Action>> pv;
    Action bestMove;
    std::vector<int> movesToMate;
    size_t tbHits;

    size_t calculate_elapsed_time_ms() const;
    size_t calculate_nps(size_t elapsedTimeMS) const;
    size_t calculate_nps() const;

    /**
     * @brief init_vectors_for_multi_pv Initializes the memory of the vectors accoring to the multi pv size
     */
    void init_vectors_for_multi_pv(size_t multiPV);
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
 * @param selDepth Selective depth, in this case the maximum reached depth
 * @param searchSettings searchSettings struct
 */
void update_eval_info(EvalInfo& evalInfo, const Node* rootNode, size_t tbHits, size_t selDepth, const SearchSettings* searchSettings);

/**
 * @brief get_best_move_q Return the value evaluation for the given next node.
 * If it is a drawn tablebase position, 0.0 is returned.
 * Warning: Must be called with d != nullptr
 * @param nextNode Node object
 * @param searchSettings Search settings
 * @return value evaluation
 */
float get_best_move_q(const Node* nextNode, const SearchSettings* searchSettings);

/**
 * @brief set_eval_for_single_pv Sets the eval struct pv line and score for a single pv
 * @param evalInfo struct
 * @param rootNode root node of the tree
 * @param idx index of the pv line
 * @param indices sorted indices of each child node
 */
void set_eval_for_single_pv(EvalInfo& evalInfo, const Node* rootNode, size_t idx, vector<size_t>& indices, const SearchSettings* searchSettings);

/**
 * @brief operator << Returns all MultiPV as a string sperated by endl
 * @param os stream handle
 * @param evalInfo struct
 * @return stream handle
 */
extern std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo);

/**
 * @brief print_single_pv Is used to return a single PV line
 * @param os stream handle
 * @param evalInfo struct
 * @param idx index of the MultiPV lines
 * @param elapsedTimeMS elapsed time during mcts search
 */
void print_single_pv(std::ostream& os, const EvalInfo& evalInfo, size_t idx, size_t elapsedTimeMS);

/**
 * @brief sort_eval_lists Sorts the eval policy, eval legal moves and indices vector based on the descending order of eval policy
 * @param evalInfo struct
 * @param indices vector of indices for each child node, given as a range incremental increasing list from 0 to numberChildNodes-1
 */
void sort_eval_lists(EvalInfo& evalInfo, vector<size_t>& indices);

#endif // EVALINFO_H
