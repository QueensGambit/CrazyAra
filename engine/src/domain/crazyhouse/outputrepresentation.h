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
 * @file: outputrepresentation.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Provides all methods to convert a move to policy representation and back.
 */

#ifndef OUTPUTREPRESENTATION_H
#define OUTPUTREPRESENTATION_H

#include "types.h"
#include <blaze/Math.h>
#include "constants.h"

using blaze::HybridVector;
using blaze::DynamicVector;

using namespace std;

/**
 * @brief get_policy_data_batch Returns the pointer of the batch for the policy predictions
 * @param batchIdx Batch index for the current predicion
 * @param policyProb All policy predicitons from the batch
 * @param isPolicyMap Sets if the policy is encoded in policy map representation
 * @return Starting pointer for predictions of the current batch
 */
const float*  get_policy_data_batch(const size_t batchIdx, const float* policyProb, bool isPolicyMap);

/**
 * @brief get_current_move_lookup Returns the look-up table to use depending on the side to move
 * @param sideToMove Current side to move
 * @return Returns either MOVE_LOOK_UP or MOVE_LOOK_UP_MIRRORED
 */
unordered_map<Move, size_t>& get_current_move_lookup(Color sideToMove);

/**
 * @brief get_probs_of_move_list Returns an array in which entry relates to the probability for the given move list.
                                 Its assumed that the moves in the move list are legal and shouldn't be mirrored.
 * @param batchIdx Index to use in policyProb when extracting the probabilities for all legal moves
 * @param policyProb Policy array from the neural net prediction
 * @param legalMoves List of legal moves for a specific board position
 * @param lastLegalMove Pointer to the last legal move
 * @param sideToMove Determine if it's white's or black's turn to move
 * @param normalize True, if the probability should be normalized
 * @param select_policy_from_plance Sets if the policy is encoded in policy map representation
 * @return policyProbSmall - A hybrid blaze vector which stores the probabilities for the given move list
 */
void get_probs_of_move_list(const size_t batchIdx, const float* policyProb, const std::vector<Move> &legalMoves, Color sideToMove,
                            bool normalize, DynamicVector<float> &policyProbSmall, bool select_policy_from_plance);

void get_probs_of_moves(const float *data, const vector<Move>& legalMoves,
                        unordered_map<Move, size_t>& moveLookup, DynamicVector<float> &policyProbSmall);

void apply_softmax(DynamicVector<float> &policyProbSmall);

#endif // OUTPUTREPRESENTATION_H
