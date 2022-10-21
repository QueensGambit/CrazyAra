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
 * @file: stateobj.h
 * Created on 17.07.2020
 * @author: queensgambit
 *
 * This file defines the wrapper function and classes which are used during MCTS.
 * Edit this file and its corresponding source file to activate a custom environment.
 */

#ifndef STATEOBJ_H
#define STATEOBJ_H

#include <unordered_map>
#include <blaze/Math.h>
#include "state.h"
#include "constants.h"
using blaze::StaticVector;
using blaze::DynamicVector;

#ifdef MODE_POMMERMAN
#include "pommermanstate.h"
#elif MODE_OPEN_SPIEL
#include "environments/open_spiel/openspielstate.h"
#elif defined (MODE_XIANGQI) || defined (MODE_BOARDGAMES)
#include "environments/fairy_state/fairystate.h"
#include "environments/fairy_state/fairyoutputrepresentation.h"
#elif MODE_STRATEGO
#include "environments/stratego_related/strategostate.h"
#else
#include "environments/chess_related/boardstate.h"
#include "environments/chess_related/outputrepresentation.h"
#endif

#ifdef MODE_POMMERMAN
    using StateObj = PommermanState;
    using StateConstants = StateConstantsPommerman;
#elif MODE_OPEN_SPIEL
    using StateObj = OpenSpielState;
    using StateConstants = StateConstantsOpenSpiel;
#elif defined (MODE_XIANGQI) || defined (MODE_BOARDGAMES)
    using StateObj = FairyState;
    using StateConstants = StateConstantsFairy;
#elif MODE_STRATEGO
    using StateObj = StrategoState;
    using StateConstants = StateConstantsStratego;
#else
    using StateObj = BoardState;
    using StateConstants = StateConstantsBoard;
#endif


/**
 * @brief get_probs_of_move_list Returns an array in which entry relates to the probability for the given move list.
                                 Its assumed that the moves in the move list are legal and shouldn't be mirrored.
 * @param batchIdx Index to use in policyProb when extracting the probabilities for all legal moves
 * @param policyProb Policy array from the neural net prediction
 * @param legalMoves List of legal moves for a specific board position
 * @param lastLegalMove Pointer to the last legal move
 * @param mirrorPolicy Determines if the policy should be mirrored
 * @param normalize True, if the probability should be normalized
 * @param selectPolicyFromPlane Sets if the policy is encoded in policy map representation
 * @return policyProbSmall - A hybrid blaze vector which stores the probabilities for the given move list
 */
void get_probs_of_move_list(const size_t batchIdx, const float* policyProb, const std::vector<Action>& legalMoves, bool mirrorPolicy,
                            bool normalize, DynamicVector<double> &policyProbSmall, bool selectPolicyFromPlane);

/**
 * @brief get_policy_data_batch Returns the pointer of the batch for the policy predictions
 * @param batchIdx Batch index for the current predicion
 * @param policyProb All policy predicitons from the batch
 * @param isPolicyMap Sets if the policy is encoded in policy map representation
 * @return Starting pointer for predictions of the current batch
 */
const float*  get_policy_data_batch(const size_t batchIdx, const float* policyProb, bool isPolicyMap);

/**
 * @brief get_auxiliary_data_batch Returns the pointer of the batch for the auxliary predictions
 * @param batchIdx Batch index for the current predicion
 * @param auxiliaryOutputs All auxiliary predictions of the batch
 * @return Starting pointer for predictions of the current batch
 */
const float*  get_auxiliary_data_batch(const size_t batchIdx, const float* auxiliaryOutputs);

#endif // STATEOBJ_H
