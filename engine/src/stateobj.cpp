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
 * @file: stateobj.cpp
 * Created on 17.07.2020
 * @author: queensgambit
 */

#include "stateobj.h"
#include "constants.h"


void get_probs_of_move_list(const size_t batchIdx, const float* policyProb, const std::vector<Action>& legalMoves, bool mirrorPolicy, bool normalize, DynamicVector<double> &policyProbSmall, bool selectPolicyFromPlane)
{
    size_t vectorIdx;
    for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx) {
        if (mirrorPolicy) {
            // find the according index in the vector
            vectorIdx = StateConstants::action_to_index<normal,mirrored>(legalMoves[mvIdx]);
        } else {
            // use the non-mirrored look-up table instead
            vectorIdx = StateConstants::action_to_index<normal,notMirrored>(legalMoves[mvIdx]);
        }
        assert(vectorIdx < StateConstants::NB_LABELS());

        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        if (selectPolicyFromPlane) {
            policyProbSmall[mvIdx] = policyProb[batchIdx*StateConstants::NB_LABELS_POLICY_MAP()+vectorIdx];
        } else {
            policyProbSmall[mvIdx] = policyProb[batchIdx*StateConstants::NB_LABELS()+vectorIdx];
        }
    }

    if (normalize) {
        policyProbSmall = softmax(policyProbSmall);
    }
}

const float* get_policy_data_batch(const size_t batchIdx, const float* probOutputs, bool isPolicyMap)
{
    if (isPolicyMap) {
        return probOutputs + batchIdx*StateConstants::NB_LABELS_POLICY_MAP();
    }
    return probOutputs + batchIdx*StateConstants::NB_LABELS();
}

const float* get_auxiliary_data_batch(const size_t batchIdx, const float* auxiliaryOutputs)
{
    return auxiliaryOutputs + batchIdx*StateConstants::NB_AUXILIARY_OUTPUTS();
}
