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
 * @file: outputrepresentation.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "outputrepresentation.h"
#include "types.h"
#include "policymaprepresentation.h"
using namespace std;

// TODO: Change this later to blaze::HybridVector<float, MAX_NB_LEGAL_MOVES>
void get_probs_of_move_list(const size_t batchIdx, const float* policyProb, const std::vector<Move> &legalMoves, Color sideToMove, bool normalize, DynamicVector<float> &policyProbSmall, bool selectPolicyFromPlane)
{
    size_t vectorIdx;
    for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx) {
        if (sideToMove == WHITE) {
            // find the according index in the vector
            vectorIdx = MV_LOOKUP[legalMoves[mvIdx]];
        } else {
            // use the mirrored look-up table instead
            vectorIdx = MV_LOOKUP_MIRRORED[legalMoves[mvIdx]];
        }
        assert(vectorIdx < NB_LABELS);

        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        if (selectPolicyFromPlane) {
            policyProbSmall[mvIdx] = policyProb[batchIdx*NB_LABELS_POLICY_MAP+vectorIdx];
        } else {
            policyProbSmall[mvIdx] = policyProb[batchIdx*NB_LABELS+vectorIdx];
        }
    }

    if (normalize) {
        policyProbSmall = softmax(policyProbSmall);
    }
}

void get_probs_of_moves(const float *data, const vector<Move>& legalMoves, unordered_map<Move, size_t>& moveLookup, DynamicVector<float> &policyProbSmall)
{
//    // allocate sufficient memory -> is assumed that it has already been done
//    policyProbSmall.resize(legalMoves.size());
    for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx) {
        // retrieve vector index from look-up table
        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        policyProbSmall[mvIdx] = data[moveLookup[legalMoves[mvIdx]]];
    }
}

const float* get_policy_data_batch(const size_t batchIdx, const float* probOutputs, bool isPolicyMap)
{
    if (isPolicyMap) {
        return probOutputs + batchIdx*NB_LABELS_POLICY_MAP;
    }
    return probOutputs + batchIdx*NB_LABELS;
}

unordered_map<Move, size_t>& get_current_move_lookup(Color sideToMove)
{
    if (sideToMove == WHITE) {
        // use the look-up table for the first player
        return MV_LOOKUP;
    }
    // use the mirrored look-up table instead
    return MV_LOOKUP_MIRRORED;
}

void apply_softmax(DynamicVector<float> &policyProbSmall)
{
    policyProbSmall = softmax(policyProbSmall);
}
