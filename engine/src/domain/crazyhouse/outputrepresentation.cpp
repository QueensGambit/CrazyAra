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
void get_probs_of_move_list(const size_t batchIdx, const NDArray* policyProb, const std::vector<Move> &legalMoves, Color sideToMove, bool normalize, DynamicVector<float> &policyProbSmall, bool selectPolicyFromPlane)
{
//    // allocate sufficient memory -> is assumed that it has already been done
//    policyProbSmall.resize(legalMoves.size());
    const float *data = policyProb->GetData();
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
            policyProbSmall[mvIdx] = data[batchIdx*NB_LABELS_POLICY_MAP+vectorIdx];
        } else {
            policyProbSmall[mvIdx] = data[batchIdx*NB_LABELS+vectorIdx];
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

const float* get_policy_data_batch(const size_t batchIdx, const NDArray *probOutputs, bool isPolicyMap)
{
    if (isPolicyMap) {
        return probOutputs->GetData() + batchIdx*NB_LABELS_POLICY_MAP;
    }
    return probOutputs->GetData() + batchIdx*NB_LABELS;
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
