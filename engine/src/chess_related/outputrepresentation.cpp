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
//#include "types.h"
#include "policymaprepresentation.h"
#include "sfutil.h"
using namespace std;

//void get_probs_of_moves(const float *data, const vector<Move>& legalMoves, unordered_map<Move, size_t, std::hash<int>>& moveLookup, DynamicVector<float> &policyProbSmall)
//{
//    //    // allocate sufficient memory -> is assumed that it has already been done
//    //    policyProbSmall.resize(legalMoves.size());
//    for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx) {
//        // retrieve vector index from look-up table
//        // set the right prob value
//        // accessing the data on the raw floating point vector is faster
//        // than calling policyProb.At(batchIdx, vectorIdx)
//        policyProbSmall[mvIdx] = data[moveLookup[legalMoves[mvIdx]]];
//    }
//}

void apply_softmax(DynamicVector<float> &policyProbSmall)
{
    policyProbSmall = softmax(policyProbSmall);
}

void init_policy_constants(bool isPolicyMap,
                           action_idx_map& MV_LOOKUP,
                           action_idx_map& MV_LOOKUP_MIRRORED,
                           action_idx_map& MV_LOOKUP_CLASSIC,
                           action_idx_map& MV_LOOKUP_MIRRORED_CLASSIC)
{
#ifdef SUPPORT960
    const bool is960 = true;
#else
    const bool is960 = false;
#endif

    // fill mirrored label list and look-up table
    for (size_t mvIdx=0; mvIdx < NB_LABELS; mvIdx++) {
        LABELS_MIRRORED[mvIdx] = mirror_move(LABELS[mvIdx]);
        std::vector<Move> moves = make_move(LABELS[mvIdx], is960);
        for (Move move : moves) {
            isPolicyMap ? MV_LOOKUP.insert({move, FLAT_PLANE_IDX[mvIdx]}) : MV_LOOKUP.insert({move, mvIdx});
            MV_LOOKUP_CLASSIC.insert({move, mvIdx});
        }
        std::vector<Move> moves_mirrored = make_move(LABELS_MIRRORED[mvIdx], is960);
        for (Move move : moves_mirrored) {
            isPolicyMap ? MV_LOOKUP_MIRRORED.insert({move, FLAT_PLANE_IDX[mvIdx]}) : MV_LOOKUP_MIRRORED.insert({move, mvIdx});
            MV_LOOKUP_MIRRORED_CLASSIC.insert({move, mvIdx});
        }
    }
}
