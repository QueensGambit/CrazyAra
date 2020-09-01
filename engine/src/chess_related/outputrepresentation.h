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

#include <blaze/Math.h>
#include "constants.h"
#include "state.h"

using blaze::HybridVector;
using blaze::DynamicVector;

using namespace std;

//void get_probs_of_moves(const float *data, const vector<Move>& legalMoves,
//                        unordered_map<Move, size_t, std::hash<int>>& moveLookup, DynamicVector<float> &policyProbSmall);

void apply_softmax(DynamicVector<float> &policyProbSmall);

void init_policy_constants(bool isPolicyMap);


#endif // OUTPUTREPRESENTATION_H
