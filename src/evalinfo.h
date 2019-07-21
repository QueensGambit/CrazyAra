/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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

#include "types.h"
#include <blaze/Math.h>
#include "constants.h"

using blaze::HybridVector;
using blaze::DynamicVector;

struct EvalInfo
{
public:
    EvalInfo();

    float value;
    std::vector<Move> legalMoves;
    DynamicVector<float> policyProbSmall;
    int centipawns;
    size_t depth;
    int nodes;
    size_t nodesPreSearch;
    float elapsedTimeMS;
    float nps;
    bool is_chess960;
    std::vector<Move> pv;
};

extern std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo);

#endif // EVALINFO_H
