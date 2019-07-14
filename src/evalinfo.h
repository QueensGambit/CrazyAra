/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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
    int depth;
    int nodes;
    int nodesPreSearch;
    float elapsedTimeMS;
    float nps;
    bool is_chess960;
    std::vector<Move> pv;
};

extern std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo);

#endif // EVALINFO_H
