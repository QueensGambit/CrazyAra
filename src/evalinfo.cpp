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
 * @file: evalinfo.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "evalinfo.h"
#include "uci.h"

EvalInfo::EvalInfo()
{

}

std::ostream& operator<<(std::ostream& os, const EvalInfo& eval_info) {

//  os << "cp " << eval_info.centipawns << " depth " << eval_info.depth;

  os << "info score cp " << eval_info.centipawns
     << " depth " << eval_info.depth
     << " nodes " << eval_info.nodes
     << " time " << eval_info.elapsedTimeMS
        // + 0.5 and int() is a simple way for rounding to the first decimal
     << " nps " << int(((eval_info.nodes-eval_info.nodesPreSearch) / (eval_info.elapsedTimeMS / 1000.0f)) + 0.5f)
     << " pv " << UCI::move(eval_info.pv[0], eval_info.is_chess960);

  return os;
}
