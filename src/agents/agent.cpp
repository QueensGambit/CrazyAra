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
 * @file: agent.cpp
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "agent.h"
#include <chrono>
#include "misc.h"
#include "uci.h"

Agent::Agent(float temperature, unsigned int temperature_moves, bool verbose)
{
    this->temperature = temperature;
    this->temperature_moves = temperature_moves;
    this->verbose = verbose;
}

void Agent::perform_action(Board *pos, SearchLimits *searchLimits)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    this->searchLimits = searchLimits;
    EvalInfo eval_info = this->evalute_board_state(pos);
    sync_cout << "end time" << sync_endl;
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    eval_info.elapsedTimeMS = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    sync_cout << eval_info << sync_endl;
    sync_cout << "bestmove " << UCI::move(eval_info.pv[0], pos->is_chess960()) << sync_endl;

}
