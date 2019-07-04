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
 * @file: agent.h
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Abstract class for defining a playing agent.
 */

#ifndef AGENT_H
#define AGENT_H

#include "../board.h"
#include "../evalinfo.h"
#include "config/searchlimits.h"

class Agent
{
protected:
    float temperature;
    float current_temperature;
    unsigned int temperature_moves;
    bool verbose;
    SearchLimits *searchLimits;
public:
    Agent(float temperature, unsigned int temperature_moves, bool verbose);

    /**
     * @brief perform_action Selects an action based on the evaluation result
     * @param pos Board position to evaluate
     * @param limits Pointer to the search limit
     */
    void perform_action(Board *pos, SearchLimits *searchLimits);

    /**
     * @brief evalute_board_state Pure virtual method which acts as an interface for all agents
     * @param pos Board position to evaluate
     * @return Evaluation information
     */
    virtual EvalInfo evalute_board_state(Board *pos) = 0;
};

#endif // AGENT_H
