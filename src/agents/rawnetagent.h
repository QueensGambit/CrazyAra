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
 * @file: rawnetagent.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * The raw network uses the the single network prediction for it's evaluation.
 * No mcts search is being done.
 */

#ifndef RAWNETAGENT_H
#define RAWNETAGENT_H

#include "position.h"
#include "../evalinfo.h"
#include "../node.h"
#include <thread>
#include "../board.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "agent.h"

class RawNetAgent: public Agent
{
private:
    NeuralNetAPI *net;
    PlaySettings playSettings;
    float input_planes[NB_VALUES_TOTAL];
    NDArray valueOutput = NDArray(Shape(1, 1), Context::cpu());
    NDArray probOutputs = NDArray(Shape(1, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH), Context::cpu());

public:
    RawNetAgent(NeuralNetAPI *net, PlaySettings playSettings,
                float temperature, unsigned int temperature_moves, bool verbose);

    EvalInfo evalute_board_state(Board *pos);

};

#endif // RAWNETAGENT_H
