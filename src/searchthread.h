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
 * @file: searchthread.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Handles the functionality of a single search thread in the tree
 */

#ifndef SEARCHTHREAD_H
#define SEARCHTHREAD_H

#include "node.h"
#include "constants.h"
//#include "neuralnetapi.h"


class SearchThread
{
private:
//    Node* rootNode;
//    NeuralNetAPI *net;

    bool isRunning;
    float *input_planes; //[NB_VALUES_TOTAL]; //34][8][8];
//    StateListPtr& states;

public:
    SearchThread(unsigned int miniBatchSize);
    void go(StateListPtr& states);
    void run_single_playout();
    void create_mini_batch();

};

#endif // SEARCHTHREAD_H
