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
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#include "inputrepresentation.h"

SearchThread::SearchThread(unsigned int miniBatchSize)
{
    input_planes = new float[miniBatchSize * NB_VALUES_TOTAL];
//    states = nullptr;
}

void SearchThread::go(StateListPtr& states)
{
    while(isRunning) {
        run_single_playout();
    }
}

void SearchThread::run_single_playout()
{
    create_mini_batch();
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to mini-batch
    Node *currentNode;
    size_t childIdx;

    currentNode = nullptr; //rootNode;
    do {
        childIdx = currentNode->select_child_node(2.5);
    }
    while(currentNode->get_child_node(childIdx) != nullptr);


//    Board new_pos(currentNode->pos);
//    currentNode->pos.do_move(currentNode->legalMoves[childIdx], states->back());

//    board_to_planes(new_pos, 0, true, input_planes); //input_planes_start);

    float value;
//    NDArray probArray = net->predict_single(input_planes, value);


//    currentNode->[childIdx] = Node();
}
