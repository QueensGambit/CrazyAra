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
 * Handles the functionality of a single search thread in the tree.
 */

#ifndef SEARCHTHREAD_H
#define SEARCHTHREAD_H

#include "node.h"
#include "constants.h"
#include "neuralnetapi.h"


class SearchThread
{
private:
    Node* rootNode;
    NeuralNetAPI *netBatch;
    unsigned int batchSize;

    float *inputPlanes; //[NB_VALUES_TOTAL]; //34][8][8];
//    StateListPtr& states; // = StateListPtr(new std::deque<StateInfo>(1)); // Drop old and create a new one
//    states
    // list of all node objects which have been selected for expansion
    std::vector<Node*> newNodes;
    std::vector<Node*> collisionNodes;
    std::vector<Node*> terminalNodes;

    std::vector<Node*> parentNode;


    // stores the corresponding value-Outputs and probability-Outputs of the nodes stored in the vector "newNodes"
    // sufficient memory according to the batch-size will be allocated in the constructor
    NDArray valueOutputs;
    NDArray probOutputs;

    const float virtualLoss;
    bool isRunning;

    unordered_map<Key, Node*> *hashTable;

    inline Node* get_new_child_to_evaluate(unsigned int &childIdx, bool &isCollision,  bool &isTerminal, size_t &depth);
    void set_NN_results_to_child_nodes();
    void backup_value_outputs(const float virtualLoss);
    void backup_collisions(const float virtualLoss);

    void revert_virtual_loss_for_collision(const float virtualLoss);

public:
    SearchThread(NeuralNetAPI *netBatch, unsigned int batchSize, const float virtualLoss, unordered_map<Key, Node*> *hashTable);
    void go();
    void run_single_playout();
    void create_mini_batch();

    void setRootNode(Node *value);
};

#endif // SEARCHTHREAD_H
