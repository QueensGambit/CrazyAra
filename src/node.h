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
 * @file: node.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Class which stores the statistics of all nodes and in the search tree.
 */

#ifndef NODE_H
#define NODE_H

#include <mutex>
#include "position.h"
#include "Dense"
#include "movegen.h"
#include "board.h"
#include "searchthread.h"

#include <blaze/Math.h>

using blaze::HybridVector;
using blaze::DynamicVector;

class Node
{
private:
    std::mutex mtx;
    float value;
    Board pos;
    StateListPtr states;
    DynamicVector<float> pVecSmall;
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;
    DynamicVector<float> tmp_res;

    // dummy
    DynamicVector<float> ones;

    std::vector<Move> legalMoves;
    int nbLegalMoves;
    bool isLeaf;
    unsigned int nbDirectChildNodes;

    float initialValue;
    int numberVisits;
    std::vector<Node> childNodes;

    Node *parentNode;
public:
    Node();
    Node(float value,
         Board pos,
         DynamicVector<float> pVecSmall,
         std::vector<Move> legalMoves,
         bool isLeaf=false);
    DynamicVector<float> getMCTSPolicy(float q_value_weight, float q_value_min_visit_fac);
    DynamicVector<float> getPVecSmall() const;
    void setPVecSmall(const DynamicVector<float> &value);
    std::vector<Move> getLegalMoves() const;
    void setLegalMoves(const std::vector<Move> &value);
    void apply_virtual_loss_to_child(int childIdx, float virtualLoss);
    float getValue() const;
    void setValue(float value);
    size_t select_child_node(float cpuct);
    Node* get_child_node(size_t childIdx);
    void set_child_node(size_t childIdx, Node *newNode);
//    friend class SearchThread;
};

#endif // NODE_H
