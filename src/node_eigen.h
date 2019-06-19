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

class Node
{
private:
    std::mutex mtx;
    float value;
    Board pos;
    Eigen::ArrayXf pVecSmall;
    Eigen::ArrayXf childNumberVisits;
    Eigen::ArrayXf actionValues;
    Eigen::ArrayXf qValues;

    // dummy
    Eigen::ArrayXf ones;

    std::vector<Move> legalMoves;
    int nbLegalMoves;
    bool isLeaf;
    unsigned int nbDirectChildNodes;

    float initialValue;
    int numberVisits;
    std::vector<Node> childNodes;
public:
    Node();
    Node(
    float value,
    Board pos,
    Eigen::ArrayXf pVecSmall,
    std::vector<Move> legalMoves,
    bool isLeaf=false);
    Eigen::ArrayXf getMCTSPolicy(float q_value_weight, float q_value_min_visit_fac);
    Eigen::ArrayXf getPVecSmall() const;
    void setPVecSmall(const Eigen::ArrayXf &value);
    std::vector<Move> getLegalMoves() const;
    void setLegalMoves(const std::vector<Move> &value);
    void apply_virtual_loss_to_child(int childIdx, float virtualLoss);
    float getValue() const;
    void setValue(float value);
    size_t select_child_node(float cpuct);
    Node* get_child_node(size_t childIdx);
    void set_child_node(size_t childIdx, Node *newNode);
};

#endif // NODE_H

