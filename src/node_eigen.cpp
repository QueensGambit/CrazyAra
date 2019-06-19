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
 * @file: node.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "node.h"

Node::Node()
{

}

Node::Node(float value, Board pos, Eigen::ArrayXf pVecSmall, std::vector<Move> legalMoves, bool isLeaf):
    pos(pos),
    value(value),
    pVecSmall(pVecSmall),
    legalMoves(legalMoves),
    isLeaf(isLeaf)
{
    // # store the initial value prediction of the current board position
    initialValue = value;

    if (isLeaf) {
        nbDirectChildNodes = 0;
    } else {
        // specify the number of direct child nodes from this node
        nbDirectChildNodes = unsigned(int(legalMoves.size()));
    }

    // # visit count of all its child nodes
    childNumberVisits = Eigen::ArrayXf::Zero(nbDirectChildNodes);

    // total action value estimated by MCTS for each child node also denoted as w
    actionValues = Eigen::ArrayXf::Zero(nbDirectChildNodes);
    // q: combined action value which is calculated by the averaging over all action values
    // u: exploration metric for each child node
    // (the q and u values are stacked into 1 list in order to speed-up the argmax() operation
    qValues = Eigen::ArrayXf::Zero(nbDirectChildNodes);

    ones = Eigen::ArrayXf::Constant(nbDirectChildNodes, 1);
    // number of total visits to this node
    numberVisits = 1;  // we initialize with 1 because if the node was created it must have been visited

    childNodes = std::vector<Node>(nbDirectChildNodes);
}

//Eigen::ArrayXf Node::getMCTSPolicy(float q_value_weight )
//{
    
//}

Eigen::ArrayXf Node::getPVecSmall() const
{
    return pVecSmall;
}

void Node::setPVecSmall(const Eigen::ArrayXf &value)
{
    pVecSmall = value;
}

std::vector<Move> Node::getLegalMoves() const
{
    return legalMoves;
}

void Node::setLegalMoves(const std::vector<Move> &value)
{
    legalMoves = value;
}

void Node::apply_virtual_loss_to_child(int childIdx, float virtualLoss)
{
    mtx.lock();
    // update the stats of the parent node
    // temporarily reduce the attraction of this node by applying a virtual loss /
    // the effect of virtual loss will be undone if the playout is over
    // virtual increase the number of visits
    numberVisits += virtualLoss;
    childNumberVisits[childIdx] += virtualLoss;
    // make it look like if one has lost X games from this node forward where X is the virtual loss value
    // self.action_value[child_idx] -= virtual_loss
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

float Node::getValue() const
{
    return value;
}

void Node::setValue(float value)
{
    value = value;
}

size_t Node::select_child_node(float cpuct)
{
    // find the move according to the q- and u-values for each move
    float pbCBase = 19652;
    float pbCInit = cpuct;

    float cpuct_current = std::log((numberVisits + pbCBase + 1) / pbCBase) + pbCInit;
    // calculate the current u values
    // it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
    Eigen::ArrayXf uValues = (
        cpuct_current
        * pVecSmall
        * (numberVisits / (ones + childNumberVisits)).sqrt()
    );
    size_t childIdx;
    (qValues + uValues).maxCoeff(&childIdx);
    return childIdx;
}

Node *Node::get_child_node(size_t childIdx)
{
    return &childNodes[childIdx];
}

void Node::set_child_node(size_t childIdx, Node *newNode)
{
//    childNodes[childIdx] = Node(); // = newNode;
}


