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
#include <iostream>
using namespace std;
#include "blazeutil.h"
#include "uci.h"

Node::Node(Board pos, Node *parentNode, unsigned int childIdxForParent):
    pos(pos),
    parentNode(parentNode),
    childIdxOfParent(childIdxForParent)
{

    // generate the legal moves and save them in the list
    for (const ExtMove& move : MoveList<LEGAL>(pos)) {
        legalMoves.push_back(move);
    }

    if (legalMoves.size() == 0) {
        // test if we have a check-mate
        if (parentNode->pos.gives_check(parentNode->legalMoves[childIdxForParent])) {
            value = -1;
            isTerminal = true;
        // we reached a stalmate
        } else {
            cout << "stalmate >> draw!" << endl;
            value = 0;
            isTerminal = true;
        }
    }
    // this can lead to segfault
    else if (pos.is_draw(pos.game_ply())) {
        // reached 50 moves rule
              cout << ">> draw!" << endl;
             value = 0;
             isTerminal = true;
    }
     else {
        // normal game position
             isTerminal = false;
    }

    // # store the initial value prediction of the current board position
    initialValue = value;

    // specify thisTerminale number of direct child nodes from this node
    nbDirectChildNodes = unsigned(int(legalMoves.size()));

    // # visit count of all its child nodes
    childNumberVisits = DynamicVector<float>(nbDirectChildNodes);
    childNumberVisits = 0;

    // total action value estimated by MCTS for each child node also denoted as w
    actionValues = DynamicVector<float>(nbDirectChildNodes);
    actionValues = 0;

    // q: combined action value which is calculated by the averaging over all action values
    // u: exploration metric for each child node
    // (the q and u values are stacked into 1 list in order to speed-up the argmax() operation
    qValues = DynamicVector<float>(nbDirectChildNodes);
    qValues = -1;

//    ones = DynamicVector<float>::Constant(nbDirectChildNodes, 1);
    ones = DynamicVector<float>(nbDirectChildNodes);
    ones = 1;

    divisor = DynamicVector<float>(nbDirectChildNodes);
    divisor = 0.25;


    // number of total visits to this node
    numberVisits = 1;  // we initialize with 1 because if the node was created it must have been visited
    scoreValues = DynamicVector<float>(nbDirectChildNodes);

    childNodes.resize(nbDirectChildNodes); // = std::vector<Node>(nbDirectChildNodes);

//    waitForNNResults.resize(nbDirectChildNodes);
//    waitForNNResults = 0.0f;
    hasNNResults = false;
//    numberWaitingChildNodes = 0;
}

DynamicVector<float> Node::getPolicyProbSmall()
{
    return policyProbSmall;
}

void Node::setPolicyProbSmall(const DynamicVector<float> &value)
{
    policyProbSmall = value;
}

void Node::get_mcts_policy(const float qValueWeight, const float q_value_min_visit_fac, DynamicVector<float>& mctsPolicy)
{
    mctsPolicy = childNumberVisits;
}

DynamicVector<float> Node::getQValues() const
{
    return qValues;
}

void Node::apply_dirichlet_noise_to_prior_policy(const float epsilon, const float alpha)
{
    DynamicVector<float> dirichlet_noise(nbDirectChildNodes);
    dirichlet_noise = 1.0f;
    dirichlet_noise /= nbDirectChildNodes;
    policyProbSmall = (1 - epsilon) * policyProbSmall + epsilon * dirichlet_noise;
}

void Node::setQValues(const DynamicVector<float> &value)
{
    qValues = value;
}

DynamicVector<float> Node::getChildNumberVisits() const
{
    return childNumberVisits;
}

unsigned int Node::getNbDirectChildNodes() const
{
    return nbDirectChildNodes;
}

void Node::setNeuralNetResults(float &value, DynamicVector<float> &pVecSmall)
{
    mtx.lock();
    this->policyProbSmall = pVecSmall;
    this->value = value;
    hasNNResults = true;
    mtx.unlock();
}


//DynamicVector<float> Node::getMCTSPolicy(float q_value_weight )
//{
    
//}

DynamicVector<float> Node::getPVecSmall() const
{
    return policyProbSmall;
}

void Node::setPVecSmall(const DynamicVector<float> &value)
{
    policyProbSmall = value;
}

std::vector<Move> Node::getLegalMoves() const
{
    return legalMoves;
}

void Node::setLegalMoves(const std::vector<Move> &value)
{
    legalMoves = value;
}

void Node::apply_virtual_loss_to_child(unsigned int childIdx, float virtualLoss)
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
    actionValues[childIdx] -= virtualLoss;
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

//    DynamicVector<float> uValues = (
//        cpuct_current
//        * pVecSmall
//        * sqrt(((1 / numberVisits) * (ones + childNumberVisits)))
//    );

//    float pb_u_base = 19652 / 10;
//    float pb_u_init = 1;
//    float pb_u_low = 0.25;
//    float u_init = std::exp((-numberVisits + 1965 + 1) / 1965) / std::exp(1) * (1 - 0.25) + 0.25;
//    divisor = u_init;

    scoreValues = qValues + ( // u-Values
                cpuct_current //cpuct_current
                * policyProbSmall
                * (sqrt(numberVisits) * (ones / (divisor + childNumberVisits)))
            );

//    cout << "scoreValue" << scoreValues << endl;
//    scoreValues += waitForNNResults;
    return argmax(scoreValues); //childIdx;
}

Node *Node::get_child_node(size_t childIdx)
{
    return childNodes[childIdx];
}

void Node::set_child_node(size_t childIdx, Node *newNode)
{
    //    childNodes[childIdx] = Node(); // = newNode;
}

void Node::backup_value(unsigned int childIdx, float virtualLoss, float value)
{
    Node* currentNode = this;
    while (true) {
        currentNode->revert_virtual_loss_and_update(childIdx, virtualLoss, value);
        value = -value;
        childIdx = currentNode->childIdxOfParent;
        currentNode = currentNode->parentNode;
        if (currentNode == nullptr) {
            return;
        }
    }
}


void Node::revert_virtual_loss_and_update(unsigned int childIdx, float virtualLoss, float value)
{
    mtx.lock();
    numberVisits -= virtualLoss - 1;
    childNumberVisits[childIdx] -= virtualLoss - 1;
    actionValues[childIdx] += virtualLoss + value;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

void Node::backup_collision(unsigned int childIdx, float virtualLoss)
{
    Node* currentNode = this;
    while (true) {
        currentNode->revert_virtual_loss(childIdx, virtualLoss);
        childIdx = currentNode->childIdxOfParent;
        currentNode = currentNode->parentNode;
        if (currentNode == nullptr) {
            return;
        }
    }
}

void Node::revert_virtual_loss(unsigned int childIdx, float virtualLoss)
{
    mtx.lock();
    numberVisits -= virtualLoss;
    childNumberVisits[childIdx] -= virtualLoss;
    actionValues[childIdx] += virtualLoss;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

void Node::make_to_root()
{
    parentNode = nullptr;
    childIdxOfParent = -1;
}


ostream &operator<<(ostream &os, const Node *node)
{
    for (size_t childIdx = 0; childIdx < node->getNbDirectChildNodes(); ++childIdx) {
        os << childIdx << ".move " << UCI::move(node->getLegalMoves()[childIdx], false)
             << " n " << node->getChildNumberVisits()[childIdx]
             << " p " << node->getPVecSmall()[childIdx]
             << " Q " << node->getQValues()[childIdx] << endl;
    }
    return os;
}
