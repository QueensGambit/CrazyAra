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
#include "misc.h"


Board* Node::getPos()
{
    return pos;
}

Node::Node(Board *pos, Node *parentNode, unsigned int childIdxForParent, SearchSettings* searchSettings):
    pos(pos),
    parentNode(parentNode),
    childIdxForParent(childIdxForParent),
    checkmateIdx(-1),
    searchSettings(searchSettings)
{
    // generate the legal moves and save them in the list
    for (const ExtMove& move : MoveList<LEGAL>(*pos)) {
        legalMoves.push_back(move);
    }

    check_for_terminal();

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

    // number of total visits to this node
    numberVisits = 1;  // we initialize with 1 because if the node was created it must have been visited

    childNodes.resize(nbDirectChildNodes); // = std::vector<Node>(nbDirectChildNodes);
    policyProbSmall.resize(nbDirectChildNodes);

    //    waitForNNResults.resize(nbDirectChildNodes);
    //    waitForNNResults = 0.0f;
    hasNNResults = false;
    //    numberWaitingChildNodes = 0;
    generator = default_random_engine(r());
}

Node::Node(const Node &b)
{
    value = b.value;
    pos = b.pos;
    nbDirectChildNodes = b.nbDirectChildNodes;
    policyProbSmall.resize(nbDirectChildNodes);
    policyProbSmall = b.policyProbSmall;
    childNumberVisits.resize(nbDirectChildNodes);
    childNumberVisits = 0;
    actionValues.resize(nbDirectChildNodes);
    actionValues = 0;
    qValues.resize(nbDirectChildNodes);
    qValues = -1;
    ones.resize(nbDirectChildNodes);
    ones = b.ones;
    legalMoves = b.legalMoves;
    isTerminal = b.isTerminal;
    initialValue = b.initialValue;
    numberVisits = 1;
    childNodes.resize(nbDirectChildNodes);
    //    parentNode = // is not copied
    //    childIdxForParent = // is not copied
    hasNNResults = b.hasNNResults;
    checkmateIdx = b.checkmateIdx;
    searchSettings = b.searchSettings;
}

Node::~Node()
{
    delete pos;
}

int Node::getNumberVisits() const
{
    return numberVisits;
}

void Node::get_principal_variation(std::vector<Move> &pv)
{
    pv.clear();
    Node* curNode = this;
    size_t childIdx;
    do {
        DynamicVector<float> mctsPolicy(curNode->nbDirectChildNodes);
        curNode->get_mcts_policy(mctsPolicy);
        childIdx = argmax(mctsPolicy);
        pv.push_back(curNode->legalMoves[childIdx]);
        curNode = curNode->childNodes[childIdx];
    } while (curNode != nullptr and !curNode->isTerminal);
}

Key Node::hash_key()
{
    return pos->hash_key();
}

void Node::check_for_terminal()
{
    if (legalMoves.size() == 0) {
        // test if we have a check-mate
        if (parentNode->pos->gives_check(parentNode->legalMoves[childIdxForParent])) {
            value = -1;
            isTerminal = true;
            parentNode->mtx.lock();
            parentNode->checkmateIdx = int(childIdxForParent);
            parentNode->mtx.unlock();
        } else {
            // we reached a stalmate
            value = 0;
            isTerminal = true;
        }
    }
    else if (pos->is_draw(pos->game_ply())) {
        // reached 50 moves rule
        value = 0;
        isTerminal = true;
    }
    else {
        // normal game position
        isTerminal = false;
    }
}

float Node::get_current_cput()
{
//    return std::log((numberVisits + 19652.0f + 1) / 19652.0f) + 2.5f;
    return std::log((numberVisits + searchSettings->cpuctBase + 1) / searchSettings->cpuctBase) + searchSettings->cpuctInit;
}

float Node::get_current_u_divisor()
{
    return searchSettings->uMin - exp(-numberVisits / searchSettings->uBase) * (searchSettings->uMin - searchSettings->uInit);
}

float Node::get_current_q_thresh()
{
    return searchSettings->qThreshMax - exp(-numberVisits / searchSettings->qThreshBase) * (searchSettings->qThreshMax - searchSettings->qThreshInit);
}

DynamicVector<float> Node::get_current_u_values()
{
    return get_current_cput() * policyProbSmall * (sqrt(numberVisits) * (ones / (childNumberVisits + get_current_u_divisor())));
}

void Node::enhance_checks()
{
    if (true) {
        const float thresh_check = 0.1f;
        const float thresh_capture = 0.01f;

        float increment_check = min(thresh_check, max(policyProbSmall)*1.0f); //0.5f);
        float increment_capture = min(thresh_capture, max(policyProbSmall)*0.1f);

        bool update = false;
        for (size_t i = 0; i < nbDirectChildNodes; ++i) {
            if (policyProbSmall[i] < thresh_check && pos->gives_check(legalMoves[i])) {
                policyProbSmall[i] += increment_check;
                update = true;
            }
            if (policyProbSmall[i] < thresh_capture && pos->capture(legalMoves[i])) {
                policyProbSmall[i] += increment_capture;
                update = true;
            }
        }

        if (update)     {
            policyProbSmall /= sum(policyProbSmall);
        }
    }
}

DynamicVector<float> Node::getPolicyProbSmall()
{
    return policyProbSmall;
}

void Node::setPolicyProbSmall(const DynamicVector<float> &value)
{
    policyProbSmall = value;
}

void Node::get_mcts_policy(DynamicVector<float>& mctsPolicy)
{
    if (searchSettings->qValueWeight != 0) {
        DynamicVector<float> qValuePruned(nbDirectChildNodes);
        qValuePruned = (qValues + ones) * 0.5f;
        float visitThresh = get_current_q_thresh() * max(childNumberVisits);
        for (size_t idx; idx < nbDirectChildNodes; ++idx) {
            if (qValuePruned[idx] < visitThresh) {
                qValuePruned[idx] = 0;
            }
        }
        mctsPolicy = (1.0f - searchSettings->qValueWeight) * (childNumberVisits / numberVisits) + searchSettings->qValueWeight * qValuePruned;
    } else {
        mctsPolicy = childNumberVisits / numberVisits;
    }
}

DynamicVector<float> Node::getQValues() const
{
    return qValues;
}

void Node::apply_dirichlet_noise_to_prior_policy()
{
    DynamicVector<float> dirichlet_noise = get_dirichlet_noise(nbDirectChildNodes, searchSettings->dirichletAlpha);
    policyProbSmall = (1 - searchSettings->dirichletEpsilon ) * policyProbSmall + searchSettings->dirichletEpsilon  * dirichlet_noise;
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

//void Node::setNeuralNetResults(float value, DynamicVector<float> &pVecSmall)
//{
//    mtx.lock();
//    this->policyProbSmall = pVecSmall;
//    this->value = value;
//    hasNNResults = true;
//    enhance_checks();
//    mtx.unlock();
//}


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

void Node::apply_virtual_loss_to_child(unsigned int childIdx)
{
    mtx.lock();
    // update the stats of the parent node
    // temporarily reduce the attraction of this node by applying a virtual loss /
    // the effect of virtual loss will be undone if the playout is over
    // virtual increase the number of visits
    numberVisits += searchSettings->virtualLoss;
    childNumberVisits[childIdx] +=  searchSettings->virtualLoss;
    // make it look like if one has lost X games from this node forward where X is the virtual loss value
    // self.action_value[child_idx] -= virtual_loss
    actionValues[childIdx] -=  searchSettings->virtualLoss;
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

size_t Node::select_child_node()
{
    if (checkmateIdx != -1) {
        return size_t(checkmateIdx);
    }
    // find the move according to the q- and u-values for each move
    // calculate the current u values
    // it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
    return argmax(qValues + get_current_u_values());
}

Node *Node::get_child_node(size_t childIdx)
{
    return childNodes[childIdx];
}

void Node::set_child_node(size_t childIdx, Node *newNode)
{
    //    childNodes[childIdx] = Node(); // = newNode;
}

void Node::backup_value(unsigned int childIdx, float value)
{
    Node* currentNode = this;
    while (true) {
        currentNode->revert_virtual_loss_and_update(childIdx, value);
        value = -value;
        childIdx = currentNode->childIdxForParent;
        currentNode = currentNode->parentNode;
        if (currentNode == nullptr) {
            return;
        }
    }
}

void Node::revert_virtual_loss_and_update(unsigned int childIdx, float value)
{
    mtx.lock();
    numberVisits -= searchSettings->virtualLoss - 1;
    childNumberVisits[childIdx] -= searchSettings->virtualLoss - 1;
    actionValues[childIdx] += searchSettings->virtualLoss + value;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

void Node::backup_collision(unsigned int childIdx)
{
    Node* currentNode = this;
    while (true) {
        currentNode->revert_virtual_loss(childIdx);
        childIdx = currentNode->childIdxForParent;
        currentNode = currentNode->parentNode;
        if (currentNode == nullptr) {
            return;
        }
    }
}

void Node::revert_virtual_loss(unsigned int childIdx)
{
    mtx.lock();
    numberVisits -= searchSettings->virtualLoss;
    childNumberVisits[childIdx] -= searchSettings->virtualLoss;
    actionValues[childIdx] += searchSettings->virtualLoss;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

void Node::make_to_root()
{
    parentNode = nullptr;
    childIdxForParent = -1;
}

void Node::delete_subtree(Node *node, unordered_map<Key, Node*>* hashTable)
{
    for (Node *child_node: node->childNodes) {
        if (child_node != nullptr) {
            delete_subtree(child_node, hashTable);
        }
    }
    //    hashTable->erase(node->hash_key());
    delete node;
}

ostream &operator<<(ostream &os, const Node *node)
{
    for (size_t childIdx = 0; childIdx < node->getNbDirectChildNodes(); ++childIdx) {
        os << childIdx << ".move " << UCI::move(node->getLegalMoves()[childIdx], false)
           << " n " << node->getChildNumberVisits()[childIdx]
              << " p " << node->getPVecSmall()[childIdx]
                 << " Q " << node->getQValues()[childIdx] << endl;
    }
    os << " sum: " << sum(node->getPVecSmall()) << endl;

    return os;
}
