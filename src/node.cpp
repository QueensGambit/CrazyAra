/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: node.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "node.h"
#include <iostream>
using namespace std;
#include "util/blazeutil.h"
#include "uci.h"
#include "misc.h"
#include <experimental/random>

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

    // number of total visits to this node
    numberVisits = 1;  // we initialize with 1 because if the node was created it must have been visited

    childNodes.resize(nbDirectChildNodes);
    policyProbSmall.resize(nbDirectChildNodes);

    //    waitForNNResults.resize(nbDirectChildNodes);
    //    waitForNNResults = 0.0f;
    hasNNResults = false;
    //    numberWaitingChildNodes = 0;
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

int Node::find_move_idx(Move move)
{
    int idx = 0;
    for (Move childMove : legalMoves) {
        if (childMove == move) {
            return idx;
        }
        ++idx;
    }
    return -1;
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
    return log((numberVisits + searchSettings->cpuctBase + 1) / searchSettings->cpuctBase) + searchSettings->cpuctInit;
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
    return get_current_cput() * policyProbSmall * (sqrt(numberVisits) / (childNumberVisits + get_current_u_divisor()));
}

bool Node::enhance_checks(const float incrementCheck, float threshCheck)
{
    bool update = false;
    for (size_t i = 0; i < nbDirectChildNodes; ++i) {
        if (policyProbSmall[i] < threshCheck && pos->gives_check(legalMoves[i])) {
            policyProbSmall[i] += incrementCheck;
            update = true;
        }
    }
    return update;
}

bool Node::enhance_captures(const float incrementCapture, float threshCapture)
{
    bool update = false;
    for (size_t i = 0; i < nbDirectChildNodes; ++i) {
        if (policyProbSmall[i] < threshCapture && pos->capture(legalMoves[i])) {
            policyProbSmall[i] += incrementCapture;
            update = true;
        }
    }

    return update;
}

void Node::enhance_moves(const float threshCheck, const float checkFactor, const float threshCapture, const float captureFactor)
{
    float maxPolicyValue = max(policyProbSmall);
    bool checkUpdate = false;
    bool captureUpdate = false;

    if (searchSettings->enhanceChecks) {
        float incrementCheck = min(threshCheck, maxPolicyValue*checkFactor);
        checkUpdate = enhance_checks(incrementCheck, threshCheck);
    }
    if (searchSettings->enhanceCaptures) {
        float incrementCapture = min(threshCapture, maxPolicyValue*captureFactor);
        captureUpdate = enhance_captures(incrementCapture, threshCapture);
    }

    if (checkUpdate || captureUpdate) {
        policyProbSmall /= sum(policyProbSmall);
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
    if (searchSettings->qValueWeight > 0) {
        DynamicVector<float> qValuePruned(nbDirectChildNodes);
        qValuePruned = (qValues + 1) * 0.5f;
        float visitThresh = get_current_q_thresh() * max(childNumberVisits);
        for (size_t idx = 0; idx < nbDirectChildNodes; ++idx) {
            if (childNumberVisits[idx] < visitThresh) {
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
    assert(childIdx < nbDirectChildNodes);
    return childNodes[childIdx];
}

void Node::backup_value(unsigned int childIdx, float value)
{
    Node* currentNode = this;
    do {
        currentNode->revert_virtual_loss_and_update(childIdx, value);
        childIdx = currentNode->childIdxForParent;
        value = -value;
        currentNode = currentNode->parentNode;
    } while(currentNode != nullptr);
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
    do {
        currentNode->revert_virtual_loss(childIdx);
        childIdx = currentNode->childIdxForParent;
        currentNode = currentNode->parentNode;
    } while (currentNode != nullptr);
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
}

void Node::delete_subtree(Node *node)
{
    for (Node *childNode: node->childNodes) {
        if (childNode != nullptr) {
            delete_subtree(childNode);
        }
    }
    delete node;
}

void Node::delete_subtree_and_hash_entries(Node *node, unordered_map<Key, Node*>* hashTable)
{
    for (Node *childNode: node->childNodes) {
        if (childNode != nullptr) {
            delete_subtree_and_hash_entries(childNode, hashTable);
        }
    }

    auto it = hashTable->find(node->pos->hash_key());
    if(it != hashTable->end()) {
        hashTable->erase(node->hash_key());
    }
    delete node;
}

void Node::delete_sibling_subtrees(unordered_map<Key, Node*>* hashTable)
{
    if (parentNode != nullptr) {
        sync_cout << "info string delete unused subtrees" << sync_endl;
        size_t i = 0;
        for (Node *childNode: parentNode->childNodes) {
            if (childNode != nullptr && childNode != this) {
                Node::delete_subtree_and_hash_entries(childNode, hashTable);
                parentNode->childNodes[i] = nullptr;
            }
            ++i;
        }
    }
}

ostream &operator<<(ostream &os, const Node *node)
{
    for (size_t childIdx = 0; childIdx < node->getNbDirectChildNodes(); ++childIdx) {
        os << childIdx << ".move " << UCI::move(node->getLegalMoves()[childIdx], false)
           << " n " << node->getChildNumberVisits()[childIdx]
              << " p " << node->getPVecSmall()[childIdx]
                 << " Q " << node->getQValues()[childIdx] << endl;
    }
    os << " initial value: " << node->getValue() << endl;
    return os;
}
