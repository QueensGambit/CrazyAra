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
 * Created on 28.08.2019
 * @author: queensgambit
 */

#include "node.h"
#include "util/blazeutil.h" // get_dirichlet_noise()
#include "constants.h"
#include "../util/sfutil.h"
#include "../util/communication.h"

Node::Node(Board *pos, Node *parentNode, size_t childIdxForParent, SearchSettings* searchSettings):
    pos(pos),
    parentNode(parentNode),
    visits(1),
    noVisitIdx(1),
    isTerminal(false),
    childIdxForParent(childIdxForParent),
    hasNNResults(false),
    isFullyExpanded(false),
    checkmateIdx(-1),
    searchSettings(searchSettings)
{
    fill_child_node_moves();

    // specify thisTerminale number of direct child nodes from this node
    numberChildNodes = legalMoves.size();

    check_for_terminal();

    // # visit count of all its child nodes
    childNumberVisits = DynamicVector<float>(numberChildNodes);
    childNumberVisits = 0;

    // total action value estimated by MCTS for each child node also denoted as w
    actionValues = DynamicVector<float>(numberChildNodes);
    actionValues = 0;

    // q: combined action value which is calculated by the averaging over all action values
    // u: exploration metric for each child node
    // (the q and u values are stacked into 1 list in order to speed-up the argmax() operation
    qValues = DynamicVector<float>(numberChildNodes);
    qValues = -1;

    childNodes.resize(numberChildNodes, nullptr);
    policyProbSmall.resize(numberChildNodes);
}

Node::Node(const Node &b)
{
    value = b.value;
    //    pos = b.pos;
    numberChildNodes = b.numberChildNodes;
    policyProbSmall.resize(numberChildNodes);
    policyProbSmall = b.policyProbSmall;
    childNumberVisits.resize(numberChildNodes);
    childNumberVisits = 0;
    actionValues.resize(numberChildNodes);
    actionValues = 0;
    qValues.resize(numberChildNodes);
    qValues = -1;
    legalMoves = b.legalMoves;
    isTerminal = b.isTerminal;
    //    initialValue = b.initialValue;
    visits = 1;
    childNodes.resize(numberChildNodes);
    //    parentNode = // is not copied
    //    childIdxForParent = // is not copied
    noVisitIdx = 1; // reset counter
    isTerminal = b.isTerminal;
    hasNNResults = b.hasNNResults;
    checkmateIdx = -1; //b.checkmateIdx;
    searchSettings = b.searchSettings;
    isFullyExpanded = false;
}

void Node::fill_child_node_moves()
{
    // generate the legal moves and save them in the list
    for (const ExtMove& move : MoveList<LEGAL>(*pos)) {
        legalMoves.push_back(move);
    }
}

void Node::mark_nodes_as_fully_expanded()
{
    noVisitIdx = numberChildNodes;
    isFullyExpanded = true;
}

Node::~Node()
{
    delete pos;
}

void Node::sort_moves_by_probabilities()
{
    auto p = sort_permutation(policyProbSmall, std::greater<float>());

    apply_permutation_in_place(policyProbSmall, p);
    apply_permutation_in_place(legalMoves, p);
}

Move Node::get_move(size_t childIdx) const
{
    return legalMoves[childIdx];
}

vector<Node*> Node::get_child_nodes() const
{
    return childNodes;
}

bool Node::is_terminal() const
{
    return isTerminal;
}

bool Node::has_nn_results() const
{
    return hasNNResults;
}

Color Node::side_to_move() const
{
    return pos->side_to_move();
}

Board* Node::get_pos() const
{
    return pos;
}

void Node::apply_virtual_loss_to_child(size_t childIdx)
{
    // update the stats of the parent node
    // temporarily reduce the attraction of this node by applying a virtual loss /
    // the effect of virtual loss will be undone if the playout is over
    // virtual increase the number of visits
    visits += searchSettings->virtualLoss;
    childNumberVisits[childIdx] +=  searchSettings->virtualLoss;
    // make it look like if one has lost X games from this node forward where X is the virtual loss value
    actionValues[childIdx] -=  searchSettings->virtualLoss;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
}

Node *Node::get_parent_node() const
{
    return parentNode;
}

void Node::increment_visits()
{
    ++visits;
}

void Node::increment_no_visit_idx()
{
    mtx.lock();
    if (noVisitIdx < numberChildNodes) {
        ++noVisitIdx;
        isFullyExpanded = true;
    }
    mtx.unlock();
}

bool Node::is_fully_expanded() const {
    return isFullyExpanded;
}

float Node::get_current_cput()
{
    return log((visits + searchSettings->cpuctBase + 1) / searchSettings->cpuctBase) + searchSettings->cpuctInit;
}

float Node::get_current_u_divisor()
{
    return searchSettings->uMin - exp(-visits / searchSettings->uBase) * (searchSettings->uMin - searchSettings->uInit);
}

float Node::get_value() const
{
    return value;
}

Key Node::hash_key() const
{
    return pos->hash_key();
}

size_t Node::get_number_child_nodes() const
{
    return numberChildNodes;
}

float Node::get_visits() const
{
    return visits;
}

float Node::get_u_parent_factor() const
{
    return uParentFactor;
}

float Node::get_u_divisor_summand() const
{
#ifdef USE_RL
    return 1.0f;
#else
    return uDivisorSummand;
#endif
}

void Node::backup_value(size_t childIdx, float value)
{
    Node* currentNode = this;
    do {
        currentNode->revert_virtual_loss_and_update(childIdx, value);
        childIdx = currentNode->childIdxForParent;
        value = -value;
        currentNode = currentNode->parentNode;
    } while(currentNode != nullptr);
}

void Node::revert_virtual_loss_and_update(size_t childIdx, float value)
{
    mtx.lock();
    visits -= searchSettings->virtualLoss - 1;
    childNumberVisits[childIdx] -= searchSettings->virtualLoss - 1;
    actionValues[childIdx] += searchSettings->virtualLoss + value;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

void Node::backup_collision(size_t childIdx)
{
    Node* currentNode = this;
    do {
        currentNode->revert_virtual_loss(childIdx);
        childIdx = currentNode->childIdxForParent;
        currentNode = currentNode->parentNode;
    } while (currentNode != nullptr);
}

void Node::revert_virtual_loss(size_t childIdx)
{
    mtx.lock();
    visits -= searchSettings->virtualLoss;
    childNumberVisits[childIdx] -= searchSettings->virtualLoss;
    actionValues[childIdx] += searchSettings->virtualLoss;
    qValues[childIdx] = actionValues[childIdx] / childNumberVisits[childIdx];
    mtx.unlock();
}

SearchSettings* Node::get_search_settings() const
{
    return searchSettings;
}

void Node::set_parent_node(Node* value)
{
    parentNode = value;
}

size_t Node::get_no_visit_idx() const
{
    return noVisitIdx;
}

DynamicVector<float>& Node::get_policy_prob_small()
{
    return policyProbSmall;
}

void Node::set_value(float value)
{
    this->value = value;
}

size_t Node::get_child_idx_for_parent() const
{
    return childIdxForParent;
}

void Node::add_new_child_node(Node *newNode, size_t childIdx)
{
    mtx.lock();
    childNodes[childIdx] = newNode;
    mtx.unlock();
}

void Node::add_transposition_child_node(Node* newNode, Board *newPos, size_t childIdx)
{
    newNode->mtx.lock();
    newNode->pos = newPos;
    newNode->parentNode = this;
    newNode->childIdxForParent = childIdx;
    newNode->mtx.unlock();
    mtx.lock();
    childNodes[childIdx] = newNode;
    mtx.unlock();
}

float Node::max_policy_prob()
{
    return max(policyProbSmall);
}

size_t Node::max_q_child()
{
    return argmax(qValues);
}

float Node::updated_value_eval()
{
    return qValues[argmax(childNumberVisits)];
}

std::vector<Move> Node::get_legal_moves() const
{
    return legalMoves;
}

int Node::get_checkmate_idx() const
{
    return checkmateIdx;
}

DynamicVector<float> Node::get_child_number_visits() const
{
    return childNumberVisits;
}

void Node::enable_has_nn_results()
{
    hasNNResults = true;
}

void Node::check_for_terminal()
{
    if (numberChildNodes == 0) {
        isTerminal = true;
#ifdef ANTI
        if (pos->is_anti()) {
            // a stalmate is a win in antichess
            value = WIN;
            return;
        }
#endif
        // test if we have a check-mate
        if (parentNode->pos->gives_check(parentNode->legalMoves[childIdxForParent])) {
            value = LOSS;
            isTerminal = true;
            parentNode->checkmateIdx = int(childIdxForParent);
            return;
        }
        // we reached a stalmate
        value = DRAW;
        return;
    }
#ifdef ANTI
    if (pos->is_anti()) {
        if (pos->is_anti_win()) {
            isTerminal = true;
            value = WIN;
            return;
        }
        if (pos->is_anti_loss()) {
            isTerminal = true;
            value = LOSS;
            parentNode->checkmateIdx = int(childIdxForParent);
            return;
        }
    }
#endif
    if (pos->can_claim_3fold_repetition() || pos->is_50_move_rule_draw()) {
        // reached 3-fold-repetition or 50 moves rule draw
        value = DRAW;
        isTerminal = true;
        return;
    }
    // normal game position
    //    isTerminal = false;  // is the default value
}

void Node::make_to_root()
{
    parentNode = nullptr;
}

void Node::lock()
{
    mtx.lock();
}

void Node::unlock()
{
    mtx.unlock();
}

void Node::apply_dirichlet_noise_to_prior_policy()
{
    DynamicVector<float> dirichlet_noise = get_dirichlet_noise(numberChildNodes, searchSettings->dirichletAlpha);
    policyProbSmall = (1 - searchSettings->dirichletEpsilon ) * policyProbSmall + searchSettings->dirichletEpsilon * dirichlet_noise;
}

void Node::apply_temperature_to_prior_policy(float temperature)
{
    apply_temperature(policyProbSmall, temperature);
}

void Node::set_probabilities_for_moves(const float *data, unordered_map<Move, size_t>& moveLookup)
{
    //    // allocate sufficient memory -> is assumed that it has already been done
    assert(legalMoves.size() == policyProbSmall.size());
    for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx) {
        // retrieve vector index from look-up table
        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        policyProbSmall[mvIdx] = data[moveLookup[legalMoves[mvIdx]]];
    }
}

void Node::apply_softmax_to_policy()
{
    softmax(policyProbSmall);
}

void Node::enhance_moves()
{
    if (!searchSettings->enhanceChecks && !searchSettings->enhanceCaptures) {
        return;
    }

    float maxPolicyValue = max(policyProbSmall);
    bool checkUpdate = false;
    bool captureUpdate = false;

    if (searchSettings->enhanceChecks) {
        checkUpdate = enhance_move_type(min(searchSettings->threshCheck, maxPolicyValue*searchSettings->checkFactor),
                                        searchSettings->threshCheck, pos, legalMoves, isCheck, policyProbSmall);
    }
    if (searchSettings->enhanceCaptures) {
        captureUpdate = enhance_move_type(min(searchSettings->threshCapture, maxPolicyValue*searchSettings->captureFactor),
                                          searchSettings->threshCheck, pos, legalMoves, isCapture, policyProbSmall);
    }

    if (checkUpdate || captureUpdate) {
        policyProbSmall /= sum(policyProbSmall);
    }
}

bool enhance_move_type(float increment, float thresh, const Board* pos, const vector<Move>& legalMoves, vFunctionMoveType func, DynamicVector<float>& policyProbSmall)
{
    bool update = false;
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        if (policyProbSmall[i] < thresh && func(pos, legalMoves[i])) {
            policyProbSmall[i] += increment;
            update = true;
        }
    }
    return update;
}

bool isCheck(const Board* pos, Move move)
{
    return pos->gives_check(move);
}

bool isCapture(const Board* pos, Move move)
{
    return pos->capture(move);
}

DynamicVector<float> Node::get_current_u_values()
{
    return get_current_cput() * blaze::subvector(policyProbSmall, 0, noVisitIdx) * (sqrt(visits) / (blaze::subvector(childNumberVisits, 0, noVisitIdx) + get_current_u_divisor()));
}

Node *Node::get_child_node(size_t childIdx)
{
    return childNodes[childIdx];
}

void Node::get_mcts_policy(DynamicVector<float>& mctsPolicy) const
{
    const float qValueWeight = searchSettings->qValueWeight;

    if (qValueWeight > 0) {
        DynamicVector<float> qValuePruned = qValues;
        qValuePruned = (qValuePruned + 1) * 0.5f;
        const DynamicVector<float> normalizedVisits = childNumberVisits / visits;
        const float quantile = get_quantile(normalizedVisits, 0.25f);
        for (size_t idx = 0; idx < numberChildNodes; ++idx) {
            if (childNumberVisits[idx] < quantile) {
                qValuePruned[idx] = 0;
            }
        }
        mctsPolicy = (1.0f - qValueWeight) * normalizedVisits + qValueWeight * qValuePruned;
        mctsPolicy /= sum(mctsPolicy);
    } else {
        mctsPolicy = childNumberVisits / visits;
    }
}

void Node::get_principal_variation(vector<Move>& pv) const
{
    pv.clear();
    const Node* curNode = this;
    do {
        DynamicVector<float> mctsPolicy(curNode->get_number_child_nodes());
        curNode->get_mcts_policy(mctsPolicy);
        size_t childIdx = argmax(mctsPolicy);
        pv.push_back(curNode->get_move(childIdx));
        curNode = curNode->childNodes[childIdx];
    } while (curNode != nullptr && !curNode->is_terminal());
}

size_t Node::select_child_node()
{
    if (checkmateIdx != -1) {
        return size_t(checkmateIdx);
    }

    if (visits == 1) {
        sort_moves_by_probabilities();
    }
    // find the move according to the q- and u-values for each move
    // calculate the current u values
    // it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
    return argmax(blaze::subvector(qValues, 0, noVisitIdx) + get_current_u_values());
}

ostream& operator<<(ostream &os, const Node *node)
{
    for (size_t childIdx = 0; childIdx < node->get_number_child_nodes(); ++childIdx) {
        os << childIdx << ".move " << UCI::move(node->get_legal_moves()[childIdx], false)
           << "\tn " << node->childNumberVisits[childIdx]
              << "\tp " << node->policyProbSmall[childIdx]
                 << "\tQ " << node->qValues[childIdx]
                    << "\tterminal "<< node->is_terminal() << endl;
    }
    os << " initial value: " << node->get_value() << endl;
    return os;
}


void delete_sibling_subtrees(Node* node, unordered_map<Key, Node*>* hashTable)
{
    if (node->get_parent_node() != nullptr) {
        info_string("delete unused subtrees");
        for (Node* childNode: node->get_parent_node()->get_child_nodes()) {
            if (childNode != node) {
                delete_subtree_and_hash_entries(childNode, hashTable);
            }
        }
    }
}

void delete_subtree_and_hash_entries(Node* node, unordered_map<Key, Node*>* hashTable)
{
    if (node == nullptr) {
        return;
    }
    // if the current node hasn't been expanded or is a terminal node then childNodes is empty and the recursion ends
    for (Node* childNode: node->get_child_nodes()) {
        delete_subtree_and_hash_entries(childNode, hashTable);
    }
    // the board position is only filled if the node has been extended
    auto it = hashTable->find(node->hash_key());
    if(it != hashTable->end()) {
        hashTable->erase(node->hash_key());
    }
    delete node;
}

float get_visits(Node* node)
{
    return node->get_visits();
}

float get_current_q_thresh(const SearchSettings* searchSettings, int numberVisits)
{
    return searchSettings->qThreshMax - exp(-numberVisits / searchSettings->qThreshBase) * (searchSettings->qThreshMax - searchSettings->qThreshInit);
}

double get_current_cput(float numberVisits, float cpuctBase, float cpuctInit)
{
    return log((numberVisits + cpuctBase + 1) / cpuctBase) + cpuctInit;
}

float get_current_u_divisor(float numberVisits, float uMin, float uInit, float uBase)
{
    return uMin - exp(-numberVisits / uBase) * (uMin - uInit);
}

void print_node_statistics(const Node* node)
{
    info_string("position", node->get_pos()->fen());
    cout << node << endl;
}

Result get_terminal_node_result(const Node *terminalNode)
{
    assert(terminalNode->is_terminal());
    if (int(terminalNode->get_value()) == 0) {
        return DRAWN;
    }
    else if ( terminalNode->side_to_move() == BLACK) {
        return WHITE_WIN;
    }
    return BLACK_WIN;
}
