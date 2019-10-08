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

Node::Node(Node *parentNode, Move move,  SearchSettings* searchSettings):
    parentNode(parentNode),
    move(move),
    value(0.0f),
    probValue(0.0f),
    qValue(-1.0f),
    actionValue(0.0f),
    visits(0.0f),
    virtualLossCounter(0),
    numberExpandedNodes(0),
    isTerminal(false),  // will be later recomputed
    isExpanded(false),
    hasNNResults(false),
    isCalibrated(false),
    areChildNodesSorted(false),
    isFullyExpanded(false),
    uParentFactor(0.0f),
    uDivisorSummand(0.0f),
    checkmateNode(nullptr),
    searchSettings(searchSettings)
{

}

Node::Node(Board *pos, Node *parentNode, Move move,  SearchSettings* searchSettings):
    Node(parentNode, move, searchSettings)
{
    this->pos = pos;
}

Node::~Node()
{
    if (isExpanded) {
        delete pos;
    }
}

void Node::operator=(const Node& b)
{
    value = b.value;
    // the position has already been expanded, no copy is required
    numberChildNodes = b.numberChildNodes;
    // copy the probability values for all child nodes
    for (Node* node : b.childNodes) {
        Node* newChild = new Node(this, node->move, node->searchSettings);
        newChild->probValue = node->probValue;
        childNodes.push_back(newChild);
    }
    isTerminal = b.isTerminal;
    value = b.value;
    visits = 1;
    childNodes.resize(numberChildNodes);
    //    parentNode = // is not copied
    hasNNResults = b.hasNNResults;
    searchSettings = b.searchSettings;
    isTerminal = b.isTerminal;
    isExpanded = true;
    checkmateNode = nullptr;  // might be set later
    uDivisorSummand = 0.0f;
    uParentFactor = 0.0f;
}

void Node::sort_child_nodes_by_probabilities()
{
    sort(childNodes.begin(), childNodes.end(), prob_value_comparision);
    areChildNodesSorted = true;
    isCalibrated = true;
}

void Node::sort_child_nodes_by_q_plus_u()
{
    update_u_divisor();
    update_u_parent_factor();
    sort(childNodes.begin(), childNodes.end(), q_plus_u_comparision);
    areChildNodesSorted = true;
    isCalibrated = true;
}

void Node::calibrate_child_node_order()
{
    // sort the child nodes which have already been expanded
    // we are only interested in the first two highest Q+U values
    // the unvisited nodes are still in order respective to themselves
    // however, it is is necessary to check if the first two nodes are worthy to be elected
    size_t idx = min(numberChildNodes, numberExpandedNodes+2);
    partial_sort(childNodes.begin(), childNodes.begin()+2, childNodes.begin()+idx,
                 q_plus_u_comparision);

    // sorting
    isCalibrated = true;

    // DEBUG
    //            if (!is_ordering_correct(childNodes)) {
    //                cout << "nodeIdxUpdate: " << numberExpandedNodes << endl;
    //                print_node_statistics(this);
    //                assert(is_ordering_correct(childNodes));
    //            }
}

void Node::expand()
{
    create_child_nodes();
    numberChildNodes = childNodes.size();
    check_for_terminal();
    isExpanded = true;
    if (parentNode != nullptr) {
        parentNode->increment_no_visit_idx();
    }
}

Move Node::get_move() const
{
    return move;
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

void Node::set_nn_results(float nn_value, const DynamicVector<float> &policyProbSmall)
{
    value = nn_value;
    for (size_t i = 0; i < childNodes.size(); ++i) {
        childNodes[i]->set_prob_value(policyProbSmall[i]);
    }
    hasNNResults = true;
}

void Node::apply_virtual_loss()
{
    ++virtualLossCounter;
    update_q_value();
}

void Node::set_prob_value(float value)
{
    probValue = value;
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
    ++numberExpandedNodes;
    if (numberExpandedNodes == numberChildNodes) {
        isFullyExpanded = true;
    }
}

float Node::get_value() const
{
    return value;
}

bool Node::is_expanded() const
{
    return isExpanded;
}

bool Node::are_child_nodes_sorted() const
{
    return areChildNodesSorted;
}

bool Node::is_calibrated() const
{
    return isCalibrated;
}

Node* Node::candidate_child_node() const
{
    return childNodes[0];
}

Node *Node::alternative_child_node() const
{
    return childNodes[1];
}

Key Node::hash_key() const
{
    return pos->hash_key();
}

size_t Node::get_number_child_nodes() const
{
    return numberChildNodes;
}

Node *Node::get_checkmate_node() const
{
    return checkmateNode;
}

float Node::get_visits() const
{
    return visits;
}

float Node::get_prob_value() const
{
    return probValue;
}

float Node::get_q_value() const
{
    return qValue;
}

void Node::update_q_value()
{
    assert(int(visits + virtualLossCounter) != 0);
    qValue = (actionValue - virtualLossCounter * searchSettings->virtualLoss) / (visits + virtualLossCounter);
}

float Node::get_u_parent_factor() const
{
    return uParentFactor;
}

float Node::get_u_divisor_summand() const
{
    return uDivisorSummand;
}

float Node::get_action_value() const
{
    return actionValue;
}

SearchSettings* Node::get_search_settings() const
{
    return searchSettings;
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
        if (parentNode->pos->gives_check(move)) {
            value = LOSS;
            isTerminal = true;
            parentNode->checkmateNode = this;
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
            parentNode->checkmateNode = this;
            return;
        }
    }
#endif
    if (pos->is_draw(pos->game_ply())) {
        // reached 50 moves rule
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

void Node::revert_virtual_loss()
{
    --virtualLossCounter;
    assert(virtualLossCounter >= 0);
    update_q_value();
}

void Node::revert_virtual_loss_and_update(float value)
{
    ++visits;
    actionValue += value;
    revert_virtual_loss();
}

void Node::init_board()
{
    StateInfo* newState = new StateInfo;
    pos = new Board(*parentNode->pos);
    pos->do_move(move, *newState);
}

void Node::update_u_divisor()
{
    uDivisorSummand = get_current_u_divisor(visits, searchSettings->uMin, searchSettings->uInit, searchSettings->uBase);
}

void Node::update_u_parent_factor()
{
    uParentFactor = get_current_cput(visits, searchSettings->cpuctBase, searchSettings->cpuctInit) * sqrt(visits + virtualLossCounter);
}

float Node::compute_current_u_value() const
{
    return parentNode->get_u_parent_factor() * (probValue / (visits + virtualLossCounter + parentNode->get_u_divisor_summand()));
}

float Node::compute_q_plus_u() const
{
    return qValue + compute_current_u_value();
}

void Node::create_child_nodes()
{
    for (const ExtMove move : MoveList<LEGAL>(*pos)) {
        childNodes.push_back(new Node(this, move, searchSettings));
    }
    numberChildNodes = childNodes.size();
}

void Node::lock()
{
    mtx.lock();
}

void Node::unlock()
{
    mtx.unlock();
}

void Node::mark_as_uncalibrated()
{
    isCalibrated = false;
}

void Node::apply_dirichlet_noise_to_prior_policy()
{
    DynamicVector<float> dirichletNoise = get_dirichlet_noise(numberChildNodes, searchSettings->dirichletAlpha);
    size_t idx = 0;
    float probEpsilon = (1 - searchSettings->dirichletEpsilon);
    for (Node* node : childNodes) {
        node->probValue = probEpsilon * node->probValue + searchSettings->dirichletEpsilon * dirichletNoise[idx++];
    }
}

void backup_value(Node* currentNode, float value)
{
    do {
        currentNode->lock();
        currentNode->revert_virtual_loss_and_update(value);
        currentNode->mark_as_uncalibrated();
        currentNode->unlock();
        value = -value;
        currentNode = currentNode->get_parent_node();
    } while(currentNode != nullptr);
}

void backup_collision(Node *currentNode)
{
    do {
        currentNode->lock();
        currentNode->revert_virtual_loss();
        currentNode->unlock();
        currentNode = currentNode->get_parent_node();
    } while(currentNode != nullptr);
}

vector<Move> retrieve_legal_moves(const vector<Node*>& childNodes)
{
    vector<Move> legalMoves;
    for (auto node: childNodes) {
        legalMoves.push_back(node->get_move());
    }
    return legalMoves;
}

void create_child_nodes(Node* parentNode, const Board* pos, vector<Node*> &childNodes, SearchSettings* searchSettings)
{
    for (const ExtMove move : MoveList<LEGAL>(*pos)) {
        childNodes.push_back(new Node(parentNode, move, searchSettings));
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

void enhance_moves(const SearchSettings* searchSettings, const Board* pos, const vector<Move>& legalMoves, DynamicVector<float>& policyProbSmall)
{
    float maxPolicyValue = max(policyProbSmall);
    bool checkUpdate = false;
    bool captureUpdate = false;

    if (searchSettings->enhanceChecks) {
        checkUpdate = enhance_move_type(min(searchSettings->threshCheck, maxPolicyValue*searchSettings->checkFactor),
                                        searchSettings->threshCheck, pos, legalMoves, isCheck, policyProbSmall);
    }
    if (searchSettings->enhanceCaptures) {
        captureUpdate = enhance_move_type( min(searchSettings->threshCapture, maxPolicyValue*searchSettings->captureFactor),
                                           searchSettings->threshCheck, pos, legalMoves, isCapture, policyProbSmall);
    }

    if (checkUpdate || captureUpdate) {
        policyProbSmall /= sum(policyProbSmall);
    }
}

Node* select_child_node(Node* node)
{
    node->lock();

    if (!node->are_child_nodes_sorted()) {
        node->sort_child_nodes_by_probabilities();
    }

    else if (node->get_number_child_nodes() != 1) {
        node->update_u_divisor();
        node->update_u_parent_factor();

        if (!node->is_calibrated() ||
                node->candidate_child_node()->compute_q_plus_u() < node->alternative_child_node()->compute_q_plus_u()) {
            node->calibrate_child_node_order();
        }
    }

    node->unlock();

    return node->candidate_child_node();
}

ostream& operator<<(ostream &os, Node *node)
{
    os << "move " << UCI::move(node->get_move(), false)
       << "\tn " << node->get_visits()
       << "\tp " << node->get_prob_value()
       << "\tQ " << node->get_q_value()
       << "\tQ+U " << node->compute_q_plus_u();
    return os;
}

void delete_sibling_subtrees(Node* node, unordered_map<Key, Node*>* hashTable)
{
    if (node->get_parent_node() != nullptr) {
        cout << "info string delete unused subtrees" << endl;
        for (Node* childNode: node->get_parent_node()->get_child_nodes()) {
            if (childNode != node) {
                delete_subtree_and_hash_entries(childNode, hashTable);
            }
        }
    }
}

void delete_subtree_and_hash_entries(Node* node, unordered_map<Key, Node*>* hashTable)
{
    // if the current node hasn't been expanded or is a terminal node then childNodes is empty and the recursion ends
    for (Node* childNode: node->get_child_nodes()) {
        delete_subtree_and_hash_entries(childNode, hashTable);
    }

    if (node->is_expanded()) {
        // the board position is only filled if the node has been extended
        auto it = hashTable->find(node->hash_key());
        if(it != hashTable->end()) {
            hashTable->erase(node->hash_key());
        }
    }
    delete node;
}

void get_mcts_policy(const Node* node, const DynamicVector<float>& childNumberVisits, DynamicVector<float>& mctsPolicy)
{
    const float qThresh = get_current_q_thresh(node->get_search_settings(), node->get_visits());
    const float qValueWeight = node->get_search_settings()->qValueWeight;

    if (qValueWeight > 0) {
        DynamicVector<float> qValuePruned = retrieve_q_values(node);
        qValuePruned = (qValuePruned + 1) * 0.5f;
        float visitThresh = qThresh * max(childNumberVisits);
        for (size_t idx = 0; idx < node->get_number_child_nodes(); ++idx) {
            if (childNumberVisits[idx] < visitThresh) {
                qValuePruned[idx] = 0;
            }
        }
        mctsPolicy = (1.0f - qValueWeight) * (childNumberVisits / node->get_visits()) + qValueWeight * qValuePruned;
        mctsPolicy /= sum(mctsPolicy);
    } else {
        mctsPolicy = childNumberVisits / node->get_visits();
    }
}

DynamicVector<float> retrieve_dynamic_vector(const vector<Node *>& childNodes, vFunctionValue func)
{
    DynamicVector<float> values(childNodes.size());
    for (size_t i = 0; i < childNodes.size(); ++i) {
        values[i] = func(childNodes[i]);
    }
    return values;
}

float get_visits(Node* node)
{
    return node->get_visits();
}

float get_q_value(Node* node)
{
    return node->get_q_value();
}

DynamicVector<float> retrieve_visits(const Node* node)
{
    return retrieve_dynamic_vector(node->get_child_nodes(), get_visits);
}

DynamicVector<float> retrieve_q_values(const Node* node)
{
    return retrieve_dynamic_vector(node->get_child_nodes(), get_q_value);
}

float get_current_q_thresh(const SearchSettings* searchSettings, int numberVisits)
{
    return searchSettings->qThreshMax - exp(-numberVisits / searchSettings->qThreshBase) * (searchSettings->qThreshMax - searchSettings->qThreshInit);
}

double updated_value(const Node* node, DynamicVector<float>& mctsPolicy)
{
    return node->get_child_nodes()[argmax(mctsPolicy)]->get_q_value();
}

double get_current_cput(float numberVisits, float cpuctBase, float cpuctInit)
{
    return log((numberVisits + cpuctBase + 1) / cpuctBase) + cpuctInit;
}

float get_current_u_divisor(float numberVisits, float uMin, float uInit, float uBase)
{
    return uMin - exp(-numberVisits / uBase) * (uMin - uInit);
}

void print_node_statistics(Node* node)
{
    size_t candidateIdx = 0;
    cout << "info string position " << node->get_pos()->fen() << endl;
    for (auto node : node->get_child_nodes()) {
        cout << candidateIdx++ << "." << node << endl;
    }
    cout << " initial value: " << node->get_value() << endl;
}

bool is_ordering_correct(vector<Node*> &childNodes)
{
    for (size_t i = 0; i < childNodes.size()-1; ++i) {
        if (childNodes[i]->compute_q_plus_u() < childNodes[i+1]->compute_q_plus_u()) {
            return false;
        }
    }
    return true;
}

bool prob_value_comparision(const Node* n1, const Node* n2)
{
    return n1->get_prob_value() > n2->get_prob_value();
}

bool q_plus_u_comparision(const Node* n1, const Node* n2)
{
    return n1->compute_q_plus_u() > n2->compute_q_plus_u();
}

void get_principal_variation(const Node* rootNode, const SearchSettings* searchSettings, vector<Move>& pv)
{
    pv.clear();
    const Node* curNode = rootNode;
    size_t childIdx;
    do {
        DynamicVector<float> mctsPolicy(curNode->get_number_child_nodes());
        get_mcts_policy(curNode, retrieve_visits(curNode), mctsPolicy);
        childIdx = argmax(mctsPolicy);
        pv.push_back(curNode->get_child_nodes()[childIdx]->get_move());
        curNode = curNode->get_child_nodes()[childIdx];
    } while (curNode->is_expanded() && !curNode->is_terminal());
}

int estimate_visits_to_switch(const float secondScore, const float cpuct, Node* n)
{
    // TODO
    //    s = (a-x*v)/(n) + c * p * ((sqrt(n))/(u+x)); solve for x
    // x = -(a * u + c * n^(1.5) * p - n * s * u) / (a - n * s)
    return -int((n->get_action_value() * n->get_parent_node()->get_u_divisor_summand() +
                 cpuct * float(pow(n->get_visits(), 1.5)) * n->get_prob_value() - n->get_visits() *
                 secondScore * n->get_parent_node()->get_u_divisor_summand()) / (n->get_action_value() - n->get_visits() * secondScore));
    //    return -(n->get_ * n-)
}
