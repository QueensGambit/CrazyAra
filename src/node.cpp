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

Node::Node(Node *parentNode, Move move,  SearchSettings* searchSettings):
    parentNode(parentNode),
    move(move),
    value(0),
    probValue(0),
    qValue(-1.0),
    actionValue(0),
    visits(0),
    virtualLossCounter(0),
    isTerminal(false), // will be later recomputed
    isExpanded(false),
    hasNNResults(false),
    isCalibrated(false),
    areChildNodesSorted(false),
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
    // deep copy of position and stateInfo -> probably change this
    pos = new Board(*b.pos);
    pos->setStateInfo(new StateInfo(*(b.pos->getStateInfo())));
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
}

void Node::sort_child_nodes_by_probabilities()
{
    sort(childNodes.begin(), childNodes.end(), prob_value_comparision);

    areChildNodesSorted = true;
    isCalibrated = true;
    nodeIdxUpdate = 2; //2;
}

void Node::sort_child_nodes_by_q_plus_u()
{
    sort(childNodes.begin(), childNodes.end(), q_plus_u_comparision);
    areChildNodesSorted = true;
    isCalibrated = true;
}

void Node::calibrate_child_node_order()
{
//    nodeIdxUpdate = numberChildNodes; // !
//    cout << "nodeIdxUpdate" << nodeIdxUpdate << endl;
//    if (nodeIdxUpdate != 0) {
        nodeIdxUpdate = min(nodeIdxUpdate+1, numberChildNodes);
//        std::partial_sort(childNodes.begin(), childNodes.begin()+int(nodeIdxUpdate), childNodes.end(), //childNodes.begin()+nodeIdxUpdate,
//            [=](const Node* n1, const Node* n2) {
//            return n1->get_q_plus_u() > n2->get_q_plus_u();
//        });
//        std::sort(childNodes.begin(), childNodes.begin()+int(nodeIdxUpdate), q_plus_u_comparision);
//    }
    partial_sort(childNodes.begin(), childNodes.begin()+1, childNodes.end(), q_plus_u_comparision);

    // sorting
    areChildNodesSorted = true;
    isCalibrated = true;

    // DEBUG
//    if (!is_ordering_correct(childNodes)) {
//        cout << "nodeIdxUpdate: " << nodeIdxUpdate << endl;
//        print_node_statistics(this);
        assert(is_ordering_correct(childNodes));
//    }
//    nodeIdxUpdate = 1;
//    nodeIdxUpdate = 0;
}

void Node::expand()
{
//    mtx.lock();
//    create_child_nodes(this, pos, childNodes, searchSettings);
    create_child_nodes();
    numberChildNodes = childNodes.size();
    check_for_terminal();
    isExpanded = true;
//    mtx.unlock();
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
//    mtx.lock();
    value = nn_value;
    for (size_t i = 0; i < childNodes.size(); ++i) {
        childNodes[i]->set_prob_value(policyProbSmall[i]);
    }
    hasNNResults = true;
//    mtx.unlock();
}

void Node::apply_virtual_loss()
{
//    mtx.lock();
    ++virtualLossCounter;
    update_q_value();
//    mtx.unlock();
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

double Node::get_q_value() const
{
    return qValue;
}

void Node::update_q_value()
{
    assert(int(visits + virtualLossCounter) != 0);
    qValue = (actionValue - virtualLossCounter * searchSettings->virtualLoss) / (visits + virtualLossCounter);
}

double Node::get_u_parent_factor() const
{
    return uParentFactor;
}

float Node::get_u_divisor_summand() const
{
    return uDivisorSummand;
}

void Node::validate_candidate_node()
{
    if (numberChildNodes != 1 && childNodes[0]->get_q_plus_u() < childNodes[1]->get_q_plus_u()) {
        // an update is required
        if (numberChildNodes == 2 || childNodes[0]->get_q_plus_u() > childNodes[2]->get_q_plus_u()) {
            swap_candidate_node_with_alternative();
        }
        else {
//             cout << "readjust" << endl;
            readjust_candidate_node_position();
        }
//        else if (numberChildNodes != 2) {
//        }
    }
    else {
//        cout << "no adjust" << endl;
    }
}

void Node::swap_candidate_node_with_alternative()
{
    // only swap first two
    Node* temp = childNodes[0];
    childNodes[0] = childNodes[1];
    childNodes[1] = temp;
//    if (nodeIdxUpdate == 0) {
//        nodeIdxUpdate = 2;
//    }
}

void Node::readjust_candidate_node_position()
{
    // put former last element at according location
//    Node* element = childNodes.back();
//    childNodes.pop_back();

    size_t idx = 3;
    for(auto it = childNodes.begin()+3; it != childNodes.end(); ++it) {
        Node* curElement = *it;
        if (curElement->get_q_plus_u() < childNodes[0]->get_q_plus_u()) {
//            childNodes.insert(it.base(), element);
            break;
        }
        ++idx;
    }

//    cout << numberChildNodes << endl;
//    cout << nodeIdxUpdate << endl;
//    std::rotate(childNodes.end()-idx,childNodes.end()-1, childNodes.end());
    std::rotate(childNodes.begin(), childNodes.begin()+1, childNodes.begin()+idx);

    nodeIdxUpdate = max(idx, nodeIdxUpdate);
}

void Node::check_for_terminal()
{
    if (childNodes.size() == 0) {
        isTerminal = true;
        // test if we have a check-mate
        if (parentNode->pos->gives_check(move)) {
            value = -1;
//            parentNode->mtx.lock();
            parentNode->checkmateNode = this;
//            parentNode->mtx.unlock();
        } else {
            // we reached a stalmate
            value = 0;
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
//    mtx.lock();
    ++visits;
    actionValue += double(value);
    revert_virtual_loss();
//    qValue = actionValue / double(visits);
//    mtx.unlock();
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

double Node::get_current_u_value() const
{
    return parentNode->get_u_parent_factor() * (probValue / (visits + virtualLossCounter + parentNode->get_u_divisor_summand()));
}

double Node::get_q_plus_u() const
{
    return qValue + get_current_u_value();
}

void Node::create_child_nodes()
{
    for (const ExtMove move : MoveList<LEGAL>(*pos)) {
//        cout << "move " << UCI::move(move, false) << endl;
        childNodes.push_back(new Node(this, move, searchSettings));
//        cout << childNodes.back() << endl;
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
        currentNode->revert_virtual_loss_and_update(value);
        value = -value;
        currentNode->mark_as_uncalibrated();
        currentNode = currentNode->get_parent_node();
    } while(currentNode != nullptr);
}

void backup_collision(Node *currentNode)
{
    do {
        currentNode->revert_virtual_loss();
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
//        cout << "move " << UCI::move(move, false) << endl;
        childNodes.push_back(new Node(parentNode, move, searchSettings));
//        cout << childNodes.back() << endl;
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
    node->update_u_divisor();
    node->update_u_parent_factor();
    if (node->is_calibrated()) {
        node->validate_candidate_node();
    }
    else {
        if (node->are_child_nodes_sorted()) {
            node->calibrate_child_node_order();
        }
        else {
            node->sort_child_nodes_by_probabilities();
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
       << "\tQ+U " << node->get_q_plus_u();
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

void get_mcts_policy(const Node *node, const float qValueWeight, const float qThresh, DynamicVector<float>& mctsPolicy)
{
    DynamicVector<float> childNumberVisits = retrieve_visits(node);
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
//    for (auto childNodeIt = childNodes.rbegin(); childNodeIt != childNodes.rend(); ++childNodeIt) {
//        cout << candidateIdx++ << "." << *childNodeIt << endl;
//    }
    cout << "info string position " << node->get_pos()->fen() << endl;
    for (auto node : node->get_child_nodes()) {
        cout << candidateIdx++ << "." << node << endl;
    }
    cout << " initial value: " << node->get_value() << endl;
}

bool is_ordering_correct(vector<Node*> &childNodes)
{
    for (size_t i = 0; i < childNodes.size(); ++i) {
        if (childNodes[0]->get_q_plus_u() < childNodes[i]->get_q_plus_u()) {
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
    return n1->get_q_plus_u() > n2->get_q_plus_u();
}

void get_principal_variation(const Node* rootNode, const SearchSettings* searchSettings, vector<Move>& pv)
{
    pv.clear();
    const Node* curNode = rootNode;
    size_t childIdx;
    do {
        DynamicVector<float> mctsPolicy(curNode->get_number_child_nodes());
        get_mcts_policy(curNode, searchSettings->qValueWeight,
                        get_current_q_thresh(searchSettings, int(rootNode->get_visits())),
                        mctsPolicy);
        childIdx = argmax(mctsPolicy);
        pv.push_back(curNode->get_child_nodes()[childIdx]->get_move());
        curNode = curNode->get_child_nodes()[childIdx];
    } while (curNode->is_expanded() && !curNode->is_terminal());
}
