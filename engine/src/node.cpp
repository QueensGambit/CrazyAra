/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

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
#include <limits.h>
#include "util/blazeutil.h" // get_dirichlet_noise()
#include "constants.h"
#include "../util/communication.h"
#include "evalinfo.h"


bool Node::is_sorted() const
{
    return sorted;
}

bool Node::is_cell() const
{
    return isCell;
}

Action Node::get_predecessor_action() const
{
    return parentNode->get_action(childIdxForParent);
}

bool Node::is_unstable() const
{
    return isUnstable;
}

void Node::mark_as_unstable()
{
    isUnstable = true;
}

Node::Node(StateObj* state, bool inCheck, Node* parentNode, size_t childIdxForParent, const SearchSettings* searchSettings):
    legalActions(state->legal_actions()),
    parentNode(parentNode),
    key(state->hash_key()),
    value(0),
    d(nullptr),
    childIdxForParent(childIdxForParent),
    pliesFromNull(state->steps_from_null()),
    isTerminal(false),
    isTablebase(false),
    hasNNResults(false),
    sorted(false),
    isCell(false),
    isUnstable(false)
{
    // specify the number of direct child nodes of this node
    const int numberChildNodes = legalActions.size();

    check_for_terminal(state, inCheck);
#ifdef MODE_CHESS
    if (searchSettings->useTablebase && !isTerminal) {
        check_for_tablebase_wdl(state);
    }
#endif
    policyProbSmall.resize(numberChildNodes);
}

Node::Node(const Node &b)
{
    set_value(b.get_value());
    key = b.key;
    pliesFromNull = b.plies_from_null();
    const int numberChildNodes = b.legalActions.size();
    policyProbSmall.resize(numberChildNodes);
    policyProbSmall = b.policyProbSmall;
    legalActions = b.legalActions;
    isTerminal = b.isTerminal;
    //    parentNode = // is not copied
    //    childIdxForParent = // is not copied
    isTerminal = b.isTerminal;
    isTablebase = b.isTablebase;
    hasNNResults = b.hasNNResults;
    sorted = b.sorted;
    if (isTerminal) {
        d = make_unique<NodeData>(numberChildNodes);
        d->nodeType = b.d->nodeType;
        return;
    }
    if (sorted) {
        d = make_unique<NodeData>(numberChildNodes);
    }
    // TODO: Allow copying checkmateIndex
}

bool Node::solved_win(const Node* childNode) const
{
    if (childNode->d->nodeType == SOLVED_LOSS) {
        d->checkmateIdx = childNode->get_child_idx_for_parent();
        return true;
    }
    return false;
}

bool Node::solved_draw(const Node* childNode) const
{
    if (d->numberUnsolvedChildNodes == 0 &&
            (childNode->d->nodeType == SOLVED_DRAW || childNode->d->nodeType == SOLVED_WIN)) {
        return at_least_one_drawn_child();
    }
    return false;
}

bool Node::at_least_one_drawn_child() const
{
    bool atLeastOneDrawnChild = false;
    for (Node* childNode : d->childNodes) {
        if (!childNode->is_playout_node() || (childNode->d->nodeType != SOLVED_DRAW && childNode->d->nodeType != SOLVED_WIN)) {
            return false;
        }
        if (childNode->d->nodeType == SOLVED_DRAW) {
            atLeastOneDrawnChild = true;
        }
    }
    if (atLeastOneDrawnChild) {
        return true;
    }
    return false;
}

bool Node::only_won_child_nodes() const
{
    for (Node* childNode : d->childNodes) {
        if (childNode->d->nodeType != SOLVED_WIN) {
            return false;
        }
    }
    return true;
}

bool Node::solved_loss(const Node* childNode) const
{
    if (d->numberUnsolvedChildNodes == 0 && childNode->d->nodeType == SOLVED_WIN) {
        return only_won_child_nodes();
    }
    return false;
}

void Node::mark_as_loss()
{
    set_value(LOSS);
    d->nodeType = SOLVED_LOSS;
}

void Node::mark_as_draw()
{
    set_value(DRAW);
    d->nodeType = SOLVED_DRAW;
}

void Node::mark_as_win()
{
    set_value(WIN);
    d->nodeType = SOLVED_WIN;
}

void Node::define_end_ply_for_solved_terminal(const Node* childNode)
{
    if (d->nodeType == SOLVED_LOSS) {
        // choose the longest pv line
        for (const Node* curChildNode : d->childNodes) {
            if (curChildNode->d->endInPly+1 > d->endInPly) {
                d->endInPly = curChildNode->d->endInPly+1;
            }
        }
        return;
    }
    if (d->nodeType == SOLVED_DRAW) {
        // choose the shortest pv line for draws
        for (const Node* curChildNode : d->childNodes) {
            if (curChildNode->d->nodeType == SOLVED_DRAW && curChildNode->d->endInPly+1 < d->endInPly) {
                d->endInPly = curChildNode->d->endInPly+1;
            }
        }
        return;
    }
    // get the endPly for WINS
    d->endInPly = childNode->d->endInPly + 1;
}

template <int targetValue>
void Node::update_solved_terminal(const Node* childNode)
{
    define_end_ply_for_solved_terminal(childNode);
    set_value(targetValue);
    if (parentNode != nullptr) {
        parentNode->lock();
        parentNode->d->numberUnsolvedChildNodes--;
        parentNode->d->qValues[childIdxForParent] = -targetValue;
        if (targetValue == LOSS) {
            parentNode->d->checkmateIdx = childIdxForParent;
        }
        else if (targetValue == WIN && !is_root_node() && parentNode->is_root_node()) {
            parentNode->disable_action(childIdxForParent);
        }
        parentNode->unlock();
    }
}

void Node::mcts_policy_based_on_wins(DynamicVector<float> &mctsPolicy) const
{
    mctsPolicy = 0;
    size_t childIdx = 0;
    for (auto childNode: get_child_nodes()) {
        if (childNode != nullptr && childNode->d != nullptr && childNode->d->nodeType == SOLVED_LOSS) {
            mctsPolicy[childIdx] = 1.0f;
        }
        ++childIdx;
    }
    mctsPolicy /= sum(mctsPolicy);
}

void Node::prune_losses_in_mcts_policy(DynamicVector<float> &mctsPolicy) const
{
    // check if PV line leads to a loss
    if (d->numberUnsolvedChildNodes != get_number_child_nodes() && d->nodeType != SOLVED_LOSS) {
        // set all entries which lead to a WIN of the opponent to zero
        for (size_t childIdx = 0; childIdx < d->noVisitIdx; ++childIdx) {
            const Node* childNode = d->childNodes[childIdx];
            if (childNode != nullptr && childNode->is_playout_node() && childNode->d->nodeType == SOLVED_WIN) {
                mctsPolicy[childIdx] = 0;
            }
        }
    }
}

void Node::mcts_policy_based_on_q_n(DynamicVector<float>& mctsPolicy, float qValueWeight) const
{
    DynamicVector<float> qValuePruned = d->qValues;
    qValuePruned = (qValuePruned + 1) * 0.5f;
    const DynamicVector<float> normalizedVisits = d->childNumberVisits / get_visits();
    const float quantile = get_quantile(normalizedVisits, 0.25f);
    for (size_t idx = 0; idx < get_number_child_nodes(); ++idx) {
        if (d->childNumberVisits[idx] < quantile) {
            qValuePruned[idx] = 0;
        }
    }
    mctsPolicy = (1.0f - qValueWeight) * normalizedVisits + qValueWeight * qValuePruned;
}

void Node::solve_for_terminal(const Node* childNode)
{
    if (d->nodeType != UNSOLVED) {
        // already solved
        return;
    }
    if (!childNode->is_playout_node()) {
        return;
    }
    if (solved_win(childNode)) {
        d->nodeType = SOLVED_WIN;
        update_solved_terminal<WIN>(childNode);
        return;
    }
    if (solved_loss(childNode)) {
        d->nodeType = SOLVED_LOSS;
        update_solved_terminal<LOSS>(childNode);
        return;
    }
    if (solved_draw(childNode)) {
        d->nodeType = SOLVED_DRAW;
        update_solved_terminal<DRAW>(childNode);
    }
}

void Node::mark_nodes_as_fully_expanded()
{
    info_string("mark as fully expanded");
    d->noVisitIdx = get_number_child_nodes();
}

bool Node::is_root_node() const
{
    return parentNode->parentNode == nullptr;
}

Node::~Node()
{
}

void Node::sort_moves_by_probabilities()
{
    auto p = sort_permutation(policyProbSmall, std::greater<float>());
    apply_permutation_in_place(policyProbSmall, p);
    apply_permutation_in_place(legalActions, p);
    sorted = true;
}

Action Node::get_action(size_t childIdx) const
{
    return legalActions[childIdx];
}

Node *Node::get_child_node(size_t childIdx) const
{
    return d->childNodes[childIdx];
}

Action Node::get_best_action() const
{
    return get_action(get_best_action_index(this, false));
}

vector<Action> Node::get_ponder_moves() const
{
    vector<Action> ponderMoves;
    const size_t visitThresh = 0.01 * get_visits();

    for (const Node* childNode : get_child_nodes()) {
        if (childNode == nullptr) {
            break;
        }
        if (childNode->is_playout_node() && childNode->get_visits() > visitThresh) {
            if (!childNode->is_terminal()) {

                if (ponderMoves.size() == 0) {
                    ponderMoves.emplace_back(childNode->get_best_action());
                }
                else if (find(ponderMoves.begin(), ponderMoves.end(), childNode->get_best_action()) == ponderMoves.end()) {
                    ponderMoves.emplace_back(childNode->get_best_action());
                }
            }
        }
    }
    return ponderMoves;
}

vector<Node*> Node::get_child_nodes() const
{
    return d->childNodes;
}

bool Node::is_terminal() const
{
    return isTerminal;
}

bool Node::has_nn_results() const
{
    return hasNNResults;
}

void Node::apply_virtual_loss_to_child(size_t childIdx, float virtualLoss)
{
    // update the stats of the parent node
    // make it look like if one has lost X games from this node forward where X is the virtual loss value
    // temporarily reduce the attraction of this node by applying a virtual loss /
    // the effect of virtual loss will be undone if the playout is over
    d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] - virtualLoss) / (d->childNumberVisits[childIdx] + virtualLoss);
    // virtual increase the number of visits
    d->childNumberVisits[childIdx] += size_t(virtualLoss);
}

Node *Node::get_parent_node() const
{
    return parentNode;
}

void Node::increment_visits(size_t numberVisits)
{
    parentNode->lock();
    parentNode->d->childNumberVisits[childIdxForParent] += numberVisits;
    parentNode->unlock();
}

void Node::subtract_visits(size_t numberVisits)
{
    parentNode->lock();
    parentNode->d->childNumberVisits[childIdxForParent] -= numberVisits;
    parentNode->unlock();
}

float Node::get_q_value(size_t idx)
{
    return d->qValues[idx];
}

void Node::set_q_value(size_t idx, float value)
{
    d->qValues[idx] = value;
}

size_t Node::get_best_q_idx() const
{
    return argmax(d->qValues);
}

vector<size_t> Node::get_q_idx_over_thresh(float qThresh)
{
    vector<size_t> indices;
    for (size_t idx = 0; idx < size(d->qValues); ++idx) {
        if (d->qValues[idx] > qThresh) {
            indices.emplace_back(idx);        }
    }
    return indices;
}

void Node::reserve_full_memory()
{
    const size_t numberChildNodes = get_number_child_nodes();
    d->childNumberVisits.reserve(numberChildNodes);
    d->qValues.reserve(numberChildNodes);
    d->childNodes.reserve(numberChildNodes);
}

void Node::increment_no_visit_idx(Cells* cells)
{
    if (d->noVisitIdx < get_number_child_nodes()) {
        ++d->noVisitIdx;
        Node* node = this;
        Node* parentNode = node->get_parent_node();
        if (!this->is_root_node() && parentNode->is_playout_node() && parentNode->is_fully_expanded() && node->plies_from_null() != 0 && node->plies_from_null() % 2 == 0) {
            size_t bestQIdx = parentNode->get_best_q_idx();
            if (-parentNode->get_q_value(bestQIdx) > 0.5 && parentNode->get_no_visit_idx() < 2 && !node->isTerminal) {
                cells->mtx.lock();
                cells->trajectories.emplace_back(get_trajectory(parentNode));
                cells->mtx.unlock();
            }
        }
        if (d->noVisitIdx == PRESERVED_ITEMS) {
            reserve_full_memory();
        }
        d->add_empty_node();
    }
}

void Node::fully_expand_node()
{
    if (d->nodeType == UNSOLVED && !is_fully_expanded()) {
        reserve_full_memory();
        for (size_t idx = d->noVisitIdx; idx < get_number_child_nodes(); ++idx) {
            d->add_empty_node();
        }
        d->noVisitIdx = get_number_child_nodes();
        // keep this exact order
        sorted = true;
    }
}

float Node::get_value() const
{
    return value;
}

Key Node::hash_key() const
{
    return key;
}

size_t Node::get_number_child_nodes() const
{
    return legalActions.size();
}

void Node::prepare_node_for_visits()
{
    sort_moves_by_probabilities();
    init_node_data();
}

uint32_t Node::get_visits() const
{
    return parentNode->d->childNumberVisits[childIdxForParent];
}

void Node::backup_value(size_t childIdx, float value, float virtualLoss)
{
    Node* currentNode = this;
    do {
        currentNode->revert_virtual_loss_and_update(childIdx, value, virtualLoss);
        childIdx = currentNode->childIdxForParent;
        value = -value;
        currentNode = currentNode->parentNode;
    } while(currentNode->parentNode != nullptr);
    // revert virtual loss for root
    if (virtualLoss != 1) {
        currentNode->get_child_node(childIdx)->subtract_visits(virtualLoss-1);
    }
}

void Node::revert_virtual_loss_and_update(size_t childIdx, float value, float virtualLoss)
{
    lock();
    if (d->childNumberVisits[childIdx] == virtualLoss) {
        // set new Q-value based on return
        // (the initialization of the Q-value was by Q_INIT which we don't want to recover.)
        d->qValues[childIdx] = value;
    }
    else {
        // revert virtual loss and update the Q-value
        assert(d->childNumberVisits[childIdx] != 0);
        d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] + virtualLoss + value) / d->childNumberVisits[childIdx];
    }

    if (virtualLoss != 1) {
        d->childNumberVisits[childIdx] -= size_t(virtualLoss) - 1;
    }
    if (is_terminal_value(value)) {
        ++d->terminalVisits;
        solve_for_terminal(d->childNodes[childIdx]);
    }
    unlock();
}

void Node::backup_collision(size_t childIdx, float virtualLoss)
{
    Node* currentNode = this;
    do {
        currentNode->revert_virtual_loss(childIdx, virtualLoss);
        childIdx = currentNode->childIdxForParent;
        currentNode = currentNode->parentNode;
    } while(currentNode->parentNode != nullptr);
    // revert virtual loss for root
    currentNode->get_child_node(childIdx)->subtract_visits(virtualLoss);
}

void Node::revert_virtual_loss(size_t childIdx, float virtualLoss)
{
    lock();
    d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] + virtualLoss) / (d->childNumberVisits[childIdx] - virtualLoss);
    d->childNumberVisits[childIdx] -= virtualLoss;
    unlock();
}

bool Node::is_playout_node() const
{
    return d != nullptr;
}

bool Node::is_blank_root_node() const
{
    return get_visits() == 0;
}

bool Node::is_solved() const
{
    return d->nodeType != UNSOLVED;
}

bool Node::has_forced_win() const
{
    return get_checkmate_idx() != NO_CHECKMATE;
}

void Node::set_parent_node(Node* value)
{
    parentNode = value;
}

size_t Node::get_no_visit_idx() const
{
    return d->noVisitIdx;
}

bool Node::is_fully_expanded() const
{
    return get_number_child_nodes() == d->noVisitIdx;
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
    d->childNodes[childIdx] = newNode;
}

void Node::add_transposition_child_node(Node* newNode, size_t childIdx)
{
    newNode->parentNode = this;
    newNode->childIdxForParent = childIdx;
    d->childNodes[childIdx] = newNode;
}

float Node::max_policy_prob()
{
    return max(policyProbSmall);
}

size_t Node::max_q_child()
{
    return argmax(d->qValues);
}

size_t Node::max_visits_child()
{
    return argmax(d->childNumberVisits);
}

float Node::updated_value_eval() const
{
    if (!is_sorted()) {
        return get_value();
    }
    if (d == nullptr || get_visits() == 1) {
        return get_value();
    }
    switch(d->nodeType) {
    case SOLVED_WIN:
        return WIN;
    case SOLVED_DRAW:
        return DRAW;
    case SOLVED_LOSS:
        return LOSS;
    default: ;  // UNSOLVED
    }
    return d->qValues[argmax(d->childNumberVisits)];
}

std::vector<Action> Node::get_legal_action() const
{
    return legalActions;
}

int Node::get_checkmate_idx() const
{
    return d->checkmateIdx;
}

DynamicVector<uint32_t> Node::get_child_number_visits() const
{
    return d->childNumberVisits;
}

void Node::enable_has_nn_results()
{
    hasNNResults = true;
}

uint16_t Node::plies_from_null() const
{
    return pliesFromNull;
}

bool Node::is_tablebase() const
{
    return isTablebase;
}

uint8_t Node::get_node_type() const
{
    return d->nodeType;
}

uint16_t Node::get_end_in_ply() const
{
    return d->endInPly;
}

uint32_t Node::get_terminal_visits() const
{
    return d->terminalVisits;
}

void Node::init_node_data(size_t numberNodes)
{
    d = make_unique<NodeData>(numberNodes);
}

void Node::init_node_data()
{
    init_node_data(get_number_child_nodes());
}

void Node::mark_as_terminal()
{
    isTerminal = true;
    init_node_data();
}

void Node::check_for_terminal(StateObj* pos, bool inCheck)
{
    TerminalType terminalType = pos->is_terminal(get_number_child_nodes(), inCheck, value);

    if (terminalType != TERMINAL_NONE) {
        mark_as_terminal();
        switch(terminalType) {
        case TERMINAL_WIN:
            mark_as_win();
            break;
        case TERMINAL_DRAW:
            mark_as_draw();
            break;
        case TERMINAL_LOSS:
            mark_as_loss();
            break;
        case TERMINAL_CUSTOM:
        case TERMINAL_NONE:
            ;  // pass
        }
    }
}

void Node::check_for_tablebase_wdl(StateObj* state)
{
    Tablebase::ProbeState result;
    Tablebase::WDLScore wdlScore = state->check_for_tablebase_wdl(result);

    if (result != Tablebase::FAIL) {
        // TODO: Change return values
        isTablebase = true;
        switch(wdlScore) {
        case Tablebase::WDLLoss:
            set_value(-0.99); //LOSS);
            break;
        case Tablebase::WDLWin:
            set_value(0.99); //WIN);
            break;
        default:
            set_value(0.00001); //DRAW);
        }
    }
    // default: isTablebase = false;
}

void Node::make_to_root()
{
    parentNode->parentNode = nullptr;
}

void Node::lock()
{
    mtx.lock();
}

void Node::unlock()
{
    mtx.unlock();
}

void Node::apply_dirichlet_noise_to_prior_policy(const SearchSettings* searchSettings)
{
    DynamicVector<float> dirichlet_noise = get_dirichlet_noise(get_number_child_nodes(), searchSettings->dirichletAlpha);
    policyProbSmall = (1 - searchSettings->dirichletEpsilon ) * policyProbSmall + searchSettings->dirichletEpsilon * dirichlet_noise;
}

void Node::apply_temperature_to_prior_policy(float temperature)
{
    apply_temperature(policyProbSmall, temperature);
}

void Node::set_probabilities_for_moves(const float *data, unordered_map<Action, size_t, std::hash<int>>& moveLookup)
{
    // allocate sufficient memory -> is assumed that it has already been done
    assert(legalActions.size() == policyProbSmall.size());
    for (size_t mvIdx = 0; mvIdx < legalActions.size(); ++mvIdx) {
        // retrieve vector index from look-up table
        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        policyProbSmall[mvIdx] = data[moveLookup[legalActions[mvIdx]]];
    }
}

void Node::apply_softmax_to_policy()
{
    policyProbSmall = softmax(policyProbSmall);
}

//void Node::mark_enhanced_moves(const Board* pos, const SearchSettings* searchSettings)
//{
//    //    const float numberChildNodes = get_number_child_nodes();
//    //    if (searchSettings->enhanceChecks || searchSettings->enhanceCaptures) {
//    //        isCheck.resize(numberChildNodes);
//    //        isCheck = false;
//    //        isCapture.resize(numberChildNodes);
//    //        isCapture = false;

//    //        for (size_t idx = 0; idx < numberChildNodes; ++idx) {
//    //            if (pos->capture(legalMoves[idx])) {
//    //                isCapture[idx] = true;
//    //            }
//    //            if (pos->gives_check(legalMoves[idx])) {
//    //                isCheck[idx] = true;
//    //            }
//    //        }
//    //    }
//}

void Node::disable_action(size_t childIdxForParent)
{
    policyProbSmall[childIdxForParent] = 0;
    d->qValues[childIdxForParent] = -INT_MAX;
}

void Node::enhance_moves(const SearchSettings* searchSettings)
{
    //    if (!searchSettings->enhanceChecks && !searchSettings->enhanceCaptures) {
    //        return;
    //    }

    //    bool checkUpdate = false;
    //    bool captureUpdate = false;

    //    if (searchSettings->enhanceChecks) {
    //        checkUpdate = enhance_move_type(min(searchSettings->threshCheck, max(policyProbSmall)*searchSettings->checkFactor),
    //                                        searchSettings->threshCheck, legalMoves, isCheck, policyProbSmall);
    //    }
    //    if (searchSettings->enhanceCaptures) {
    //        captureUpdate = enhance_move_type(min(searchSettings->threshCapture, max(policyProbSmall)*searchSettings->captureFactor),
    //                                          searchSettings->threshCheck, legalMoves, isCapture, policyProbSmall);
    //    }

    //    if (checkUpdate || captureUpdate) {
    //        policyProbSmall /= sum(policyProbSmall);
    //    }
}

DynamicVector<float> Node::get_current_u_values(const SearchSettings* searchSettings)
{
    return get_current_cput(get_visits(), searchSettings) * blaze::subvector(policyProbSmall, 0, d->noVisitIdx) * (sqrt(get_visits()) / (d->childNumberVisits + 1.0));
}

Node *Node::get_child_node(size_t childIdx)
{
    return d->childNodes[childIdx];
}

void Node::get_mcts_policy(DynamicVector<float>& mctsPolicy, size_t& bestMoveIdx, float qValueWeight) const
{
    // fill only the winning moves in case of a known win
    if (d->nodeType == SOLVED_WIN) {
        mcts_policy_based_on_wins(mctsPolicy);
        return;
    }
    if (qValueWeight > 0) {
        size_t secondArg;
        float firstMax;
        float secondMax;
        mctsPolicy = d->childNumberVisits;
        first_and_second_max(mctsPolicy, d->noVisitIdx, firstMax, secondMax, bestMoveIdx, secondArg);
        if (d->qValues[secondArg]-Q_VALUE_DIFF > d->qValues[bestMoveIdx]) {
            mctsPolicy[bestMoveIdx] = secondMax;
            mctsPolicy[secondArg] = firstMax;
            bestMoveIdx = secondArg;
        }
// TODO: check if this is useful
//        else {
//            size_t qIdx = get_best_q_idx();
//            if (bestMoveIdx != qIdx) {
//                const float qDiff = 1.0f - (d->childNumberVisits[qIdx] / d->childNumberVisits[bestMoveIdx]);
//                if (d->qValues[qIdx]-qDiff > d->qValues[bestMoveIdx]) {
//                    mctsPolicy[bestMoveIdx] = d->childNumberVisits[qIdx];
//                    mctsPolicy[secondArg] = firstMax;
//                    bestMoveIdx = qIdx;
//                }
//            }
//        }
    }
    else {
        mctsPolicy = d->childNumberVisits;
        bestMoveIdx = argmax(d->childNumberVisits);
    }
    mctsPolicy /= sum(mctsPolicy);
}

void Node::get_principal_variation(vector<Action>& pv) const
{
    const Node* curNode = this;
    while (curNode != nullptr && curNode->is_playout_node() && !curNode->is_terminal()) {
        size_t childIdx = get_best_action_index(curNode, true);
        pv.push_back(curNode->get_action(childIdx));
        curNode = curNode->d->childNodes[childIdx];
    }
}

size_t get_best_action_index(const Node *curNode, bool fast)
{
    if (curNode->get_checkmate_idx() != NO_CHECKMATE) {
        // chose mating line
        return curNode->get_checkmate_idx();
    }
    if (curNode->get_node_type() == SOLVED_LOSS) {
        // choose node which delays the mate
        size_t longestPVlength = 0;
        size_t childIdx = 0;
        for (size_t idx = 0; idx < curNode->get_number_child_nodes(); ++idx) {
            if (curNode->get_child_nodes()[idx]->get_end_in_ply() > longestPVlength) {
                longestPVlength = curNode->get_child_nodes()[idx]->get_end_in_ply();
                childIdx = idx;
            }
        }
        return childIdx;
    }
    if (fast) {
        return argmax(curNode->get_child_number_visits());
    }
    DynamicVector<float> mctsPolicy(curNode->get_number_child_nodes());
    size_t bestMoveIdx;
    curNode->get_mcts_policy(mctsPolicy, bestMoveIdx);
    return bestMoveIdx;
}

bool Node::all_q_values_smaller_X(float thresh) const {
    for (auto qValue : d->get_q_values()) {
        if (qValue > thresh) {
            return false;
        }
    }
    return true;
}

size_t Node::select_child_node(const SearchSettings* searchSettings)
{
    if (!sorted) {
        prepare_node_for_visits();
    }
    if (d->noVisitIdx == 1) {
        return 0;
    }
    if (has_forced_win()) {
        return d->checkmateIdx;
    }
    if (is_unstable() && is_fully_expanded() && get_visits() > 10000 && all_q_values_smaller_X(-0.1f)) {
        disable_node_acces(this);
    }
    // find the move according to the q- and u-values for each move
    // calculate the current u values
    // it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
    return argmax(d->qValues + get_current_u_values(searchSettings));
}

const char* node_type_to_string(enum NodeType nodeType)
{
    switch(nodeType) {
    case SOLVED_WIN:
        return "WIN";
    case SOLVED_DRAW:
        return "DRAW";
    case SOLVED_LOSS:
        return "LOSS";
    default:
        return "UNSOLVED";
    }
}

NodeType flip_node_type(const enum NodeType nodeType) {
    switch(nodeType) {
    case SOLVED_WIN:
        return SOLVED_LOSS;
    case SOLVED_LOSS:
        return SOLVED_WIN;
    default:
        return nodeType;
    }
}

void delete_sibling_subtrees(Node* node, unordered_map<Key, Node*>& hashTable, GCThread<Node>& gcThread)
{
    if (node->get_parent_node() != nullptr) {
        info_string("delete unused subtrees");
        for (Node* childNode: node->get_parent_node()->get_child_nodes()) {
            if (childNode != node) {
                delete_subtree_and_hash_entries(childNode, hashTable, gcThread);
            }
        }
    }
}

void delete_subtree_and_hash_entries(Node* node, unordered_map<Key, Node*>& hashTable, GCThread<Node>& gcThread)
{
    if (node == nullptr) {
        return;
    }
    // if the current node hasn't been expanded or is a terminal node then childNodes is empty and the recursion ends
    if (node->is_sorted()) {
        for (Node* childNode: node->get_child_nodes()) {
            delete_subtree_and_hash_entries(childNode, hashTable, gcThread);
        }
    }
    // the board position is only filled if the node has been extended
    auto it = hashTable.find(node->hash_key());
    if(it != hashTable.end()) {
        hashTable.erase(node->hash_key());
    }
    gcThread.add_item_to_delete(node);
}

float get_visits(Node* node)
{
    return node->get_visits();
}

float get_current_q_thresh(const SearchSettings* searchSettings, int numberVisits)
{
    return searchSettings->qThreshMax - exp(-numberVisits / searchSettings->qThreshBase) * (searchSettings->qThreshMax - searchSettings->qThreshInit);
}

float get_current_cput(float visits, const SearchSettings* searchSettings)
{
    return log((visits + searchSettings->cpuctBase + 1) / searchSettings->cpuctBase) + searchSettings->cpuctInit;
}

void Node::print_node_statistics(const StateObj* state) const
{
    const string header = "  #  | Move  |    Visits    |  Policy   |  Q-values  |  CP   |    Type    ";
    const string filler = "-----+-------+--------------+-----------+------------+-------+------------";
    cout << header << endl
       << std::showpoint << std::fixed << std::setprecision(7) // << std::noshowpcout
       << filler << endl;
    for (size_t childIdx = 0; childIdx < get_number_child_nodes(); ++childIdx) {
        size_t n = 0;
        float q = Q_INIT;
        if (childIdx < d->noVisitIdx) {
            n = d->childNumberVisits[childIdx];
            q = max(d->qValues[childIdx], -1.0f);
        }

        const Action move = get_legal_action()[childIdx];
        cout << " " << setfill('0') << setw(3) << childIdx << " | " << setfill(' ');
        if (state == nullptr) {
            cout << setw(5) << action_to_uci(move, false) << " | ";
        }
        else {
            cout << setw(5) << state->action_to_san(move, get_legal_action(), false, false) << " | ";
        }
        cout << setw(12) << n << " | "
             << setw(9) << policyProbSmall[childIdx] << " | "
             << setw(10) << q << " | "
             << setw(5) << value_to_centipawn(q) << " | ";
        if (childIdx < get_no_visit_idx() && d->childNodes[childIdx] != nullptr && d->childNodes[childIdx]->d != nullptr && d->childNodes[childIdx]->get_node_type() != UNSOLVED) {
            cout << setfill(' ') << setw(4) << node_type_to_string(flip_node_type(NodeType(d->childNodes[childIdx]->d->nodeType)))
               << " in " << setfill('0') << setw(2) << d->childNodes[childIdx]->d->endInPly+1;
        }
        else {
            cout << setfill(' ') << setw(9) << node_type_to_string(UNSOLVED);
        }
        cout << endl;
    }
    cout << filler << endl
       << "initial value:\t" << get_value() << endl
       << "nodeType:\t" << node_type_to_string(NodeType(d->nodeType)) << endl
       << "isTerminal:\t" << is_terminal() << endl
       << "isTablebase:\t" << is_tablebase() << endl
       << "unsolvedNodes:\t" << d->numberUnsolvedChildNodes << endl
       << "Visits:\t\t" << get_visits() << endl
       << "terminalVisits:\t" << get_terminal_visits() << endl;
}

void Node::make_to_cell()
{
    isCell = true;
}

bool is_terminal_value(float value)
{
    return (value == WIN || value == DRAW || value == LOSS);
}

size_t get_node_count(const Node *node)
{
    return node->get_visits() - node->get_terminal_visits();
}

deque<size_t> get_trajectory(Node* currentNode)
{
    deque<size_t> trajectory;
    while (!currentNode->is_root_node()) {
        trajectory.emplace_front(currentNode->get_child_idx_for_parent());
        currentNode = currentNode->get_parent_node();
    }
    return trajectory;
}

void disable_node_acces(Node* node)
{
    Node* parentNode = node->get_parent_node();
    // disable drawing move for selection
    if (parentNode != nullptr) {
        parentNode->lock();
        parentNode->disable_action(node->get_child_idx_for_parent());
        parentNode->mark_as_unstable();
        parentNode->unlock();
    }
}
