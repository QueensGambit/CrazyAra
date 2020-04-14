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
#include "syzygy/tbprobe.h"
#include "util/blazeutil.h" // get_dirichlet_noise()
#include "constants.h"
#include "../util/sfutil.h"
#include "../util/communication.h"


Node::Node(Board *pos, bool inCheck, Node *parentNode, size_t childIdxForParent, SearchSettings* searchSettings):
    key(pos->get_state_info()->key),
    pliesFromNull(pos->get_state_info()->pliesFromNull),
    sideToMove(pos->side_to_move()),
    parentNode(parentNode),
    visits(1),
    nodeType(UNSOLVED),
    endInPly(0),
    noVisitIdx(1),
    isTerminal(false),
    isTablebase(false),
    childIdxForParent(childIdxForParent),
    hasNNResults(false),
    isFullyExpanded(false),
    checkmateIdx(-1),
    searchSettings(searchSettings)
{
    fill_child_node_moves(pos);

    // specify the number of direct child nodes of this node
    numberChildNodes = legalMoves.size();
    numberUnsolvedChildNodes = numberChildNodes;
    mark_enhanced_moves(pos);

    check_for_terminal(pos, inCheck);
#ifdef MODE_CHESS
    if (!isTerminal) {
        check_for_tablebase_wdl(pos);
    }
#endif

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
    key = b.key;
    pliesFromNull = b.plies_from_null();
    sideToMove = b.side_to_move();
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
    visits = 1;
    childNodes.resize(numberChildNodes);
    //    parentNode = // is not copied
    //    childIdxForParent = // is not copied
    noVisitIdx = 1; // reset counter
    isTerminal = b.isTerminal;
    isTablebase = b.isTablebase;
    hasNNResults = b.hasNNResults;
    checkmateIdx = b.checkmateIdx;
    numberUnsolvedChildNodes = numberChildNodes;
    nodeType = UNSOLVED;
    endInPly = 0;
    searchSettings = b.searchSettings;
    isFullyExpanded = false;
}

void Node::fill_child_node_moves(Board* pos)
{
    // generate the legal moves and save them in the list
    for (const ExtMove& move : MoveList<LEGAL>(*pos)) {
        legalMoves.push_back(move);
    }
}

bool Node::solved_win(const Node* childNode) const
{
    if (childNode->nodeType == SOLVED_LOSS) {
        return true;
    }
    return false;
}

bool Node::solved_draw(const Node* childNode) const
{
    if (numberUnsolvedChildNodes == 0 &&
            (childNode->nodeType == SOLVED_DRAW || childNode->nodeType == SOLVED_WIN)) {
        return at_least_one_drawn_child();
    }
    return false;
}

bool Node::at_least_one_drawn_child() const
{
    bool atLeastOneDrawnChild = false;
    for (Node* childNode : childNodes) {
        if (childNode->nodeType != SOLVED_DRAW && childNode->nodeType != SOLVED_WIN) {
            return false;
        }
        if (childNode->nodeType == SOLVED_DRAW) {
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
    for (Node* childNode : childNodes) {
        if (childNode->nodeType != SOLVED_WIN) {
            return false;
        }
    }
    return true;
}

bool Node::solved_loss(const Node* childNode) const
{
    if (numberUnsolvedChildNodes == 0 && childNode->nodeType == SOLVED_WIN) {
        return only_won_child_nodes();
    }
    return false;
}

void Node::mark_as_loss()
{
    value = LOSS;
    parentNode->lock();
    parentNode->checkmateIdx = int(childIdxForParent);
    parentNode->nodeType = SOLVED_WIN;
    parentNode->numberUnsolvedChildNodes--;
    parentNode->endInPly = 1;
    parentNode->unlock();
    if (parentNode->parentNode != nullptr) {
        parentNode->parentNode->lock();
        if (parentNode->parentNode->numberUnsolvedChildNodes > 0) {
            parentNode->parentNode->numberUnsolvedChildNodes--;
        }
        if (parentNode->parentNode->is_root_node()) {
            parentNode->parentNode->disable_move(parentNode->childIdxForParent);
        }
        parentNode->parentNode->unlock();
    }
    nodeType = SOLVED_LOSS;
}

void Node::mark_as_draw()
{
    value = DRAW;
    nodeType = SOLVED_DRAW;
    if (parentNode != nullptr) {
        parentNode->lock();
        parentNode->numberUnsolvedChildNodes--;
        parentNode->endInPly = 1;
        parentNode->unlock();
    }
}

void Node::define_end_ply_for_solved_terminal(const Node* childNode)
{
    if (nodeType == SOLVED_LOSS) {
        // choose the longest pv line
        for (const Node* curChildNode : childNodes) {
            if (curChildNode->endInPly+1 > endInPly) {
                endInPly = curChildNode->endInPly+1;
            }
        }
        return;
    }
    if (nodeType == SOLVED_DRAW) {
        // choose the shortest pv line for draws
        for (const Node* curChildNode : childNodes) {
            if (curChildNode->nodeType == SOLVED_DRAW && curChildNode->endInPly+1 < endInPly) {
                endInPly = curChildNode->endInPly+1;
            }
        }
        return;
    }
    // get the endPly for WINS
    endInPly = childNode->endInPly + 1;
}

void Node::update_solved_terminal(const Node* childNode, int targetValue)
{
    define_end_ply_for_solved_terminal(childNode);
    value = targetValue;
    if (parentNode != nullptr) {
        parentNode->lock();
        parentNode->numberUnsolvedChildNodes--;
        parentNode->qValues[childIdxForParent] = targetValue;
        if (targetValue == LOSS) {
            parentNode->checkmateIdx = childIdxForParent;
        }
        else if (targetValue == WIN && parentNode->is_root_node()) {
            parentNode->disable_move(childIdxForParent);
        }
        parentNode->unlock();
    }
}

void Node::mcts_policy_based_on_wins(DynamicVector<float> &mctsPolicy) const
{
    mctsPolicy = 0;
    for (size_t childIdx = 0; childIdx < numberChildNodes; ++childIdx) {
        if (childNodes[childIdx] != nullptr && childNodes[childIdx]->nodeType == SOLVED_LOSS) {
            mctsPolicy[childIdx] = 1.0f;
        }
    }
    mctsPolicy /= sum(mctsPolicy);
}

void Node::prune_losses_in_mcts_policy(DynamicVector<float> &mctsPolicy) const
{
    // check if PV line leads to a loss
    if (nodeType != SOLVED_LOSS) {
        // set all entries which lead to a WIN of the oppenent to zero
        for (size_t childIdx = 0; childIdx < numberChildNodes; ++childIdx) {
            if (childNodes[childIdx] != nullptr && childNodes[childIdx]->nodeType == SOLVED_WIN) {
                mctsPolicy[childIdx] = 0;
            }
        }
    }
}

void Node::mcts_policy_based_on_q_n(DynamicVector<float>& mctsPolicy) const
{
    DynamicVector<float> qValuePruned = qValues;
    qValuePruned = (qValuePruned + 1) * 0.5f;
    const DynamicVector<float> normalizedVisits = childNumberVisits / visits;
    const float quantile = get_quantile(normalizedVisits, 0.25f);
    for (size_t idx = 0; idx < numberChildNodes; ++idx) {
        if (childNumberVisits[idx] < quantile) {
            qValuePruned[idx] = 0;
        }
    }
    mctsPolicy = (1.0f - searchSettings->qValueWeight) * normalizedVisits + searchSettings->qValueWeight * qValuePruned;
}

void Node::solve_for_terminal(const Node* childNode)
{
    if (nodeType != UNSOLVED) {
        // already solved
        return;
    }
    if (solved_win(childNode)) {
        nodeType = SOLVED_WIN;
        update_solved_terminal(childNode, WIN);
        return;
    }
    if (solved_loss(childNode)) {
        nodeType = SOLVED_LOSS;
        update_solved_terminal(childNode, LOSS);
        return;
    }
    if (solved_draw(childNode)) {
        nodeType = SOLVED_DRAW;
        update_solved_terminal(childNode, DRAW);
    }
}

void Node::mark_nodes_as_fully_expanded()
{
    noVisitIdx = numberChildNodes;
    isFullyExpanded = true;
}

bool Node::is_root_node() const
{
    return parentNode == nullptr;
}

Node::~Node()
{
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
    return numberChildNodes;
}

float Node::get_visits() const
{
    return visits;
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
    if (searchSettings->useSolver) {
        solve_for_terminal(childNodes[childIdx]);
    }
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

void Node::add_transposition_child_node(Node* newNode, size_t childIdx)
{
    newNode->mtx.lock();
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

float Node::updated_value_eval() const
{
    switch(nodeType) {
    case SOLVED_WIN:
        return WIN;
    case SOLVED_DRAW:
        return DRAW;
    case SOLVED_LOSS:
        return LOSS;
    default: ;  // UNSOLVED
    }
    if (visits == 1) {
        return value;
    }
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

int Node::plies_from_null() const
{
    return pliesFromNull;
}

Color Node::side_to_move() const
{
    return sideToMove;
}

bool Node::is_tablebase() const
{
    return isTablebase;
}

uint8_t Node::get_node_type() const
{
    return nodeType;
}

uint16_t Node::get_end_in_ply() const
{
    return endInPly;
}

void Node::check_for_terminal(Board* pos, bool inCheck)
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
        if (inCheck) {
            mark_as_loss();
            return;
        }
        // we reached a stalmate
        mark_as_draw();
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
    if (pos->can_claim_3fold_repetition() || pos->is_50_move_rule_draw() || pos->draw_by_insufficient_material()) {
        // reached 3-fold-repetition or 50 moves rule draw or insufficient material
        mark_as_draw();
        isTerminal = true;
        return;
    }
    // normal game position
    //    isTerminal = false;  // is the default value
}

void Node::check_for_tablebase_wdl(Board *pos)
{
    Tablebases::ProbeState result;
    Tablebases::WDLScore wdlScore = probe_wdl(*pos, &result);

    if (result != Tablebases::FAIL) {
        isTablebase = true;
        switch(wdlScore) {
        case Tablebases::WDLLoss:
            value = LOSS;
            break;
        case Tablebases::WDLWin:
            value = WIN;
            break;
        default:
            value = DRAW;
        }
    }
    // default: isTablebase = false;
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
    policyProbSmall = softmax(policyProbSmall);
}

void Node::mark_enhanced_moves(const Board* pos)
{
    if (searchSettings->enhanceChecks || searchSettings->enhanceCaptures) {
        isCheck.resize(numberChildNodes);
        isCheck = false;
        isCapture.resize(numberChildNodes);
        isCapture = false;

        for (size_t idx = 0; idx < numberChildNodes; ++idx) {
            if (pos->capture(legalMoves[idx])) {
                isCapture[idx] = true;
            }
            if (pos->gives_check(legalMoves[idx])) {
                isCheck[idx] = true;
            }
        }
    }
}

void Node::disable_move(size_t childIdxForParent)
{
    policyProbSmall[childIdxForParent] = 0;
    actionValues[childIdxForParent] = -INT_MAX;
}

void Node::enhance_moves()
{
    if (!searchSettings->enhanceChecks && !searchSettings->enhanceCaptures) {
        return;
    }

    bool checkUpdate = false;
    bool captureUpdate = false;

    if (searchSettings->enhanceChecks) {
        checkUpdate = enhance_move_type(min(searchSettings->threshCheck, max(policyProbSmall)*searchSettings->checkFactor),
                                        searchSettings->threshCheck, legalMoves, isCheck, policyProbSmall);
    }
    if (searchSettings->enhanceCaptures) {
        captureUpdate = enhance_move_type(min(searchSettings->threshCapture, max(policyProbSmall)*searchSettings->captureFactor),
                                          searchSettings->threshCheck, legalMoves, isCapture, policyProbSmall);
    }

    if (checkUpdate || captureUpdate) {
        policyProbSmall /= sum(policyProbSmall);
    }
}

bool enhance_move_type(float increment, float thresh, const vector<Move>& legalMoves, const DynamicVector<bool>& moveType, DynamicVector<float>& policyProbSmall)
{
    bool update = false;
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        if (moveType[i] && policyProbSmall[i] < thresh) {
            policyProbSmall[i] += increment;
            update = true;
        }
    }
    return update;
}

bool is_check(const Board* pos, Move move)
{
    return pos->gives_check(move);
}

bool is_capture(const Board* pos, Move move)
{
    return pos->capture(move);
}

DynamicVector<float> Node::get_current_u_values()
{
    return get_current_cput() * blaze::subvector(policyProbSmall, 0, noVisitIdx) * (sqrt(visits) / (blaze::subvector(childNumberVisits, 0, noVisitIdx) + 1.f));
}

Node *Node::get_child_node(size_t childIdx)
{
    return childNodes[childIdx];
}

void Node::get_mcts_policy(DynamicVector<float>& mctsPolicy) const
{
    // fill only the winning moves in case of a known win
    if (nodeType == SOLVED_WIN) {
        mcts_policy_based_on_wins(mctsPolicy);
        return;
    }
    if (searchSettings->qValueWeight > 0) {
        mcts_policy_based_on_q_n(mctsPolicy);
    }
    else {
        mctsPolicy = childNumberVisits;
    }
    prune_losses_in_mcts_policy(mctsPolicy);
    mctsPolicy /= sum(mctsPolicy);
}

void Node::get_principal_variation(vector<Move>& pv) const
{
    pv.clear();
    const Node* curNode = this;
    do {
        size_t childIdx = get_best_move_index(curNode);
        pv.push_back(curNode->get_move(childIdx));
        curNode = curNode->childNodes[childIdx];
    } while (curNode != nullptr && !curNode->is_terminal());
}

size_t get_best_move_index(const Node *curNode)
{
    if (curNode->get_checkmate_idx() != -1) {
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
    if (curNode->get_node_type() == SOLVED_DRAW) {
        // choose shortest pv line
        size_t longestPVlength = curNode->get_end_in_ply();
        size_t childIdx = 0;
        for (size_t idx = 0; idx < curNode->get_number_child_nodes(); ++idx) {
            if (curNode->get_child_nodes()[idx]->get_node_type() == SOLVED_DRAW &&
                    curNode->get_child_nodes()[idx]->get_end_in_ply() < longestPVlength) {
                longestPVlength = curNode->get_child_nodes()[idx]->get_end_in_ply();
                childIdx = idx;
            }
        }
        return childIdx;
    }
    // default case
    DynamicVector<float> mctsPolicy(curNode->get_number_child_nodes());
    curNode->get_mcts_policy(mctsPolicy);
    return argmax(mctsPolicy);
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

ostream& operator<<(ostream &os, const Node *node)
{
    os << "  #  | Move  |   Visits    |  Policy   |  Q-values  |    Type    " << endl
       << std::showpoint << std::noshowpos << std::fixed << std::setprecision(7)
       << "-----+-------+-------------+-----------+------------+-------------" << endl;

    for (size_t childIdx = 0; childIdx < node->get_number_child_nodes(); ++childIdx) {
        os << " " << setfill('0') << setw(3) << childIdx << " | "
           << setfill(' ') << setw(5) << UCI::move(node->get_legal_moves()[childIdx], false) << " |"
           << setw(12) << int(node->childNumberVisits[childIdx]) << " | "
           << setw(9) << node->policyProbSmall[childIdx] << " | "
           << setw(10) << node->qValues[childIdx] << " | ";
        if (node->childNodes[childIdx] != nullptr && node->childNodes[childIdx]->get_node_type() != UNSOLVED) {
            os << setfill(' ') << setw(4) << node_type_to_string(flip_node_type(NodeType(node->childNodes[childIdx]->nodeType)))
               << " in " << setfill('0') << setw(2) << node->childNodes[childIdx]->endInPly+1;
        }
        else {
            os << setfill(' ') << setw(9) << node_type_to_string(UNSOLVED);
        }
        os << endl;
    }
    os << "-----+-------+-------------+-----------+------------+-------------" << endl
       << "initial value:\t" << node->get_value() << endl
       << "checkmateIdx:\t" << node->get_checkmate_idx() << endl
       << "nodeType:\t" << node_type_to_string(NodeType(node->nodeType)) << endl
       << "endInPly:\t" << node->endInPly << endl
       << "isTerminal:\t" << node->is_terminal() << endl
       << "isTablebase:\t" << node->is_tablebase() << endl
       << "unsolvedNodes:\t" << node->numberUnsolvedChildNodes << endl;

    return os;
}

void generate_dtz_values(const vector<Move> legalMoves, Board& pos, DynamicVector<int>& dtzValues) {
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(0));
    // fill dtz value vector
    for (size_t idx = 0; idx < legalMoves.size(); ++idx) {
        states->emplace_back();
        pos.do_move(legalMoves[idx], states->back());
        Tablebases::ProbeState result;
        int dtzValue = -probe_dtz(pos, &result);
        if (result != Tablebases::FAIL) {
            dtzValues[idx] = dtzValue;
        }
        else {
            info_string("DTZ tablebase look-up failed!");
        }
        pos.undo_move(legalMoves[idx]);
    }
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

void print_node_statistics(const Node* node)
{
    cout << node << endl;
}
