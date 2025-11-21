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

double Node::get_q_sum_virtual_loss(ChildIdx childIdx) const
{
    return get_child_number_visits(childIdx) * double(get_q_value(childIdx)) + get_virtual_loss_counter(childIdx);
}

bool Node::is_transposition() const
{
    return numberParentNodes != 1;
}

void Node::decrement_number_parents()
{
    --numberParentNodes;
}

uint8_t Node::get_virtual_loss_counter(ChildIdx childIdx) const
{
    return d->virtualLossCounter[childIdx];
}

bool Node::has_transposition_child_node()
{
    for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
        const Node* childNode = it->get();
        if (childNode != nullptr && childNode->is_transposition()) {
            return true;
        }
    }
    return false;
}

#ifdef MCTS_STORE_STATES
StateObj* Node::get_state() const
{
    return state.get();
}

void Node::set_auxiliary_outputs(const float* auxiliaryOutputs)
{
    state->set_auxiliary_outputs(auxiliaryOutputs);
}
#endif

Node::Node(StateObj* state, const SearchSettings* searchSettings):
    legalActions(state->legal_actions()),
    key(state->hash_key()),
    valueSum(0),
    d(nullptr),
    #ifdef MCTS_STORE_STATES
    state(state),
    #endif
    realVisitsSum(0),
    pliesFromNull(state->steps_from_null()),
    numberParentNodes(1),
    isTerminal(false),
    isTablebase(false),
    hasNNResults(false),
    sorted(false)
{
    // specify the number of direct child nodes of this node
    check_for_terminal(state);
#ifdef MCTS_TB_SUPPORT
    if (searchSettings->useTablebase && !isTerminal) {
        check_for_tablebase_wdl(state);
    }
#endif
    policyProbSmall.resize(legalActions.size());
}

bool Node::solved_win(const Node* childNode, const SearchSettings* searchSettings) const
{
    switch (searchSettings->searchPlayerMode) {
    case MODE_TWO_PLAYER:
        return childNode->d->nodeType == LOSS;
    case MODE_SINGLE_PLAYER:
    default:
        return childNode->d->nodeType == WIN;
    }
}

bool Node::solved_draw(const Node* childNode, const SearchSettings* searchSettings) const
{
    if (d->numberUnsolvedChildNodes == 0) {
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            if (childNode->d->nodeType != LOSS) {
                return at_least_one_drawn_child();
            }
            break;
        case MODE_SINGLE_PLAYER:
            if (childNode->d->nodeType != WIN) {
                return at_least_one_drawn_child();
            }
        }
    }
    return false;
}

bool Node::at_least_one_drawn_child() const
{
    bool atLeastOneDrawnChild = false;
    for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
        const Node* childNode = it->get();
        if (!childNode->is_playout_node() || (childNode->d->nodeType != DRAW && childNode->d->nodeType != WIN)) {
            return false;
        }
        if (childNode->d->nodeType == DRAW) {
            atLeastOneDrawnChild = true;
        }
    }
    return atLeastOneDrawnChild;
}

bool Node::only_won_child_nodes() const
{
    return only_child_nodes_of_one_kind<WIN>();
}

bool Node::solved_loss(const Node* childNode, const SearchSettings* searchSettings) const
{
    if (d->numberUnsolvedChildNodes == 0) {
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            if (childNode->d->nodeType == WIN) {
                return only_won_child_nodes();
            }
            break;
        case MODE_SINGLE_PLAYER:
            if (childNode->d->nodeType == LOSS) {
                return only_child_nodes_of_one_kind<LOSS>();
            }
        }
    }
    return false;
}

void Node::mark_as_loss()
{
    set_value(LOSS_VALUE);
    d->nodeType = LOSS;
}

void Node::mark_as_draw()
{
    set_value(DRAW_VALUE);
    d->nodeType = DRAW;
}

void Node::mark_as_win()
{
    set_value(WIN_VALUE);
    d->nodeType = WIN;
}

#ifdef MCTS_TB_SUPPORT
bool Node::solve_tb_win(const Node* childNode, const SearchSettings* searchSettings) const
{
    switch (searchSettings->searchPlayerMode) {
    case MODE_TWO_PLAYER:
        return childNode->d->nodeType == TB_LOSS;
    case MODE_SINGLE_PLAYER:
    default:
        return childNode->d->nodeType == TB_WIN;
    }
}

bool Node::solved_tb_draw(const Node* childNode, const SearchSettings* searchSettings) const
{
    if (d->numberUnsolvedChildNodes == 0) {
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            if (childNode->d->nodeType != TB_LOSS || childNode->d->nodeType == LOSS) {
                return at_least_one_drawn_child();
            }
            break;
        case MODE_SINGLE_PLAYER:
            if (childNode->d->nodeType != TB_WIN || childNode->d->nodeType == WIN) {
                return at_least_one_drawn_child();
            }
        }
    }
    return false;
}

bool Node::solved_tb_loss(const Node* childNode, const SearchSettings* searchSettings) const
{
    if (d->numberUnsolvedChildNodes == 0) {
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            if (childNode->d->nodeType == TB_WIN) {
                return only_won_tb_child_nodes();
            }
            break;
        case MODE_SINGLE_PLAYER:
            if (childNode->d->nodeType == TB_LOSS) {
                return only_child_nodes_of_one_kind<TB_LOSS>();
            }
        }
    }
    return false;
}

bool Node::only_won_tb_child_nodes() const
{
    return only_child_nodes_of_one_kind<TB_WIN>();
}

void Node::mark_as_tb_loss()
{
    set_value(LOSS_VALUE);
    d->nodeType = TB_LOSS;
}

void Node::mark_as_tb_draw()
{
    set_value(DRAW_VALUE);
    d->nodeType = TB_DRAW;
}

void Node::mark_as_tb_win()
{
    set_value(WIN_VALUE);
    d->nodeType = TB_WIN;
}
#endif

void Node::define_end_ply_for_solved_terminal(const Node* childNode)
{
    if (d->nodeType == LOSS) {
        // choose the longest pv line
        for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
            const Node* curChildNode = it->get();
            if (curChildNode->d->endInPly+1 > d->endInPly) {
                d->endInPly = curChildNode->d->endInPly+1;
            }
        }
        return;
    }
    if (d->nodeType == DRAW) {
        // choose the shortest pv line for draws
        for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
            const Node* curChildNode = it->get();
            if (curChildNode->d->nodeType == DRAW && curChildNode->d->endInPly+1 < d->endInPly) {
                d->endInPly = curChildNode->d->endInPly+1;
            }
        }
        return;
    }
    // get the endPly for WINS
    d->endInPly = childNode->d->endInPly + 1;
}

template <int targetValue>
void Node::update_solved_terminal(const Node* childNode, ChildIdx childIdx)
{
    define_end_ply_for_solved_terminal(childNode);
    set_value(targetValue);
    d->qValues[childIdx] = targetValue;
}

void Node::mcts_policy_based_on_wins(DynamicVector<double> &mctsPolicy, const SearchSettings* searchSettings) const
{
    mctsPolicy = 0;
    ChildIdx childIdx = 0;
    for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
        const Node* childNode = it->get();
        if (childNode != nullptr && childNode->d != nullptr) {
            switch (searchSettings->searchPlayerMode) {
            case MODE_TWO_PLAYER:
                if (childNode->d->nodeType == LOSS) {
                    mctsPolicy[childIdx] = 1.0f;
                }
                break;
            case MODE_SINGLE_PLAYER:
                if (childNode->d->nodeType == WIN) {
                    mctsPolicy[childIdx] = 1.0f;
                }
            }
        }
        ++childIdx;
    }
}

void Node::mcts_policy_based_on_losses(DynamicVector<double> &mctsPolicy) const
{
    mctsPolicy = 0;
    ChildIdx childIdx = 0;
    ChildIdx longestChildIdx = 0;
    size_t endInPly = 0;
    for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
        const Node* childNode = it->get();
        if (childNode != nullptr && childNode->d != nullptr) {
            if (childNode->d->endInPly > endInPly) {
                endInPly = childNode->d->endInPly;
                longestChildIdx = childIdx;
            }
        }
        ++childIdx;
    }
    mctsPolicy[longestChildIdx] = 1.0f;
}

void Node::prune_losses_in_mcts_policy(DynamicVector<double> &mctsPolicy, const SearchSettings* searchSettings) const
{
    // check if PV line leads to a loss
    if (d->numberUnsolvedChildNodes != get_number_child_nodes() && !is_loss_node_type(d->nodeType)) {
        // set all entries which lead to a WIN of the opponent to zero
        for (size_t childIdx = 0; childIdx < d->noVisitIdx; ++childIdx) {
            const Node* childNode = d->childNodes[childIdx].get();
            if (childNode != nullptr && childNode->is_playout_node()) {
                switch (searchSettings->searchPlayerMode) {
                case MODE_TWO_PLAYER:
                    if (is_win_node_type(childNode->d->nodeType)) {
                        mctsPolicy[childIdx] = 0;
                    }
                    break;
                case MODE_SINGLE_PLAYER:
                    if (is_loss_node_type(childNode->d->nodeType)) {
                        mctsPolicy[childIdx] = 0;
                    }
                }
            }
        }
    }
}

bool Node::solve_for_terminal(ChildIdx childIdx, const SearchSettings* searchSettings)
{
    const Node* childNode = d->childNodes[childIdx].get();

    if (!childNode->is_playout_node()) {
        return false;
    }
    if (childNode->d->nodeType == UNSOLVED) {
        return false;
    }

#ifdef MCTS_TB_SUPPORT
    // disable draws for winning tb positions
    if (d->nodeType == TB_WIN) {
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            if (is_win_node_type(childNode->d->nodeType) || is_draw_node_type(childNode->d->nodeType)) {
                disable_action(childIdx);
            }
            break;
        case MODE_SINGLE_PLAYER:
            if (is_loss_node_type(childNode->d->nodeType) || is_draw_node_type(childNode->d->nodeType)) {
                disable_action(childIdx);
            }
        }
    }
#endif

    if (d->nodeType == WIN || d->nodeType == LOSS || d->nodeType == DRAW) {
        // already solved
        return false;
    }

    if (d->nodeTypes[childIdx] == UNSOLVED) {
        --d->numberUnsolvedChildNodes;
        d->nodeTypes[childIdx] = childNode->d->nodeType;
        switch (searchSettings->searchPlayerMode) {
        case MODE_TWO_PLAYER:
            switch(childNode->d->nodeType) {
            case WIN:
                disable_action(childIdx);
            default: ; // pass
            }
            break;
        case MODE_SINGLE_PLAYER:
            switch(childNode->d->nodeType) {
            case LOSS:
                disable_action(childIdx);
            default: ; // pass
            }

        }
    }

    if (solved_win(childNode, searchSettings)) {
        d->nodeType = WIN;
        update_solved_terminal<WIN_VALUE>(childNode, childIdx);
        d->checkmateIdx = childIdx;
        return true;
    }
    if (solved_loss(childNode, searchSettings)) {
        d->nodeType = LOSS;
        update_solved_terminal<LOSS_VALUE>(childNode, childIdx);
        return true;
    }
    if (solved_draw(childNode, searchSettings)) {
        d->nodeType = DRAW;
        update_solved_terminal<DRAW_VALUE>(childNode, childIdx);
        return true;
    }
#ifdef MCTS_TB_SUPPORT
    if (solve_tb_win(childNode, searchSettings)) {
        d->nodeType = TB_WIN;
        update_solved_terminal<WIN_VALUE>(childNode, childIdx);
        return true;
    }
    if (solved_tb_draw(childNode, searchSettings)) {
        d->nodeType = TB_DRAW;
        update_solved_terminal<DRAW_VALUE>(childNode, childIdx);
        return true;
    }
    if (solved_tb_loss(childNode, searchSettings)) {
        d->nodeType = TB_LOSS;
        update_solved_terminal<LOSS_VALUE>(childNode, childIdx);
        return true;
    }
#endif
    return false;
}

bool Node::is_root_node() const
{
    return numberParentNodes == 0;
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

Action Node::get_action(ChildIdx childIdx) const
{
    return legalActions[childIdx];
}

Node *Node::get_child_node(ChildIdx childIdx) const
{
    return d->childNodes[childIdx].get();
}

shared_ptr<Node> Node::get_child_node_shared(ChildIdx childIdx) const
{
    return d->childNodes[childIdx];
}

vector<shared_ptr<Node>>::const_iterator Node::get_node_it_begin() const
{
    return d->childNodes.begin();
}

vector<shared_ptr<Node>>::const_iterator Node::get_node_it_end() const
{
    return d->childNodes.end();
}

bool Node::is_terminal() const
{
    return isTerminal;
}

bool Node::has_nn_results() const
{
    return hasNNResults;
}

void Node::apply_virtual_loss_to_child(ChildIdx childIdx, const SearchSettings* searchSettings)
{
    // update the stats of the parent node
    // make it look like if one has lost X games from this node forward where X is the virtual loss value
    // temporarily reduce the attraction of this node by applying a virtual loss /
    // the effect of virtual loss will be undone if the playout is over
    switch (get_virtual_style(searchSettings, d->childNumberVisits[childIdx])) {
    case VIRTUAL_LOSS:
        d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] - 1) / double(d->childNumberVisits[childIdx] + 1);
        break;
    case VIRTUAL_OFFSET:
        d->qValues[childIdx] -= searchSettings->virtualOffsetStrenght;
    case VIRTUAL_VISIT: ;  // ignore
    case VIRTUAL_MIX: ;  // unreachable
    }

    // virtual increase the number of visits
    ++d->childNumberVisits[childIdx];
    ++d->visitSum;

    // increment virtual loss counter
    update_virtual_loss_counter<true>(childIdx);
}

float Node::get_q_value(ChildIdx childIdx) const
{
    return d->qValues[childIdx];
}

DynamicVector<float> Node::get_q_values() const
{
    return d->qValues;
}

void Node::set_q_value(ChildIdx childIdx, float value)
{
    d->qValues[childIdx] = value;
}

ChildIdx Node::get_best_q_idx() const
{
    return argmax(d->qValues);
}

vector<ChildIdx> Node::get_q_idx_over_thresh(float qThresh)
{
    vector<ChildIdx> indices;
    for (ChildIdx idx = 0; idx < size(d->qValues); ++idx) {
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
    d->virtualLossCounter.reserve(numberChildNodes);
    d->nodeTypes.reserve(numberChildNodes);
}

void Node::increment_no_visit_idx()
{
    if (d->noVisitIdx < get_number_child_nodes()) {
        ++d->noVisitIdx;
        if (d->noVisitIdx == PRESERVED_ITEMS) {
            reserve_full_memory();
        }
        d->add_empty_node();
    }
}

void Node::fully_expand_node()
{
    if (!is_fully_expanded()) {
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
    return valueSum / realVisitsSum;
}

float Node::get_value_display() const
{
    if (is_win_node_type(d->nodeType)) {
        return WIN_VALUE;
    }
    if (is_loss_node_type(d->nodeType)) {
        return LOSS_VALUE;
    }
    if (is_draw_node_type(d->nodeType)) {
        return DRAW_VALUE;
    }
    return valueSum / realVisitsSum;
}

double Node::get_value_sum() const
{
    return valueSum;
}

uint32_t Node::get_real_visits() const
{
    return realVisitsSum;
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
    if (d == nullptr) {  // mark_tablebase() initializes the NodeData
        init_node_data();
    }
#ifdef MCTS_STORE_STATES
    state->prepare_action();
#endif
}

uint32_t Node::get_visits() const
{
    return d->visitSum;
}

uint32_t Node::get_real_visits(ChildIdx childIdx) const
{
    return d->childNumberVisits[childIdx] - d->virtualLossCounter[childIdx];
}

void backup_collision(const SearchSettings* searchSettings, const Trajectory& trajectory) {
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        it->node->revert_virtual_loss(it->childIdx, searchSettings);
    }
}

void Node::revert_virtual_loss(ChildIdx childIdx, const SearchSettings* searchSettings)
{
    lock();
    switch (get_virtual_style(searchSettings, d->childNumberVisits[childIdx])) {
    case VIRTUAL_LOSS:
        d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] + 1) / (d->childNumberVisits[childIdx] - 1);
        break;
    case VIRTUAL_OFFSET:
        d->qValues[childIdx] += searchSettings->virtualOffsetStrenght;
    case VIRTUAL_MIX: ; // ignore
    case VIRTUAL_VISIT: ; // ignore
    }
    --d->childNumberVisits[childIdx];
    --d->visitSum;

    // decrement virtual loss counter
    update_virtual_loss_counter<false>(childIdx);
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

uint16_t Node::get_no_visit_idx() const
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
    ++this->realVisitsSum;
    this->valueSum = value * this->realVisitsSum;
}

Node* Node::add_new_node_to_tree(MapWithMutex* mapWithMutex, StateObj* newState, ChildIdx childIdx, const SearchSettings* searchSettings, bool& transposition)
{
    if(searchSettings->useMCGS) {
        mapWithMutex->mtx.lock();
        HashMap::const_iterator it = mapWithMutex->hashTable.find(newState->hash_key());
        if (it != mapWithMutex->hashTable.end()) {
            shared_ptr<Node> transpositionNode = it->second.lock();
            Node* tranpositionNode = get_child_node(childIdx);
            if (tranpositionNode != nullptr) {
                if(is_transposition_verified(tranpositionNode, newState)) {
                    d->childNodes[childIdx] = atomic_load(&transpositionNode);
                    mapWithMutex->mtx.unlock();
                    tranpositionNode->lock();
                    tranpositionNode->add_transposition_parent_node();
                    tranpositionNode->unlock();
                    switch (searchSettings->searchPlayerMode) {
                    case MODE_TWO_PLAYER:
                        if (tranpositionNode->is_playout_node() && tranpositionNode->get_node_type() == LOSS) {
                            set_checkmate_idx(childIdx);
                        }
                    case MODE_SINGLE_PLAYER: ;
                    }
                    transposition = true;
                    return tranpositionNode;
                }
            }
        }
        mapWithMutex->mtx.unlock();
    }

    // connect the Node to the parent
    shared_ptr<Node> newNode = make_shared<Node>(newState, searchSettings);
    atomic_store(&d->childNodes[childIdx], newNode);
    if (searchSettings->useMCGS) {
        mapWithMutex->mtx.lock();
        mapWithMutex->hashTable.insert({d->childNodes[childIdx]->hash_key(), d->childNodes[childIdx]});
        mapWithMutex->mtx.unlock();
    }
    transposition = false;
    return d->childNodes[childIdx].get();
}

void Node::add_transposition_parent_node()
{
    ++numberParentNodes;
}

float Node::max_policy_prob()
{
    return max(policyProbSmall);
}

ChildIdx Node::max_q_child() const
{
    return argmax(d->qValues);
}

ChildIdx Node::max_visits_child() const
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
    case WIN:
        return WIN_VALUE;
    case DRAW:
        return DRAW_VALUE;
    case LOSS:
        return LOSS_VALUE;
#ifdef MCTS_TB_SUPPORT
    case TB_WIN:
        return WIN_VALUE;
    case TB_DRAW:
        return DRAW_VALUE;
    case TB_LOSS:
        return LOSS_VALUE;
#endif
    default: ;  // UNSOLVED
    }
    return d->qValues[argmax(d->childNumberVisits)];
}

std::vector<Action> Node::get_legal_actions() const
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

uint32_t Node::get_child_number_visits(ChildIdx childIdx) const
{
    return d->childNumberVisits[childIdx];
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

NodeType Node::get_node_type() const
{
    return d->nodeType;
}

uint16_t Node::get_end_in_ply() const
{
    return d->endInPly;
}

uint32_t Node::get_free_visits() const
{
    return d->freeVisits;
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
    d = make_unique<NodeData>();
    sorted = true;
    d->noVisitIdx = 0;
}

void Node::check_for_terminal(StateObj* pos)
{
    float customValue;
    TerminalType terminalType = pos->is_terminal(get_number_child_nodes(), customValue);

    if (terminalType != TERMINAL_NONE) {
        mark_as_terminal();
        switch(terminalType) {
        case TERMINAL_WIN:
            mark_as_win();
            break;
        case TERMINAL_DRAW:
            mark_as_draw();
            legalActions.clear();  // in case of non stale-mate, legal actions could have moves
            break;
        case TERMINAL_LOSS:
            mark_as_loss();
            break;
        case TERMINAL_CUSTOM:
            set_value(customValue);
        case TERMINAL_NONE:
            ;  // pass
        }
    }
}

#ifdef MCTS_TB_SUPPORT
void Node::check_for_tablebase_wdl(StateObj* state)
{
    Tablebase::ProbeState result;
    Tablebase::WDLScore wdlScore = state->check_for_tablebase_wdl(result);

    if (result != Tablebase::FAIL) {
        mark_as_tablebase();
        switch(wdlScore) {
        case Tablebase::WDLLoss:
            mark_as_tb_loss();
            break;
        case Tablebase::WDLWin:
            mark_as_tb_win();
            break;
        default:
            mark_as_tb_draw();
        }
    }
    // default: isTablebase = false;
}

void Node::mark_as_tablebase()
{
    init_node_data();
    isTablebase = true;
}
#endif

void Node::make_to_root()
{
    numberParentNodes = 0;
}

void Node::lock()
{
    spinlock.lock();
}

void Node::unlock()
{
    spinlock.unlock();
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

void Node::set_probabilities_for_moves(const float *data, bool mirrorPolicy)
{
    // allocate sufficient memory -> is assumed that it has already been done
    assert(legalActions.size() == policyProbSmall.size());
    for (size_t mvIdx = 0; mvIdx < legalActions.size(); ++mvIdx) {
        // retrieve vector index from look-up table
        // set the right prob value
        // accessing the data on the raw floating point vector is faster
        // than calling policyProb.At(batchIdx, vectorIdx)
        if (mirrorPolicy) {
            // use mirrored action_to_index
            policyProbSmall[mvIdx] = data[StateConstants::action_to_index<normal,mirrored>(legalActions[mvIdx])];
        }
        else {
            // use non-mirrored action_to_index
            policyProbSmall[mvIdx] = data[StateConstants::action_to_index<normal,notMirrored>(legalActions[mvIdx])];
        }
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

double Node::get_transposition_q_value(const SearchSettings *searchSettings, ChildIdx childIdx, uint_fast32_t transposVisits)
{
    double transposQValue;
    switch(get_virtual_style(searchSettings, d->childNumberVisits[childIdx])) {
    case VIRTUAL_LOSS:
        transposQValue = get_q_sum_virtual_loss(childIdx) / transposVisits;
        break;
    case VIRTUAL_VISIT:
        transposQValue = get_q_value(childIdx);
        break;
    case VIRTUAL_OFFSET:
        transposQValue = double(get_q_value(childIdx)) + get_virtual_loss_counter(childIdx) * searchSettings->virtualOffsetStrenght;
    case VIRTUAL_MIX: ;
        // unreachable
    }
    if (searchSettings->searchPlayerMode == MODE_TWO_PLAYER) {
        return -transposQValue;
    }
    return transposQValue;
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
#ifdef SEARCH_UCT
    return searchSettings->cpuctInit * (sqrt(log(d->visitSum)) / (d->childNumberVisits + FLT_EPSILON));
#else
    return get_current_cput(d->visitSum, searchSettings) * blaze::subvector(policyProbSmall, 0, d->noVisitIdx) * (sqrt(d->visitSum) / (d->childNumberVisits + 1.0));
#endif
}

Node* Node::get_child_node(ChildIdx childIdx)
{
    return d->childNodes[childIdx].get();
}

void Node::get_mcts_policy(DynamicVector<double>& mctsPolicy, ChildIdx& bestMoveIdx, const SearchSettings* searchSettings) const
{
    // fill only the winning moves in case of a known win
    if (d->nodeType == WIN) {
        mctsPolicy = DynamicVector<float>(d->noVisitIdx);
        mcts_policy_based_on_wins(mctsPolicy, searchSettings);
    }
    else if (is_loss_node_type(d->nodeType)) {
        mctsPolicy = DynamicVector<float>(d->noVisitIdx);
        mcts_policy_based_on_losses(mctsPolicy);
    }
    else if (searchSettings->qValueWeight > 0) {
        ChildIdx secondArg;
        double firstMaxValue;
        double secondMaxValue;
        mctsPolicy = d->childNumberVisits;
        prune_losses_in_mcts_policy(mctsPolicy, searchSettings);
        ChildIdx bestQIdx = argmax(d->qValues);
        first_and_second_max(mctsPolicy, ChildIdx(d->noVisitIdx), firstMaxValue, secondMaxValue, bestMoveIdx, secondArg);
        if (searchSettings->qVetoDelta != 0 && bestQIdx != bestMoveIdx && d->qValues[bestQIdx] > d->qValues[bestMoveIdx] + searchSettings->qVetoDelta && d->childNumberVisits[bestQIdx] > 1) {
            if (mctsPolicy[bestMoveIdx] > mctsPolicy[bestQIdx]) {
                // swap values of highest qValues and most visits
                info_string("veto move");
                const double qSavePolicy = mctsPolicy[bestQIdx];
                mctsPolicy[bestQIdx] = mctsPolicy[bestMoveIdx];
                mctsPolicy[bestMoveIdx] = qSavePolicy;
            }
        }
        else if (bestMoveIdx != secondArg && d->qValues[secondArg] > d->qValues[bestMoveIdx]) {
            const float qDiff = d->qValues[secondArg] - d->qValues[bestMoveIdx];
            mctsPolicy[secondArg] += qDiff * searchSettings->qValueWeight * mctsPolicy[bestMoveIdx];
        }
    }
    else {
        mctsPolicy = d->childNumberVisits;
        prune_losses_in_mcts_policy(mctsPolicy, searchSettings);
    }
    mctsPolicy /= sum(mctsPolicy);
    bestMoveIdx = argmax(mctsPolicy);
}

void Node::get_principal_variation(vector<Action>& pv, const SearchSettings* searchSettings)
{
    Node* curNode = this;
    while (curNode != nullptr && curNode->is_playout_node() && !curNode->is_terminal()) {
        curNode->lock();
        size_t childIdx = get_best_action_index(curNode, true, searchSettings);
        pv.push_back(curNode->get_action(childIdx));
        curNode->unlock();
        curNode = curNode->d->childNodes[childIdx].get();
    }
}

size_t get_best_action_index(const Node *curNode, bool fast, const SearchSettings* searchSettings)
{
    if (curNode->get_checkmate_idx() != NO_CHECKMATE) {
        // chose mating line
        return curNode->get_checkmate_idx();
    }
    if (curNode->get_node_type() == LOSS) {
        // choose node which delays the mate
        size_t longestPVlength = 0;
        size_t childIdx = 0;
        for (size_t idx = 0; idx < curNode->get_number_child_nodes(); ++idx) {
            if (curNode->get_child_node(idx)->get_end_in_ply() > longestPVlength) {
                longestPVlength = curNode->get_child_node(idx)->get_end_in_ply();
                childIdx = idx;
            }
        }
        return childIdx;
    }
    if (fast) {
        return argmax(curNode->get_child_number_visits());
    }
    DynamicVector<double> mctsPolicy(curNode->get_number_child_nodes());
    ChildIdx bestMoveIdx;
    curNode->get_mcts_policy(mctsPolicy, bestMoveIdx, searchSettings);
    return bestMoveIdx;
}

ChildIdx Node::select_child_node(const SearchSettings* searchSettings)
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

    assert(sum(d->childNumberVisits) == d->visitSum);
    // find the move according to the q- and u-values for each move
    // calculate the current u values
    // it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
    return argmax(d->qValues + get_current_u_values(searchSettings));
}

NodeSplit Node::select_child_nodes(const SearchSettings* searchSettings, uint_fast16_t budget)
{
    NodeSplit nodeSplit;

    if (!sorted) {
        prepare_node_for_visits();
    }
    if (d->noVisitIdx == 1) {
        nodeSplit.only_first(0, budget);
        return nodeSplit;
    }

    if (has_forced_win()) {
        nodeSplit.only_first(d->checkmateIdx, budget);
        return nodeSplit;
    }
    DynamicVector<float> q_u_sum = d->qValues + get_current_u_values(searchSettings);
    float firstMax;
    float secondMax;
    assert(q_u_sum.size() > 1);
    first_and_second_max(q_u_sum, ChildIdx(d->noVisitIdx), firstMax, secondMax, nodeSplit.firstArg, nodeSplit.secondArg);

    float firstShare = 0.5 + std::min(float(firstMax - secondMax), 0.5f);
    nodeSplit.firstBudget = firstShare * budget + 0.5;  // rounding
    nodeSplit.secondBudget = budget - nodeSplit.firstBudget;

    assert(nodeSplit.secondArg != nodeSplit.firstArg);
    return nodeSplit;
}

const char* node_type_to_string(enum NodeType nodeType)
{
    switch(nodeType) {
    case WIN:
        return "WIN";
    case DRAW:
        return "DRAW";
    case LOSS:
        return "LOSS";
#ifdef MCTS_TB_SUPPORT
    case TB_WIN:
        return "TB_WIN";
    case TB_DRAW:
        return "TB_DRAW";
    case TB_LOSS:
        return "TB_LOSS";
#endif
    default:
        return "UNSOLVED";
    }
}

NodeType flip_node_type(const enum NodeType nodeType) {
    switch(nodeType) {
    case WIN:
        return LOSS;
    case LOSS:
        return WIN;
#ifdef MCTS_TB_SUPPORT
    case TB_WIN:
        return TB_LOSS;
    case TB_LOSS:
        return TB_WIN;
#endif
    default:
        return nodeType;
    }
}

float get_visits(Node* node)
{
    return node->get_visits();
}

float get_current_cput(float visits, const SearchSettings* searchSettings)
{
    return log((visits + searchSettings->cpuctBase + 1) / searchSettings->cpuctBase) + searchSettings->cpuctInit;
}

void Node::print_node_statistics(const StateObj* state, const vector<size_t>& customOrdering, const SearchSettings* searchSettings) const
{
    const string header = "  #  | Move  |    Visits    |  Policy   |  Q-values  |  CP   |    Type    ";
    const string filler = "-----+-------+--------------+-----------+------------+-------+------------";
    cout << header << endl
         << std::showpoint << std::fixed << std::setprecision(7)
         << filler << endl;
    for (size_t idx = 0; idx < get_number_child_nodes(); ++idx) {
        const size_t childIdx = customOrdering.size() == get_number_child_nodes() ? customOrdering[idx] : idx;
        size_t n = 0;
        double q = Q_INIT;
        if (childIdx < d->noVisitIdx) {
            n = d->childNumberVisits[childIdx];
            q = d->qValues[childIdx];
        }

        const Action move = get_legal_actions()[childIdx];
        cout << " " << setfill('0') << setw(3) << childIdx << " | " << setfill(' ');
        if (state == nullptr) {
            cout << setw(5) << StateConstants::action_to_uci(move, false) << " | ";
        }
        else {
            cout << setw(5) << state->action_to_san(move, get_legal_actions(), false, false) << " | ";
        }
        cout << setw(12) << n << " | "
             << setw(9) << policyProbSmall[childIdx] << " | "
             << setw(10) << max(q, -9.9999999) << " | "
             << setw(5) << value_to_centipawn(q) << " | ";
        if (childIdx < get_no_visit_idx() && d->childNodes[childIdx] != nullptr && d->childNodes[childIdx]->d != nullptr && d->childNodes[childIdx]->get_node_type() != UNSOLVED) {
            string nodeTypeToPrint;
            switch (searchSettings->searchPlayerMode) {
                case MODE_TWO_PLAYER:
                nodeTypeToPrint = node_type_to_string(flip_node_type(NodeType(d->childNodes[childIdx]->d->nodeType)));
                break;
            case MODE_SINGLE_PLAYER:
                nodeTypeToPrint = node_type_to_string(NodeType(d->childNodes[childIdx]->d->nodeType));

            }
            cout << setfill(' ') << setw(4) << nodeTypeToPrint << " in " << setfill('0') << setw(2) << d->childNodes[childIdx]->d->endInPly+1;
        }
        else {
            cout << setfill(' ') << setw(9) << node_type_to_string(UNSOLVED);
        }
        cout << endl;
    }
    cout << filler << endl
         << "value:\t" << get_value() << endl
         << "nodeType:\t" << node_type_to_string(NodeType(d->nodeType)) << endl
         << "isTerminal:\t" << is_terminal() << endl
         << "isTablebase:\t" << is_tablebase() << endl
         << "unsolvedNodes:\t" << d->numberUnsolvedChildNodes << endl
         << "Visits:\t\t" << get_visits() << endl
         << "freeVisits:\t" << get_free_visits() << "/" << get_visits() << endl;
}

uint32_t Node::get_node_count() const
{
    return get_visits() - get_free_visits();
}

bool Node::is_transposition_return(double myQvalue) const
{
    return abs(myQvalue - get_value()) > Q_TRANSPOS_DIFF;
}

void Node::set_checkmate_idx(ChildIdx childIdx) const
{
    d->checkmateIdx = childIdx;
}

bool Node::was_inspected()
{
    return d->inspected;
}

void Node::set_as_inspected()
{
    d->inspected = true;
}

uint32_t Node::get_number_of_nodes() const
{
    return get_visits() - get_free_visits();
}

bool is_terminal_value(float value)
{
    return (value == WIN_VALUE || value == DRAW_VALUE || value == LOSS_VALUE);
}

float get_transposition_backup_value(uint_fast32_t transposVisits, double transposQValue, double targetQValue)
{
    return std::clamp(transposVisits * (targetQValue - transposQValue) + targetQValue, double(LOSS_VALUE), double(WIN_VALUE));
}

bool is_transposition_verified(const Node* node, const StateObj* state) {
    return  node->has_nn_results() &&
            node->plies_from_null() == state->steps_from_null() &&
            state->number_repetitions() == 0;
}
