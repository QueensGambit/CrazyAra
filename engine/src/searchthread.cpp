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
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#ifdef TENSORRT
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"
#endif

#include <stdlib.h>
#include <climits>
#include "util/blazeutil.h"


size_t SearchThread::get_max_depth() const
{
    return depthMax;
}

SearchThread::SearchThread(NeuralNetAPI *netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex):
    NeuralNetAPIUser(netBatch),
    isRunning(false), mapWithMutex(mapWithMutex), searchSettings(searchSettings)
{
    searchLimits = nullptr;  // will be set by set_search_limits() every time before go()

    newNodes = make_unique<FixedVector<Node*>>(searchSettings->batchSize);
    newNodeSideToMove = make_unique<FixedVector<SideToMove>>(searchSettings->batchSize);
    transpositionValues = make_unique<FixedVector<float>>(searchSettings->batchSize*2);

    trajectoryBuffer.reserve(DEPTH_INIT);
    actionsBuffer.reserve(DEPTH_INIT);
}

void SearchThread::set_root_node(Node *value)
{
    rootNode = value;
    visitsPreSearch = rootNode->get_visits();
}

void SearchThread::set_search_limits(SearchLimits *s)
{
    searchLimits = s;
}

bool SearchThread::is_running() const
{
    return isRunning;
}

void SearchThread::set_is_running(bool value)
{
    isRunning = value;
}

NodeBackup SearchThread::add_new_node_to_tree(StateObj* newState, Node* parentNode, ChildIdx childIdx, bool inCheck)
{
    if(searchSettings->useTranspositionTable) {
        mapWithMutex->mtx.lock();
        unordered_map<Key, Node*>::const_iterator it = mapWithMutex->hashTable.find(newState->hash_key());
        if(it != mapWithMutex->hashTable.end() &&
                is_transposition_verified(it, newState)) {
            mapWithMutex->mtx.unlock();
            it->second->lock();
            const float qValue =  it->second->get_value();

            it->second->add_transposition_parent_node();
            it->second->unlock();
    #ifndef MODE_POMMERMAN
            if (it->second->is_playout_node() && it->second->get_node_type() == SOLVED_LOSS) {
                parentNode->set_checkmate_idx(childIdx);
            }
    #endif
            transpositionValues->add_element(qValue);
            parentNode->add_new_child_node(it->second, childIdx);
            return NODE_TRANSPOSITION;
        }
        mapWithMutex->mtx.unlock();
    }
    assert(parentNode != nullptr);
    Node *newNode = new Node(newState, inCheck, searchSettings);
    // connect the Node to the parent
    parentNode->add_new_child_node(newNode, childIdx);
    return NODE_NEW_NODE;
}

void SearchThread::stop()
{
    isRunning = false;
}

Node *SearchThread::get_root_node() const
{
    return rootNode;
}

SearchLimits *SearchThread::get_search_limits() const
{
    return searchLimits;
}

void random_playout(NodeDescription& description, Node* currentNode, ChildIdx& childIdx)
{
    if (currentNode->get_real_visits() % int(pow(RANDOM_MOVE_COUNTER, description.depth + 1)) == 0 && currentNode->is_sorted()) {
        if (currentNode->is_fully_expanded()) {
            const size_t idx = rand() % currentNode->get_number_child_nodes();
            if (currentNode->get_child_node(idx) == nullptr || !currentNode->get_child_node(idx)->is_playout_node()) {
                childIdx = idx;
                return;
            }
            if (currentNode->get_child_node(idx)->get_node_type() != SOLVED_WIN) {
                childIdx = idx;
                return;
            }
        }
        else {
            childIdx = min(currentNode->get_no_visit_idx(), currentNode->get_number_child_nodes()-1);
            currentNode->increment_no_visit_idx();
        }
    }
}

Node* SearchThread::get_new_child_to_evaluate(ChildIdx& childIdx, NodeDescription& description)
{
    description.depth = 0;
    Node* currentNode = rootNode;

    while (true) {
        childIdx = uint16_t(-1);
        currentNode->lock();
        if (searchSettings->useRandomPlayout) {
            random_playout(description, currentNode, childIdx);
        }
        if (searchSettings->enhanceChecks) {
            childIdx = select_enhanced_move(currentNode);
        }

        if (childIdx == uint16_t(-1)) {
            childIdx = currentNode->select_child_node(searchSettings);
        }
        currentNode->apply_virtual_loss_to_child(childIdx, searchSettings->virtualLoss);
        trajectoryBuffer.emplace_back(NodeAndIdx(currentNode, childIdx));

        Node* nextNode = currentNode->get_child_node(childIdx);
        description.depth++;
        if (nextNode == nullptr) {
#ifdef MCTS_STORE_STATES
            StateObj* newState = currentNode->get_state()->clone();
#else
            newState = unique_ptr<StateObj>(rootState->clone());
            for (Action action : actionsBuffer) {
                newState->do_action(action);
            }
#endif
            const bool inCheck = newState->gives_check(currentNode->get_action(childIdx));
            newState->do_action(currentNode->get_action(childIdx));
            currentNode->increment_no_visit_idx();
#ifdef MCTS_STORE_STATES
            description.type = add_new_node_to_tree(newState, currentNode, childIdx, inCheck);
#else
            description.type = add_new_node_to_tree(newState.get(), currentNode, childIdx, inCheck);
#endif
            currentNode->unlock();

            if (description.type == NODE_NEW_NODE) {
                // fill a new board in the input_planes vector
                // we shift the index by NB_VALUES_TOTAL each time
                newState->get_state_planes(true, inputPlanes+newNodes->size()*StateConstants::NB_VALUES_TOTAL());
                // save a reference newly created list in the temporary list for node creation
                // it will later be updated with the evaluation of the NN
                newNodeSideToMove->add_element(newState->side_to_move());
            }

            return currentNode;
        }
        if (nextNode->is_terminal()) {
            description.type = NODE_TERMINAL;
            currentNode->unlock();
            return currentNode;
        }
        if (!nextNode->has_nn_results()) {
            description.type = NODE_COLLISION;
            currentNode->unlock();
            return currentNode;
        }
        if (nextNode->is_transposition()) {
            nextNode->lock();
            const uint_fast32_t transposVisits = currentNode->get_real_visits(childIdx);
            const double transposQValue = -currentNode->get_q_sum(childIdx, searchSettings->virtualLoss) / transposVisits;
            if (nextNode->is_transposition_return(transposQValue)) {
                const float qValue = get_transposition_q_value(transposVisits, transposQValue, nextNode->get_value());
                nextNode->unlock();
                description.type = NODE_TRANSPOSITION;
                transpositionValues->add_element(qValue);
                currentNode->unlock();
                return currentNode;
            }
            nextNode->unlock();
        }
        currentNode->unlock();
#ifndef MCTS_STORE_STATES
        actionsBuffer.emplace_back(currentNode->get_action(childIdx));
#endif
        currentNode = nextNode;
    }
}

void SearchThread::set_root_state(StateObj* value)
{
    rootState = value;
}

size_t SearchThread::get_tb_hits() const
{
    return tbHits;
}

void SearchThread::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
}

void fill_nn_results(size_t batchIdx, bool is_policy_map, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, is_policy_map), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, is_policy_map, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx);
    node->enable_has_nn_results();
}

void SearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_nn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, node, tbHits, newNodeSideToMove->get_element(batchIdx), searchSettings);
        }
        ++batchIdx;
        if (searchSettings->useTranspositionTable) {
            mapWithMutex->mtx.lock();
            mapWithMutex->hashTable.insert({node->hash_key(), node});
            mapWithMutex->mtx.unlock();
        }
    }
}

void SearchThread::backup_value_outputs()
{
    backup_values(newNodes.get(), newTrajectories);
    newNodeSideToMove->reset_idx();
    backup_values(transpositionValues.get(), transpositionTrajectories);
}

void SearchThread::backup_collisions() {
    for (size_t idx = 0; idx < collisionTrajectories.size(); ++idx) {
        backup_collision(searchSettings->virtualLoss, collisionTrajectories[idx]);
    }
    collisionTrajectories.clear();
}

bool SearchThread::nodes_limits_ok()
{
    return (searchLimits->nodes == 0 || (rootNode->get_nodes() < searchLimits->nodes)) &&
            (searchLimits->simulations == 0 || (rootNode->get_visits() < searchLimits->simulations));
}

bool SearchThread::is_root_node_unsolved()
{
    return rootNode->get_node_type() == UNSOLVED;
}

size_t SearchThread::get_avg_depth()
{
    return size_t(double(depthSum) / (rootNode->get_visits() - visitsPreSearch) + 0.5);
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batch
    Node *parentNode;
    NodeDescription description;
    ChildIdx childIdx;
    size_t numTerminalNodes = 0;

    while (!newNodes->is_full() &&
           collisionTrajectories.size() != searchSettings->batchSize &&
           !transpositionValues->is_full() &&
           numTerminalNodes < TERMINAL_NODE_CACHE) {

        trajectoryBuffer.clear();
        actionsBuffer.clear();
        parentNode = get_new_child_to_evaluate(childIdx, description);
        Node* newNode = parentNode->get_child_node(childIdx);
        depthSum += description.depth;
        depthMax = max(depthMax, description.depth);

        if(description.type == NODE_TERMINAL) {
            ++numTerminalNodes;
            backup_value(newNode->get_value(), searchSettings->virtualLoss, trajectoryBuffer);
        }
        else if (description.type == NODE_COLLISION) {
            // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
            collisionTrajectories.emplace_back(trajectoryBuffer);
        }
        else if (description.type == NODE_TRANSPOSITION) {
            transpositionTrajectories.emplace_back(trajectoryBuffer);
        }
        else {  // NODE_NEW_NODE
            newNodes->add_element(newNode);
            newTrajectories.emplace_back(trajectoryBuffer);
        }
    }
}

void SearchThread::thread_iteration()
{
    create_mini_batch();
    if (newNodes->size() != 0) {
        net->predict(inputPlanes, valueOutputs, probOutputs);
        set_nn_results_to_child_nodes();
    }
    backup_value_outputs();
    backup_collisions();
}

void run_search_thread(SearchThread *t)
{
    t->set_is_running(true);
    t->reset_stats();
    while(t->is_running() && t->nodes_limits_ok() && t->is_root_node_unsolved()) {
        t->thread_iteration();
    }
    t->set_is_running(false);
}

void SearchThread::backup_values(FixedVector<Node*>* nodes, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < nodes->size(); ++idx) {
        const Node* node = nodes->get_element(idx);
        backup_value(node->get_value(), searchSettings->virtualLoss, trajectories[idx]);
    }
    nodes->reset_idx();
    trajectories.clear();
}

void SearchThread::backup_values(FixedVector<float>* values, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < values->size(); ++idx) {
        backup_value(values->get_element(idx), searchSettings->virtualLoss, trajectories[idx]);
    }
    values->reset_idx();
    trajectories.clear();
}

ChildIdx SearchThread::select_enhanced_move(Node* currentNode) const {
    if (currentNode->get_real_visits() % CHECK_ENHANCE_COUNTER_PERIOD == 0 && currentNode->is_playout_node() && !currentNode->was_inspected() && !currentNode->is_terminal()) {

        // iterate over the current state
        unique_ptr<StateObj> pos = unique_ptr<StateObj>(rootState->clone());
        for (Action action : actionsBuffer) {
            pos->do_action(action);
        }

        // make sure a check has been explored at least once
        for (size_t childIdx = currentNode->get_no_visit_idx(); childIdx < currentNode->get_number_child_nodes(); ++childIdx) {
            if (pos->gives_check(currentNode->get_action(childIdx))) {
                for (size_t idx = currentNode->get_no_visit_idx(); idx < childIdx+1; ++idx) {
                    currentNode->increment_no_visit_idx();
                }
                return childIdx;
            }
        }
        // a full loop has been done
        currentNode->set_as_inspected();
    }
    return uint16_t(-1);
}

void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx)
{
    if (!node->is_tablebase()) {
        node->set_value(valueOutputs[batchIdx]);
    }
    else {
        ++tbHits;
        // TODO: Improvement the value assignment for table bases
        if (node->get_value() != 0) {
            // use the average of the TB entry and NN eval for non-draws
            node->set_value((valueOutputs[batchIdx] + node->get_value()) * 0.5f);
        }
    }
}

void node_post_process_policy(Node *node, float temperature, bool isPolicyMap, const SearchSettings* searchSettings)
{
    if (!isPolicyMap) {
        node->apply_softmax_to_policy();
    }
    node->enhance_moves(searchSettings);
    node->apply_temperature_to_prior_policy(temperature);
}

bool is_transposition_verified(const unordered_map<Key,Node*>::const_iterator& it, const StateObj* state) {
    return  it->second->has_nn_results() &&
            it->second->plies_from_null() == state->steps_from_null() &&
            state->number_repetitions() == 0;
}
