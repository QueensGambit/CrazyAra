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

SearchThread::SearchThread(NeuralNetAPI *netBatch, const SearchSettings* searchSettings, MapWithMutex* mapWithMutex):
    NeuralNetAPIUser(netBatch),
    rootNode(nullptr), rootState(nullptr),  // will be be set via setter methods
    newNodes(make_unique<FixedVector<Node*>>(searchSettings->batchSize)),
    newNodeSideToMove(make_unique<FixedVector<SideToMove>>(searchSettings->batchSize)),
    transpositionValues(make_unique<FixedVector<float>>(searchSettings->batchSize*2)),
    numTerminalNodes(0),
    isRunning(true), mapWithMutex(mapWithMutex), searchSettings(searchSettings),
    tbHits(0), depthSum(0), depthMax(0), visitsPreSearch(0),
    #ifdef MCTS_SINGLE_PLAYER
    terminalNodeCache(1),
    #else
    terminalNodeCache(searchSettings->batchSize*2),
    #endif
    reachedTablebases(false)
{
    searchLimits = nullptr;  // will be set by set_search_limits() every time before go()
    trajectoryBuffer.reserve(DEPTH_INIT);
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

void SearchThread::set_reached_tablebases(bool value)
{
    reachedTablebases = value;
}

Node* SearchThread::add_new_node_to_tree(StateObj* newState, Node* parentNode, ChildIdx childIdx, NodeBackup& nodeBackup)
{
    bool transposition;
    Node* newNode = parentNode->add_new_node_to_tree(mapWithMutex, newState, childIdx, searchSettings, transposition);
    if (transposition) {
        const float qValue =  parentNode->get_child_node(childIdx)->get_value();
        transpositionValues->add_element(qValue);
        nodeBackup = NODE_TRANSPOSITION;
        return newNode;
    }
    nodeBackup = NODE_NEW_NODE;
    return newNode;
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

void random_playout(Node* currentNode, ChildIdx& childIdx)
{
    if (currentNode->is_fully_expanded()) {
        const size_t idx = rand() % currentNode->get_number_child_nodes();
        if (currentNode->get_child_node(idx) == nullptr || !currentNode->get_child_node(idx)->is_playout_node()) {
            childIdx = idx;
            return;
        }
        if (currentNode->get_child_node(idx)->get_node_type() == UNSOLVED) {
            childIdx = idx;
            return;
        }
        childIdx = uint16_t(-1);
    }
    else {
        childIdx = min(size_t(currentNode->get_no_visit_idx()), currentNode->get_number_child_nodes()-1);
        currentNode->increment_no_visit_idx();
        return;
    }
}

Node* SearchThread::get_starting_node(Node* currentNode, StateObj* currentState, NodeDescription& description, ChildIdx& childIdx)
{
    size_t depth = get_random_depth();
    for (uint curDepth = 0; curDepth < depth; ++curDepth) {
        currentNode->lock();
        childIdx = get_best_action_index(currentNode, true, 0, 0);
        Node* nextNode = currentNode->get_child_node(childIdx);
        if (nextNode == nullptr || !nextNode->is_playout_node() || nextNode->get_visits() < searchSettings->epsilonGreedyCounter || nextNode->get_node_type() != UNSOLVED) {
            currentNode->unlock();
            break;
        }
        currentNode->unlock();
        currentState->do_action(currentNode->get_action(childIdx));
        currentNode = nextNode;
        ++description.depth;
    }
    return currentNode;
}

Node* SearchThread::handle_single_split(size_t mainIdx, ChildIdx childIdx, Budget budget, NodeDescription& description)
{
    Node* currentNode = entryNodes[mainIdx].node;
    assert(currentNode != nullptr);
    StateObj* curState = entryNodes[mainIdx].curState;
    assert(curState != nullptr);

    currentNode->apply_virtual_loss_to_child(childIdx, budget);
    Node* nextNode = currentNode->get_child_node(childIdx);
    stateStore.emplace_back(unique_ptr<StateObj>(curState->clone()));
    StateObj* newState = stateStore.back().get();
    entryNodes.emplace_back(NodeAndBudget(nextNode, budget, stateStore.back().get()));
    entryNodes.back().curTrajectory = entryNodes[mainIdx].curTrajectory;
    entryNodes.back().curTrajectory.emplace_back(NodeAndIdx(currentNode, childIdx));

    assert(entryNodes.back().curTrajectory.front().node == rootNode);
    Node* returnNode = check_next_node(currentNode, newState, nextNode, childIdx, description);

    if (returnNode != nullptr) {
        currentNode->unlock();
        return returnNode;
    }

    // extend trajectory
    assert(nextNode != nullptr);
    newState->do_action(currentNode->get_action(childIdx));

    return nullptr;
}

void SearchThread::single_split(size_t mainIdx, ChildIdx childIdx, Budget budget, NodeDescription& description)
{
    assert(budget > 0);
    Node* returnNode = handle_single_split(mainIdx, childIdx, budget, description);

    if (returnNode != nullptr) {
        handle_simulation_return(returnNode, description.type, entryNodes.back().curTrajectory);

        assert(entryNodes.back().curTrajectory.size() > 0);
        // issue "budget-1" collision trajectories
        for (Budget idx = 0; idx < budget-1; ++idx) {
            collisionTrajectories.emplace_back(entryNodes.back().curTrajectory);
        }
        entryNodes.pop_back();
    }
}

void SearchThread::distribute_mini_batch_across_nodes()
{
    // initialize node budget
    entryNodes.emplace_back(NodeAndBudget(rootNode, net->get_batch_size(), rootState->clone()));
    NodeDescription description;

    while(!entryNodes.empty()) {
        size_t mainIdx = entryNodes.size() - 1;

        if (entryNodes[mainIdx].budget == 1) {
            // extend single trajectory as used to
            Node* newNode = get_new_child_to_evaluate(description, entryNodes[mainIdx].node, entryNodes[mainIdx].curState, entryNodes[mainIdx].curTrajectory);
            handle_simulation_return(newNode, description.type, entryNodes[mainIdx].curTrajectory);
        }
        else {
            // branch and split
            NodeSplit nodeSplit = entryNodes[mainIdx].node->select_child_nodes(searchSettings, entryNodes[mainIdx].budget);

            assert(entryNodes[mainIdx].node != nullptr);
            if (nodeSplit.secondBudget > 0) {
                // 2nd branch
                assert(nodeSplit.secondArg != nodeSplit.firstArg);
                single_split(mainIdx, nodeSplit.secondArg, nodeSplit.secondBudget, description);
            }

            // 1st branch
            single_split(mainIdx, nodeSplit.firstArg, nodeSplit.firstBudget, description);
        }

        // delete old main branch
        entryNodes.erase(entryNodes.begin()+mainIdx);
    }
}

NodeBackup SearchThread::handle_returns(Node* currentNode, Node* nextNode, ChildIdx childIdx) {

#ifdef MCTS_TB_SUPPORT
    if (nextNode->is_terminal() || (!reachedTablebases && nextNode->is_playout_node() && nextNode->is_solved())) {
#else
    if (nextNode->is_terminal() || (nextNode->is_playout_node() && nextNode->is_solved())) {
#endif
        return NODE_TERMINAL;
    }
    if (!nextNode->has_nn_results()) {
        return NODE_COLLISION;
    }
    if (nextNode->is_transposition()) {
        nextNode->lock();
        const uint_fast32_t transposVisits = currentNode->get_real_visits(childIdx);
        const double transposQValue = -currentNode->get_q_sum(childIdx, searchSettings->virtualLoss) / transposVisits;
        if (nextNode->is_transposition_return(transposQValue)) {
            const float qValue = get_transposition_q_value(transposVisits, transposQValue, nextNode->get_value());
            nextNode->unlock();
            transpositionValues->add_element(qValue);
            return NODE_TRANSPOSITION;
        }
        nextNode->unlock();
    }
    return NODE_UNKNOWN;
}

Node* SearchThread::create_new_node(Node* currentNode, StateObj* currentState, ChildIdx childIdx, NodeDescription& description) {
#ifdef MCTS_STORE_STATES
    StateObj* currentState = currentNode->get_state()->clone();
#endif
    currentState->do_action(currentNode->get_action(childIdx));
    currentNode->increment_no_visit_idx();
    Node* nextNode = add_new_node_to_tree(currentState, currentNode, childIdx, description.type);
    currentNode->unlock();

    if (description.type == NODE_NEW_NODE) {
#ifdef SEARCH_UCT
        Node* nextNode = currentNode->get_child_node(childIdx);
        nextNode->set_value(newState->random_rollout());
        nextNode->enable_has_nn_results();
        if (searchSettings->useTranspositionTable && !nextNode->is_terminal()) {
            mapWithMutex->mtx.lock();
            mapWithMutex->hashTable.insert({nextNode->hash_key(), nextNode});
            mapWithMutex->mtx.unlock();
        }
#else
        // fill a new board in the input_planes vector
        // we shift the index by nbNNInputValues each time
        currentState->get_state_planes(true, inputPlanes + newNodes->size() * net->get_nb_input_values_total(), net->get_version());
        // save a reference newly created list in the temporary list for node creation
        // it will later be updated with the evaluation of the NN
        newNodeSideToMove->add_element(currentState->side_to_move());
#endif
    }
    return nextNode;
}

Node* SearchThread::check_next_node(Node* currentNode, StateObj* currentState, Node* nextNode, ChildIdx childIdx, NodeDescription& description)
{
    if (nextNode == nullptr) {
        nextNode = create_new_node(currentNode, currentState, childIdx, description);
        return nextNode;
    }

    description.type = handle_returns(currentNode, nextNode, childIdx);
    if (description.type != NODE_UNKNOWN) {
        currentNode->unlock();
        return nextNode;
    }
    return nullptr;
}

Node* SearchThread::init_child_index(Node* currentNode, StateObj* currentState, NodeDescription& description, ChildIdx& childIdx)
{
    childIdx = uint16_t(-1);

//    if (searchSettings->epsilonGreedyCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonGreedyCounter == 0) {
//        currentNode = get_starting_node(currentNode, currentState, description, childIdx);
//        currentNode->lock();
//        random_playout(currentNode, childIdx);
//        currentNode->unlock();
//        return currentNode;
//    }
//    if (searchSettings->epsilonChecksCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonChecksCounter == 0) {
//        currentNode = get_starting_node(currentNode, currentState, description, childIdx);
//        currentNode->lock();
//        childIdx = select_enhanced_move(currentNode, currentState);
//        if (childIdx ==  uint16_t(-1)) {
//            random_playout(currentNode, childIdx);
//        }
//        currentNode->unlock();
//        return currentNode;
//    }
    return currentNode;
}


Node* SearchThread::get_new_child_to_evaluate(NodeDescription& description, Node* currentNode, StateObj* currentState, Trajectory& trajectoryBuffer)
{
    description.depth = 0;
    Node* nextNode;
    ChildIdx childIdx = uint16_t(-1);
//    unique_ptr<StateObj> currentState = unique_ptr<StateObj>(startingState->clone());
//    currentNode = init_child_index(currentNode, currentState.get(), description, childIdx);

    while (true) {
        currentNode->lock();
        if (childIdx == uint16_t(-1)) {
            childIdx = currentNode->select_child_node(searchSettings);
        }
        currentNode->apply_virtual_loss_to_child(childIdx, searchSettings->virtualLoss);
        trajectoryBuffer.emplace_back(NodeAndIdx(currentNode, childIdx));

        nextNode = currentNode->get_child_node(childIdx);
        description.depth++;

        Node* returnNode = check_next_node(currentNode, currentState, nextNode, childIdx, description);

        if (returnNode != nullptr) {
            return returnNode;
        }

        currentNode->unlock();
        currentState->do_action(currentNode->get_action(childIdx));
        currentNode = nextNode;
        childIdx = uint16_t(-1);
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

void fill_nn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, const float* auxiliaryOutputs, Node *node, size_t& tbHits, bool mirrorPolicy, const SearchSettings* searchSettings, bool isRootNodeTB)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), mirrorPolicy);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx, isRootNodeTB);
#ifdef MCTS_STORE_STATES
    node->set_auxiliary_outputs(get_auxiliary_data_batch(batchIdx, auxiliaryOutputs));
#endif
    node->enable_has_nn_results();
}

void SearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_nn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, auxiliaryOutputs, node,
                            tbHits, rootState->mirror_policy(newNodeSideToMove->get_element(batchIdx)),
                            searchSettings, rootNode->is_tablebase());
        }
        ++batchIdx;
    }
}

void SearchThread::backup_value_outputs()
{
    backup_values(*newNodes, newTrajectories);
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
    return (searchLimits->nodes == 0 || (rootNode->get_node_count() < searchLimits->nodes)) &&
            (searchLimits->simulations == 0 || (rootNode->get_visits() < searchLimits->simulations)) &&
            (searchLimits->nodesLimit == 0 || (rootNode->get_node_count() < searchLimits->nodesLimit));
}

bool SearchThread::is_root_node_unsolved()
{
#ifdef MCTS_TB_SUPPORT
    return is_unsolved_or_tablebase(rootNode->get_node_type());
#else
    return rootNode->get_node_type() == UNSOLVED;
#endif
}

size_t SearchThread::get_avg_depth()
{
    return size_t(double(depthSum) / (rootNode->get_visits() - visitsPreSearch) + 0.5);
}

void SearchThread::handle_simulation_return(Node* newNode, NodeBackup nodeBackup, const Trajectory& trajectoryBuffer)
{
    assert(trajectoryBuffer.size() != 0);
    switch (nodeBackup) {
    case NODE_TERMINAL:
        ++numTerminalNodes;
        backup_value<true>(newNode->get_value(), searchSettings->virtualLoss, trajectoryBuffer, searchSettings->mctsSolver);
        break;
    case NODE_COLLISION:
        // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
        collisionTrajectories.emplace_back(trajectoryBuffer);
        break;
    case NODE_TRANSPOSITION:
        transpositionTrajectories.emplace_back(trajectoryBuffer);
        break;
    case NODE_NEW_NODE:
        newNodes->add_element(newNode);
        newTrajectories.emplace_back(trajectoryBuffer);
        break;
    case NODE_UNKNOWN:
        throw "NODE_UNKNOWN is invalid as return type";
    }
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batch
    NodeDescription description;
    numTerminalNodes = 0;

    while (!newNodes->is_full() &&
           collisionTrajectories.size() != searchSettings->batchSize &&
           !transpositionValues->is_full() &&
           numTerminalNodes < terminalNodeCache) {

        trajectoryBuffer.clear();
        Node* newNode = get_new_child_to_evaluate(description, rootNode, rootState, trajectoryBuffer);
        depthSum += description.depth;
        depthMax = max(depthMax, description.depth);

        handle_simulation_return(newNode, description.type, trajectoryBuffer);
    }
}

void SearchThread::thread_iteration()
{
//    create_mini_batch();
    distribute_mini_batch_across_nodes();

#ifndef SEARCH_UCT
    if (newNodes->size() != 0) {
        net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
        set_nn_results_to_child_nodes();
    }
#endif
    backup_value_outputs();
    backup_collisions();
    stateStore.clear();
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

void SearchThread::backup_values(FixedVector<Node*>& nodes, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        Node* node = nodes.get_element(idx);
#ifdef MCTS_TB_SUPPORT
        const bool solveForTerminal = searchSettings->mctsSolver && node->is_tablebase();
        backup_value<false>(node->get_value(), searchSettings->virtualLoss, trajectories[idx], solveForTerminal);
#else
        backup_value<false>(node->get_value(), searchSettings->virtualLoss, trajectories[idx], false);
#endif
    }
    nodes.reset_idx();
    trajectories.clear();
}

void SearchThread::backup_values(FixedVector<float>* values, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < values->size(); ++idx) {
        const float value = values->get_element(idx);
        backup_value<true>(value, searchSettings->virtualLoss, trajectories[idx], false);
    }
    values->reset_idx();
    trajectories.clear();
}

ChildIdx SearchThread::select_enhanced_move(Node* currentNode, StateObj* currentState) const {
    if (currentNode->is_playout_node() && !currentNode->was_inspected() && !currentNode->is_terminal()) {

        // iterate over the current state
        unique_ptr<StateObj> pos = unique_ptr<StateObj>(currentState->clone());

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

void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx, bool isRootNodeTB)
{
#ifdef MCTS_TB_SUPPORT
    if (node->is_tablebase()) {
        ++tbHits;
        // TODO: Improvement the value assignment for table bases
        if (node->get_value() != 0 && isRootNodeTB) {
            // use the average of the TB entry and NN eval for non-draws
            node->set_value((valueOutputs[batchIdx] + node->get_value()) * 0.5f);
        }
        return;
    }
#endif
    node->set_value(valueOutputs[batchIdx]);
}

void node_post_process_policy(Node *node, float temperature, const SearchSettings* searchSettings)
{
    node->enhance_moves(searchSettings);
    node->apply_temperature_to_prior_policy(temperature);
}

size_t get_random_depth()
{
    const int randInt = rand() % 100 + 1;
    return std::ceil(-std::log2(1 - randInt / 100.0) - 1);
}
