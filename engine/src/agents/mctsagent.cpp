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
 * @file: mctsagent.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include <thread>
#include "mctsagent.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"
#include "util/gcthread.h"


MCTSAgent::MCTSAgent(NeuralNetAPI *netSingle, vector<unique_ptr<NeuralNetAPI>>& netBatches,
                     SearchSettings* searchSettings, PlaySettings* playSettings):
    Agent(playSettings, true),
    netSingle(netSingle),
    searchSettings(searchSettings),
    rootNode(nullptr),
    rootState(nullptr),
    oldestRootNode(nullptr),
    ownNextRoot(nullptr),
    opponentsNextRoot(nullptr),
    lastValueEval(-1.0f),
    reusedFullTree(false),
    isRunning(false),
    overallNPS(0.0f),
    nbNPSentries(0),
    threadManager(nullptr),
    gcThread()
{
    mapWithMutex.hashTable.reserve(1e6);

    for (auto i = 0; i < searchSettings->threads; ++i) {
        searchThreads.emplace_back(new SearchThread(netBatches[i].get(), searchSettings, &mapWithMutex, &cells));
    }
    probOutputs = make_unique<float[]>(netSingle->get_policy_output_length());
    timeManager = make_unique<TimeManager>(searchSettings->randomMoveFactor);
    generator = default_random_engine(r());
    fill(inputPlanes, inputPlanes+NB_VALUES_TOTAL, 0.0f);  // will be filled in evalute_board_state()
}

MCTSAgent::~MCTSAgent()
{
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

Node* MCTSAgent::get_opponents_next_root() const
{
    return opponentsNextRoot;
}

Node* MCTSAgent::get_root_node() const
{
    return rootNode;
}

string MCTSAgent::get_device_name() const
{
    return netSingle->get_device_name();
}

float MCTSAgent::get_dirichlet_noise() const
{
    return searchSettings->dirichletEpsilon;
}

float MCTSAgent::get_q_value_weight() const
{
    return searchSettings->qValueWeight;
}

void MCTSAgent::update_q_value_weight(float value)
{
    searchSettings->qValueWeight = value;
}

void MCTSAgent::update_dirichlet_epsilon(float value)
{
    searchSettings->dirichletEpsilon = value;
}

StateObj *MCTSAgent::get_root_state() const
{
    return rootState;
}

bool MCTSAgent::is_running() const
{
    return isRunning;
}

size_t MCTSAgent::init_root_node(StateObj *state)
{
    size_t nodesPreSearch;
    rootNode = get_root_node_from_tree(state);

    if (rootNode != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        nodesPreSearch = size_t(rootNode->get_visits());
        if (rootNode->is_playout_node()) {
            nodesPreSearch -= rootNode->get_terminal_visits();
        }
        info_string(nodesPreSearch, "nodes of former tree will be reused");
    }
    else {
        create_new_root_node(state);
        nodesPreSearch = 0;
    }
    return nodesPreSearch;
}

Node *MCTSAgent::get_root_node_from_tree(StateObj *state)
{
    reusedFullTree = false;

    if (rootNode == nullptr) {
        return nullptr;
    }
    if (same_hash_key(rootNode, state)) {
        info_string("reuse the full tree");
        reusedFullTree = true;
        return rootNode;
    }

    if (same_hash_key(ownNextRoot, state) && ownNextRoot->is_playout_node()) {
        delete_sibling_subtrees(ownNextRoot, mapWithMutex.hashTable, gcThread);
        delete_sibling_subtrees(opponentsNextRoot, mapWithMutex.hashTable, gcThread);
        if (rootNode->get_parent_node() != nullptr) {
            gcThread.add_item_to_delete(rootNode->get_parent_node());
        }
        gcThread.add_item_to_delete(rootNode);
        return ownNextRoot;
    }
    if (same_hash_key(opponentsNextRoot, state) && opponentsNextRoot->is_playout_node()) {
        delete_sibling_subtrees(opponentsNextRoot, mapWithMutex.hashTable, gcThread);
        if (opponentsNextRoot->get_parent_node() != nullptr) {
            gcThread.add_item_to_delete(opponentsNextRoot->get_parent_node());
        }
        return opponentsNextRoot;
    }

    // the node wasn't found, clear the old tree
    delete_old_tree();
    if (rootNode->get_parent_node() != nullptr) {
        gcThread.add_item_to_delete(rootNode->get_parent_node());
    }
    gcThread.add_item_to_delete(rootNode);

    return nullptr;
}

void MCTSAgent::create_new_root_node(StateObj* state)
{
    info_string("create new tree");
    // TODO: Make sure that "inCheck=False" does not cause issues
    Node* dummyNode = new Node(state, false, nullptr, 0, searchSettings);
    dummyNode->init_node_data(1);
    rootNode = new Node(state, false, dummyNode, 0, searchSettings);
    dummyNode->add_new_child_node(rootNode, 0);
    oldestRootNode = rootNode;
    state->get_state_planes(true, begin(inputPlanes));
    netSingle->predict(inputPlanes, &valueOutput, probOutputs.get());
    size_t tbHits = 0;
    fill_nn_results(0, netSingle->is_policy_map(), &valueOutput, probOutputs.get(), rootNode, tbHits, state->side_to_move(), searchSettings);
    rootNode->prepare_node_for_visits();
}

void MCTSAgent::delete_old_tree()
{
    // clear all remaining node of the former root node
    if (rootNode != nullptr) {
        for (Node* childNode: rootNode->get_child_nodes()) {
            delete_subtree_and_hash_entries(childNode, mapWithMutex.hashTable, gcThread);
        }
    }
}

void MCTSAgent::sleep_and_log_for(size_t timeMS, size_t updateIntervalMS)
{
    if (!isRunning) {
        return;
    }
    for (size_t var = 0; var < timeMS / updateIntervalMS && isRunning; ++var) {
        this_thread::sleep_for(chrono::milliseconds(updateIntervalMS));
        evalInfo->end = chrono::steady_clock::now();
        info_msg(evalInfo);
        if (!searchThreads[0]->is_running()) {
            isRunning = false;
            return;
        }
    }
    this_thread::sleep_for(chrono::milliseconds(timeMS % 1000));
}

void MCTSAgent::update_nps_measurement(float curNPS)
{
    if (searchSettings->useNPSTimemanager) {
        ++nbNPSentries;
        overallNPS += 1/nbNPSentries * (curNPS - overallNPS);
    }
}

void MCTSAgent::apply_move_to_tree(Action move, bool ownMove)
{
    if (!reusedFullTree && rootNode != nullptr && rootNode->is_playout_node()) {
        if (ownMove) {
            info_string("apply move to tree");
            opponentsNextRoot = pick_next_node(move, rootNode);
        }
        else if (opponentsNextRoot != nullptr && opponentsNextRoot->is_playout_node()){
            info_string("apply move to tree");
            ownNextRoot = pick_next_node(move, opponentsNextRoot);
        }
    }
}

void MCTSAgent::clear_game_history()
{
    delete_old_tree();
    assert(mapWithMutex.hashTable.size() == 0);
    mapWithMutex.hashTable.clear();
    oldestRootNode = nullptr;
    ownNextRoot = nullptr;
    opponentsNextRoot = nullptr;
    rootNode = nullptr;
    lastValueEval = -1.0f;
    nbNPSentries = 0;
    overallNPS = 0;
}

bool MCTSAgent::is_policy_map()
{
    return netSingle->is_policy_map();
}

string MCTSAgent::get_name() const
{
    return engineName + "-" + engineVersion + "-" + netSingle->get_model_name();
}

void MCTSAgent::update_stats()
{
    avgDepth = get_avg_depth(searchThreads);
    maxDepth = get_max_depth(searchThreads);
    tbHits = get_tb_hits(searchThreads);
}

void MCTSAgent::evaluate_board_state()
{
    evalInfo->nodesPreSearch = init_root_node(state);
    thread tGCThread = thread(run_gc_thread<Node>, &gcThread);
    evalInfo->isChess960 = state->is_chess960();
    rootState = state;
    if (rootNode->get_number_child_nodes() == 1 && !rootNode->is_blank_root_node()) {
        info_string("Only single move available -> early stopping");
    }
    else if (rootNode->get_number_child_nodes() == 0) {
        info_string("The given position has no legal moves");
    }
    else {
        if (searchSettings->dirichletEpsilon > 0.009f) {
            info_string("apply dirichlet noise");
            // TODO: Check for dirichlet compability
            rootNode->apply_dirichlet_noise_to_prior_policy(searchSettings);
            rootNode->fully_expand_node();
        }

        if (!rootNode->is_root_node()) {
            rootNode->make_to_root();
        }
        info_string("run mcts search");
        run_mcts_search();
        update_stats();
    }
    update_eval_info(*evalInfo, rootNode, tbHits, maxDepth, searchSettings->multiPV);
    lastValueEval = evalInfo->bestMoveQ[0];
    update_nps_measurement(evalInfo->calculate_nps());
    tGCThread.join();
}

void MCTSAgent::run_mcts_search()
{
    thread** threads = new thread*[searchSettings->threads];
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        searchThreads[i]->set_root_node(rootNode);
        searchThreads[i]->set_root_state(rootState);
        searchThreads[i]->set_search_limits(searchLimits);
        threads[i] = new thread(run_search_thread, searchThreads[i]);
    }
    int curMovetime = timeManager->get_time_for_move(searchLimits, rootState->side_to_move(), rootNode->plies_from_null()/2);
    threadManager = make_unique<ThreadManager>(rootNode, evalInfo, searchThreads, curMovetime, 250, searchSettings->multiPV, overallNPS, lastValueEval,
                                               is_game_sceneario(searchLimits),
                                               can_prolong_search(rootNode->plies_from_null()/2, timeManager->get_thresh_move()));
    unique_ptr<thread> tManager = make_unique<thread>(run_thread_manager, threadManager.get());
    isRunning = true;

    for (size_t i = 0; i < searchSettings->threads; ++i) {
        threads[i]->join();
    }
    threadManager->kill();
    tManager->join();
    delete[] threads;
    isRunning = false;
}

void MCTSAgent::stop()
{
    isRunning = false;
    if (threadManager != nullptr) {
        threadManager->stop_search();
    }
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        info_string("You must do a search before you can print the root node statistics");
        return;
    }
    rootNode->print_node_statistics(rootState);
}
