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
#include <fstream>
#include "mctsagent.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"

MCTSAgent::MCTSAgent(const vector<unique_ptr<NeuralNetAPI>>& netSingleVector, const vector<vector<unique_ptr<NeuralNetAPI>>>& netBatchesVector,
                     SearchSettings* searchSettings, PlaySettings* playSettings):
    Agent(netSingleVector, playSettings, true),
    searchSettings(searchSettings),
    rootNode(nullptr),
    rootState(nullptr),
    ownNextRoot(nullptr),
    opponentsNextRoot(nullptr),
    lastValueEval(-1.0f),
    reusedFullTree(false),
    overallNPS(0.0f),
    nbNPSentries(0),
    threadManager(nullptr),
    reachedTablebases(false)
{
    mapWithMutex.hashTable.reserve(1e6);
    inferenceQueue = std::make_shared<InferenceQueue>();

    for (size_t idx = 0; idx < searchSettings->threads; ++idx) {
        shared_ptr<NeuralNetAPIUser> nnUserThread = make_shared<NeuralNetAPIUser>(netBatchesVector[idx]);
        batchBarriers.emplace_back(make_shared<ReusableBarrier>(searchSettings->numberParallelGames));
        searchThreads.emplace_back(new SearchThread(agentID, nnUserThread, searchSettings, &mapWithMutex, batchBarriers[idx].get(), inferenceQueue.get()));
    }
    timeManager = make_unique<TimeManager>(searchSettings->randomMoveFactor);
    generator = default_random_engine(r());

    auto nnUser = make_shared<NeuralNetAPIUser>(netBatchesVector[0]);
    size_t maxBatchSize = searchSettings->numberParallelGames; // or tuned
    size_t policySize;
    if (nnUser->nets.front()->is_policy_map()) {
        policySize = StateConstants::NB_LABELS_POLICY_MAP();
    }
    else {
        policySize = StateConstants::NB_LABELS();
    }
    size_t auxSize = StateConstants::NB_AUXILIARY_OUTPUTS();
    inferenceWorker = std::make_unique<InferenceWorker>(inferenceQueue, nnUser, maxBatchSize, policySize, auxSize);
    inferenceWorker->start();
}

MCTSAgent::~MCTSAgent()
{
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

MCTSAgent::MCTSAgent(const MCTSAgent& other):
    Agent(other)
{
    this->mapWithMutex.hashTable.reserve(1e6);
    this->searchSettings = other.searchSettings;
    this->playSettings = other.playSettings;
    this->batchBarriers = other.batchBarriers;
    this->agentID = other.agentID;
    this->inferenceQueue = other.inferenceQueue;
    for (size_t idx = 0; idx < searchSettings->threads; ++idx) {
        this->searchThreads.emplace_back(new SearchThread(agentID, other.searchThreads[idx]->get_nn_user(), searchSettings, &mapWithMutex, other.batchBarriers[idx].get(), other.inferenceQueue.get()));
    }
    timeManager = make_unique<TimeManager>(searchSettings->randomMoveFactor);
}

Node* MCTSAgent::get_opponents_next_root() const
{
    return opponentsNextRoot.get();
}

Node* MCTSAgent::get_root_node() const
{
    return rootNode.get();
}

string MCTSAgent::get_device_name() const
{
    return nnUser->nets.front()->get_device_name();
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
    return rootState.get();
}

bool MCTSAgent::is_running() const
{
    return isRunning;
}

size_t MCTSAgent::init_root_node(StateObj *state)
{
    size_t nodesPreSearch;
    gcThread.oldRootNode = rootNode;
    rootNode = get_root_node_from_tree(state);

    if (rootNode != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        nodesPreSearch = size_t(rootNode->get_visits());
        if (rootNode->is_playout_node()) {
            nodesPreSearch -= rootNode->get_free_visits();
        }
        info_string(nodesPreSearch, "nodes of former tree will be reused");
    }
    else {
        create_new_root_node(state);
        nodesPreSearch = 0;
    }
    reachedTablebases = rootNode->is_tablebase() || reachedTablebases;
    return nodesPreSearch;
}

shared_ptr<Node> MCTSAgent::get_root_node_from_tree(StateObj *state)
{
    reusedFullTree = false;

    if (rootNode == nullptr) {
        return nullptr;
    }
    if (!searchSettings->reuseTree) {
        delete_old_tree();
        return nullptr;
    }

    if (same_hash_key(rootNode.get(), state)) {
        info_string("reuse the full tree");
        reusedFullTree = true;
        return rootNode;
    }

    if (same_hash_key(ownNextRoot.get(), state) && ownNextRoot->is_playout_node() && ownNextRoot->get_number_of_nodes() > 0) {
        return ownNextRoot;
    }
    if (same_hash_key(opponentsNextRoot.get(), state) && opponentsNextRoot->is_playout_node() && opponentsNextRoot->get_number_of_nodes() > 0) {
        return opponentsNextRoot;
    }
    // the node wasn't found, clear the old tree
    delete_old_tree();

    return nullptr;
}

void MCTSAgent::set_root_node_predictions()
{
    if (searchSettings->numberParallelGames == 1) {
        state->get_state_planes(true, nnUser->inputPlanes, nnUser->nets.front()->get_version());
        size_t netIdx = 0;
        if (nnUser->nets.size() > 1) {
            GamePhase currentPhase = state->get_phase(nnUser->numPhases, searchSettings->gamePhaseDefinition);
            netIdx = nnUser->phaseToNetsIndex.at(currentPhase);
        }

        nnUser->nets[netIdx]->predict(nnUser->inputPlanes, nnUser->valueOutputs, nnUser->probOutputs, nnUser->auxiliaryOutputs);
        size_t tbHits = 0;
        fill_nn_results(0, nnUser->nets[netIdx]->is_policy_map(), nnUser->valueOutputs, nnUser->probOutputs, nnUser->auxiliaryOutputs, rootNode.get(), tbHits,
                        rootState->mirror_policy(state->side_to_move()), searchSettings, rootNode->is_tablebase());
    }
    else {
        InferenceRequest request;

        // --- basic metadata ---
        request.inputSize  = StateConstants::NB_VALUES_TOTAL();
        request.batchCount = 1;
        request.agentID    = agentID;

        // --- OWNED input buffer ---
        const size_t localBatchSize = searchSettings->batchSize;
        const size_t elems = request.inputSize;

        request.inputData.resize(elems);

        state->get_state_planes(true, request.inputData.data(), nnUser->nets.front()->get_version());

        // --- enqueue & wait synchronously ---
        auto future = request.promise.get_future();
        inferenceQueue->push(std::move(request));

        InferenceResult result = future.get();
        size_t tbHits = 0;
        fill_nn_results(0, nnUser->nets[0]->is_policy_map(), result.valueOutputs.data(), result.probOutputs.data(), result.auxiliaryOutputs.data(), rootNode.get(), tbHits,
                        rootState->mirror_policy(state->side_to_move()), searchSettings, rootNode->is_tablebase());
    }

}

void MCTSAgent::create_new_root_node(StateObj* state)
{
    info_string("create new tree");
#ifdef MCTS_STORE_STATES
    rootNode = make_shared<Node>(state->clone(), searchSettings);
#else
    rootNode = make_shared<Node>(state, searchSettings);
#endif
#ifdef SEARCH_UCT
    unique_ptr<StateObj> newState = unique_ptr<StateObj>(state->clone());
    rootNode->set_value(newState->random_rollout());
    rootNode->enable_has_nn_results();
#else
    set_root_node_predictions();
#endif
    rootNode->prepare_node_for_visits();
}

void MCTSAgent::delete_old_tree()
{
    // clear all remaining node of the former root node
    mapWithMutex.hashTable.clear();
    assert(mapWithMutex.hashTable.size() == 0);
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
        overallNPS += 1.0f/nbNPSentries * (curNPS - overallNPS);
    }
}

unsigned int MCTSAgent::get_num_phases()
{
    return nnUser->get_num_phases();
}

size_t MCTSAgent::get_agent_id()
{
    return agentID;
}

void MCTSAgent::set_agent_id(size_t value)
{
    agentID = value;
    for (auto searchThread : searchThreads) {
        searchThread->set_agent_id(value);
    }
}

void MCTSAgent::apply_move_to_tree(Action move, bool ownMove)
{
    if (!reusedFullTree && rootNode != nullptr && rootNode->is_playout_node()) {
        if (ownMove) {
            info_string("apply move to tree");
            opponentsNextRoot = pick_next_node(move, rootNode.get());
            return;
        }
        else if (opponentsNextRoot != nullptr && opponentsNextRoot->is_playout_node()){
            info_string("apply move to tree");
            ownNextRoot = pick_next_node(move, opponentsNextRoot.get());
            return;
        }
    }
    // the full tree will be deleted next search
    opponentsNextRoot = nullptr;
    ownNextRoot = nullptr;
}

void MCTSAgent::clear_game_history()
{
    delete_old_tree();
    ownNextRoot = nullptr;
    opponentsNextRoot = nullptr;
    rootNode = nullptr;
    lastValueEval = -1.0f;
    nbNPSentries = 0;
    overallNPS = 0;
    reachedTablebases = false;
}

bool MCTSAgent::is_policy_map()
{
    return nnUser->nets.front()->is_policy_map();
}

string MCTSAgent::get_name() const
{
    return engineName + "-" + engineVersion + "-" + nnUser->nets.front()->get_model_name() + "-" + to_string(agentID);
}

void MCTSAgent::update_stats()
{
    avgDepth = get_avg_depth(searchThreads);
    maxDepth = get_max_depth(searchThreads);
    tbHits = get_tb_hits(searchThreads);
}

void MCTSAgent::handle_single_move()
{
    float targetEval = lastValueEval;
    switch(searchSettings->searchPlayerMode) {
    case MODE_TWO_PLAYER:
        if (lastSideToMove != state->side_to_move()) {
            targetEval = -lastValueEval;
        }
    case MODE_SINGLE_PLAYER: ;
    }
    rootNode->set_value(targetEval);
    rootNode->set_q_value(0, targetEval);
}

void MCTSAgent::evaluate_board_state()
{
    rootState = unique_ptr<StateObj>(state->clone());
    evalInfo->nodesPreSearch = init_root_node(state);
    thread tGCThread = thread(run_gc_thread, &gcThread);
#ifdef USE_RL
    tGCThread.join();
#endif
    evalInfo->isChess960 = state->is_chess960();
    if (rootNode->get_number_child_nodes() == 1) {
        info_string("Only single move available -> early stopping");
        handle_single_move();
        unlock_and_notify();
    }
    else if (rootNode->get_number_child_nodes() == 0) {
        info_string("The given position has no legal moves");
        unlock_and_notify();
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
        info_string("hash size: ", mapWithMutex.hashTable.size());
        if (mapWithMutex.hashTable.size() > MAX_HASH_SIZE) {
            info_string("clear hash");
            mapWithMutex.hashTable.clear();
        }
        info_string("run mcts search");
        run_mcts_search();
        update_stats();
    }
    update_eval_info(*evalInfo, rootNode.get(), tbHits, maxDepth, searchSettings);
    lastValueEval = evalInfo->bestMoveQ[0];
    lastSideToMove = state->side_to_move();
    update_nps_measurement(evalInfo->calculate_nps());
#ifndef USE_RL
    tGCThread.join();
#endif
}

void MCTSAgent::run_mcts_search()
{
    thread** threads = new thread*[searchSettings->threads];
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        searchThreads[i]->set_root_node(rootNode.get());
        searchThreads[i]->set_root_state(rootState.get());
        searchThreads[i]->set_search_limits(searchLimits);
        searchThreads[i]->set_reached_tablebases(reachedTablebases);
        threads[i] = new thread(run_search_thread, searchThreads[i]);
    }
    int curMovetime = timeManager->get_time_for_move(searchLimits, rootState->side_to_move(), rootNode->plies_from_null()/2);
    ThreadManagerData tData(rootNode.get(), searchThreads, evalInfo, lastValueEval);
    ThreadManagerInfo tInfo(searchSettings, searchLimits, overallNPS, rootState->side_to_move());
    ThreadManagerParams tParams(curMovetime, 250, is_game_sceneario(searchLimits), can_prolong_search(rootNode->plies_from_null()/2, timeManager->get_thresh_move()));
    threadManager = make_unique<ThreadManager>(&tData, &tInfo, &tParams);
    unique_ptr<thread> tManager = make_unique<thread>(run_thread_manager, threadManager.get());
    unlock_and_notify();
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        threads[i]->join();
    }
    threadManager->kill();
    tManager->join();
    delete[] threads;
}

void MCTSAgent::stop()
{
    if (!isRunning) {
        return;
    }
    if (threadManager != nullptr) {
        threadManager->stop_search();
    }
    isRunning = false;
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        info_string("You must do a search before you can print the root node statistics");
        return;
    }
    const vector<size_t> customOrdering = sort_permutation(evalInfo->policyProbSmall, std::greater<float>());
    rootNode->print_node_statistics(rootState.get(), customOrdering, searchSettings);
}

void print_child_nodes_to_file(const Node* parentNode, StateObj* state, size_t parentId, size_t& nodeId, ostream& outFile, size_t depth, size_t maxDepth)
{
    int initialId = nodeId;
    if (maxDepth != 0 && depth > maxDepth) {
        return;
    }
    size_t childIdx = 0;
    for (auto it = parentNode->get_node_it_begin(); it != parentNode->get_node_it_end(); ++it) {
        const Node* node = it->get();
        if (node != nullptr) {
            Action action = parentNode->get_action(childIdx);
            outFile << "N" << ++nodeId << " [label = \""
                    <<  state->action_to_san(action, state->legal_actions(), false, false)
                     << "\"]" << endl;
            int perc = (float(parentNode->get_child_number_visits()[childIdx++]) / parentNode->get_visits()) * 100 + 0.5;
            perc = min(perc+10, 100);
            outFile << "N" << parentId << " -> " << "N" << nodeId
                    << " [color = gray" << 100-perc << "]"
                    <<   ";" << endl;
        }
    }
    outFile  << "{ rank=same; ";
    for (size_t idx = initialId+1; idx < initialId+parentNode->get_no_visit_idx(); ++idx) {
        outFile << "N" << idx << " ";
    }
    outFile << "}" << endl;
    for (auto it = parentNode->get_node_it_begin(); it != parentNode->get_node_it_end(); ++it) {
        const Node* node = it->get();
        if (node != nullptr && node->is_playout_node()) {
            unique_ptr<StateObj> state2 = unique_ptr<StateObj>(state->clone());
            Action action = parentNode->get_action(childIdx);
            state2->do_action(action);
            print_child_nodes_to_file(node, state2.get(), ++initialId, nodeId, outFile, depth+1, maxDepth);
        }
    }
}

void MCTSAgent::export_search_tree(size_t maxDepth, const string& filename)
{
    size_t nodeId = 0;
    ofstream outFile;
    outFile.open (filename);
    outFile << "digraph g {" << endl;
    outFile << "graph [layout = dot]" << endl << endl;
    outFile << "node [shape = circle," << endl
            << " fontname = Helvetica," << endl
            << " fontsize = 8.5," << endl
            << " fixedsize = true," << endl
            << " color = black," << endl
            << " width = 0.3," << endl
            << " height = 0.3," << endl
            << " label = \"\"]" << endl << endl;

    outFile << "edge [" << endl
            << "arrowhead = vee," << endl
            << "arrowsize = 0.2," << endl
            << "color = grey" << endl
            << "]" << endl << endl;

    outFile << "N0 [label = \"root\", xlabel=\"fen: " << rootState->fen() << "\"]" << endl << endl;
    print_child_nodes_to_file(rootNode.get(), rootState.get(), 0, nodeId, outFile, 1, maxDepth);
    outFile << "}" << endl;
    outFile.close();
}
