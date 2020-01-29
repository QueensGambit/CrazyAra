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

#include "mctsagent.h"
#include "../evalinfo.h"
#include "movegen.h"
#include "inputrepresentation.h"
#include "outputrepresentation.h"
#include "constants.h"
#include "../util/blazeutil.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "uci.h"
#include "../manager/statesmanager.h"
#include "../manager/treemanager.h"
#include "../node.h"
#include "../util/communication.h"

using namespace mxnet::cpp;

MCTSAgent::MCTSAgent(NeuralNetAPI *netSingle, NeuralNetAPI** netBatches,
                     SearchSettings* searchSettings, PlaySettings* playSettings_,
                     StatesManager *states
                     ):
    Agent(playSettings_, true),
    netSingle(netSingle),
    netBatches(netBatches),
    searchSettings(searchSettings),
    rootNode(nullptr),
    oldestRootNode(nullptr),
    ownNextRoot(nullptr),
    opponentsNextRoot(nullptr),
    states(states),
    lastValueEval(-1.0f),
    reusedFullTree(false)
{
    mapWithMutex = new MapWithMutex();
    mapWithMutex->hashTable = new unordered_map<Key, Node*>;
    mapWithMutex->hashTable->reserve(1e6);

    for (auto i = 0; i < searchSettings->threads; ++i) {
        searchThreads.push_back(new SearchThread(netBatches[i], searchSettings, mapWithMutex));
    }

    valueOutput = new NDArray(Shape(1, 1), Context::cpu());

    if (netSingle->is_policy_map()) {
        probOutputs = new NDArray(Shape(1, NB_LABELS_POLICY_MAP), Context::cpu());
    } else {
        probOutputs = new NDArray(Shape(1, NB_LABELS), Context::cpu());
    }
    timeManager = new TimeManager(searchSettings->randomMoveFactor);
    generator = default_random_engine(r());
    fill(inputPlanes, inputPlanes+NB_VALUES_TOTAL, 0.0f);  // will be filled in evalute_board_state()
}

MCTSAgent::~MCTSAgent()
{
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        delete netBatches[i];
    }
    delete netBatches;
    delete mapWithMutex;
    delete valueOutput;
    delete probOutputs;
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

Board *MCTSAgent::get_root_pos() const
{
    return rootPos;
}

size_t MCTSAgent::init_root_node(Board *pos)
{
    size_t nodesPreSearch;
    rootNode = get_root_node_from_tree(pos);

    if (rootNode != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        states->swap_states();
        nodesPreSearch = size_t(rootNode->get_visits());
        info_string(nodesPreSearch, "nodes of former tree will be reused");
    }
    else {
        create_new_root_node(pos);
        nodesPreSearch = 0;
    }
    return nodesPreSearch;
}

Node *MCTSAgent::get_root_node_from_tree(Board *pos)
{
    reusedFullTree = false;

    if (rootNode == nullptr) {
        return nullptr;
    }
    if (same_hash_key(rootNode, pos)) {
        info_string("reuse the full tree");
        reusedFullTree = true;
        return rootNode;
    }

    if (same_hash_key(ownNextRoot, pos)) {
        delete_sibling_subtrees(ownNextRoot, mapWithMutex->hashTable);
        delete_sibling_subtrees(opponentsNextRoot, mapWithMutex->hashTable);
        return ownNextRoot;
    }
    if (same_hash_key(opponentsNextRoot, pos)) {
        delete_sibling_subtrees(opponentsNextRoot, mapWithMutex->hashTable);
        return opponentsNextRoot;
    }

    // the node wasn't found, clear the old tree except the gameNodes (rootNode, opponentNextRoot)
    delete_old_tree();

    return nullptr;
}

void MCTSAgent::stop_search_based_on_limits()
{
    int curMovetime = timeManager->get_time_for_move(searchLimits, rootNode->side_to_move(), rootNode->plies_from_null()/2);
    info_string("movetime", curMovetime);
    this_thread::sleep_for(chrono::milliseconds(curMovetime/2));
    if (early_stopping()) {
        stop_search();
    } else {
        this_thread::sleep_for(chrono::milliseconds(curMovetime/2));
        if (continue_search()) {
            this_thread::sleep_for(chrono::milliseconds(curMovetime/2));
        }
        stop_search();
    }
}

void MCTSAgent::stop_search()
{
    for (auto searchThread : searchThreads) {
        searchThread->stop();
    }
}

bool MCTSAgent::early_stopping()
{
    if (rootNode->max_policy_prob() > 0.9f && rootNode->max_q_child() == 0) {
        info_string("Early stopping");
        return true;
    }
    return false;
}

bool MCTSAgent::continue_search() {
    if (searchLimits->movetime == 0 && searchLimits->movestogo != 1 && rootNode->updated_value_eval()+0.1f < lastValueEval) {
        info_string("Increase search time");
        return true;
    }
    return false;
}

void MCTSAgent::create_new_root_node(Board *pos)
{
    Board* newPos = new Board(*pos);
    newPos->set_state_info(new StateInfo(*(pos->get_state_info())));

    info_string("create new tree");
    // TODO: Make sure that "inCheck=False" does not cause issues
    rootNode = new Node(newPos, false, nullptr, 0, searchSettings);
    oldestRootNode = rootNode;
    board_to_planes(pos, pos->number_repetitions(), true, begin(inputPlanes));
    netSingle->predict(inputPlanes, *valueOutput, *probOutputs);
    fill_nn_results(0, netSingle->is_policy_map(), valueOutput, probOutputs, rootNode, searchSettings->nodePolicyTemperature);
    gameNodes.push_back(rootNode);
}

void MCTSAgent::delete_old_tree()
{
    // clear all remaining node of the former root node
    if (rootNode != nullptr) {
        for (Node* childNode: rootNode->get_child_nodes()) {
            if (childNode != opponentsNextRoot) {
                delete_subtree_and_hash_entries(childNode, mapWithMutex->hashTable);
            }
        }
        if (opponentsNextRoot != nullptr) {
            for (Node* childNode: opponentsNextRoot->get_child_nodes()) {
                delete_subtree_and_hash_entries(childNode, mapWithMutex->hashTable);
            }
        }
    }
}

void MCTSAgent::delete_game_nodes()
{
    for (Node* node: gameNodes) {
        mapWithMutex->hashTable->erase(node->hash_key());
        delete node;
    }
    gameNodes.clear();
}


void MCTSAgent::apply_move_to_tree(Move move, bool ownMove, Board* pos)
{
    if (!reusedFullTree && rootNode != nullptr) {
        if (ownMove) {
            opponentsNextRoot = pick_next_node(move, rootNode);
            if (opponentsNextRoot != nullptr) {
                info_string("apply move to tree");
                gameNodes.push_back(opponentsNextRoot);
            }
        }
        else {
            ownNextRoot = pick_next_node(move, opponentsNextRoot);
            if (ownNextRoot != nullptr && !ownNextRoot->is_terminal()) {
                if (ownNextRoot->hash_key() == pos->hash_key()) {
                    info_string("apply move to tree");
                    gameNodes.push_back(ownNextRoot);
                }
                else {
                    ownNextRoot = nullptr;
                }
            }
        }
    }
}

void MCTSAgent::clear_game_history()
{
    delete_old_tree();
    delete_game_nodes();

    assert(mapWithMutex->hashTable->size() == 0);
    mapWithMutex->hashTable->clear();
    oldestRootNode = nullptr;
    ownNextRoot = nullptr;
    opponentsNextRoot = nullptr;
    rootNode = nullptr;
    lastValueEval = -1.0f;
}

bool MCTSAgent::is_policy_map()
{
    return netSingle->is_policy_map();
}

string MCTSAgent::get_name() const
{
    return engineName + "-" + engineVersion + "-" + netSingle->get_model_name();
}

void MCTSAgent::evaluate_board_state(Board *pos, EvalInfo& evalInfo)
{
    size_t nodesPreSearch = init_root_node(pos);
    rootPos = pos;
    if (rootNode->get_number_child_nodes() == 1 && int(rootNode->get_visits()) != 0) {
        info_string("Only single move available -> early stopping");
    }
    else if (rootNode->get_checkmate_idx() != -1) {
        info_string("Checkmate in one -> early stopping");
    }
    else if (rootNode->get_number_child_nodes() == 0) {
        info_string("The given position has no legal moves");
    }
    else {
        if (searchSettings->dirichletEpsilon > 0.009f) {
            info_string("apply dirichlet noise");
            rootNode->apply_dirichlet_noise_to_prior_policy();
            rootNode->mark_nodes_as_fully_expanded();
        }

        if (rootNode->get_parent_node() != nullptr) {
            rootNode->make_to_root();
        }

        info_string("run mcts search");
        run_mcts_search();
    }
    evalInfo.childNumberVisits = rootNode->get_child_number_visits();
    evalInfo.policyProbSmall.resize(rootNode->get_number_child_nodes());
    rootNode->get_mcts_policy(evalInfo.policyProbSmall);

    lastValueEval = rootNode->updated_value_eval();
    evalInfo.bestMoveQ = lastValueEval;
    evalInfo.centipawns = value_to_centipawn(lastValueEval);
    evalInfo.legalMoves = rootNode->get_legal_moves();
    rootNode->get_principal_variation(evalInfo.pv);
    evalInfo.depth = evalInfo.pv.size();
    evalInfo.isChess960 = pos->is_chess960();
    evalInfo.nodes = size_t(rootNode->get_visits());
    evalInfo.nodesPreSearch = nodesPreSearch;
}

void MCTSAgent::run_mcts_search()
{
    thread** threads = new thread*[searchSettings->threads];
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        searchThreads[i]->set_root_node(rootNode);
        searchThreads[i]->set_root_pos(rootPos);
        searchThreads[i]->set_search_limits(searchLimits);
        threads[i] = new thread(go, searchThreads[i]);
    }
    if (searchSettings->allowEarlyStopping || searchLimits->nodes == 0) {
        // otherwise the threads will stop by themselves
        stop_search_based_on_limits();
    }
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        threads[i]->join();
    }
    delete[] threads;
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        info_string("You must do a search before you can print the root node statistics");
        return;
    }
    print_node_statistics(rootNode);
}
