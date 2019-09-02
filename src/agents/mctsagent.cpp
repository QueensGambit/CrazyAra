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

using namespace mxnet::cpp;

MCTSAgent::MCTSAgent(NeuralNetAPI *netSingle, NeuralNetAPI** netBatches,
                     SearchSettings* searchSettings, PlaySettings playSettings,
                     StatesManager *states
                     ):
    Agent(playSettings.temperature, playSettings.temperatureMoves, true),
    netSingle(netSingle),
    netBatches(netBatches),
    searchSettings(searchSettings),
    playSettings(playSettings),
    rootNode(nullptr),
    oldestRootNode(nullptr),
    ownNextRoot(nullptr),
    opponentsNextRoot(nullptr),
    states(states),
    lastValueEval(-1.0f),
    reusedFullTree(false)
{
    hashTable = new unordered_map<Key, Node*>;
    hashTable->reserve(1e6);

    for (auto i = 0; i < searchSettings->threads; ++i) {
        searchThreads.push_back(new SearchThread(netBatches[i], searchSettings, hashTable));
    }

    valueOutput = new NDArray(Shape(1, 1), Context::cpu());

    if (netSingle->is_policy_map()) {
        probOutputs = new NDArray(Shape(1, NB_LABELS_POLICY_MAP), Context::cpu());
    } else {
        probOutputs = new NDArray(Shape(1, NB_LABELS), Context::cpu());
    }
    timeManager = new TimeManager(searchSettings->randomMoveFactor);
    generator = default_random_engine(r());
}

size_t MCTSAgent::init_root_node(Board *pos)
{
    size_t nodesPreSearch;
    rootNode = get_root_node_from_tree(pos);

    if (rootNode != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        states->swap_states();
        nodesPreSearch = rootNode->get_visits();
        cout << "info string reuse the tree with " << nodesPreSearch << " nodes" << endl;
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
        cout << "info string reuse the full tree" << endl;
        reusedFullTree = true;
        return rootNode;
    }

    if (same_hash_key(ownNextRoot, pos)) {
        delete_sibling_subtrees(ownNextRoot, hashTable);
        delete_sibling_subtrees(opponentsNextRoot, hashTable);
        ownNextRoot->make_to_root();
        return ownNextRoot;
    }
    if (same_hash_key(opponentsNextRoot, pos)) {
        delete_sibling_subtrees(opponentsNextRoot, hashTable);
        opponentsNextRoot->make_to_root();
        return opponentsNextRoot;
    }
    return nullptr;
}

void MCTSAgent::stop_search_based_on_limits()
{
    int curMovetime = timeManager->get_time_for_move(searchLimits, rootNode->get_pos()->side_to_move(), rootNode->get_pos()->plies_from_null()/2);
    cout << "info string movetime " << curMovetime << endl;
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
    if (rootNode->candidate_child_node()->get_prob_value() > 0.9f && rootNode->candidate_child_node()->get_q_value() > rootNode->alternative_child_node()->get_q_value()) {
        cout << "info string Early stopping" << endl;
        return true;
    }
    return false;
}

bool MCTSAgent::continue_search() {
    if (searchLimits->movetime == 0 && searchLimits->movestogo != 1 && rootNode->candidate_child_node()->get_q_value()+0.1f < lastValueEval) {
        cout << "info Increase search time" << endl;
        return true;
    }
    return false;
}

void MCTSAgent::create_new_root_node(Board *pos)
{
    Board* newPos = new Board(*pos);
    newPos->setStateInfo(new StateInfo(*(pos->getStateInfo())));
    if (oldestRootNode != nullptr) {
        cout << "info string delete the old tree " << endl;
        if (opponentsNextRoot != nullptr) {
            delete_sibling_subtrees(opponentsNextRoot, hashTable);
        }
    }
    cout << "info string create new tree" << endl;
    rootNode = new Node(pos, nullptr, MOVE_NONE, searchSettings);
    rootNode->expand();
    oldestRootNode = rootNode;
    board_to_planes(pos, 0, true, begin(input_planes));
    netSingle->predict(input_planes, *valueOutput, *probOutputs);
    fill_nn_results(0, netSingle->is_policy_map(), searchSettings, valueOutput, probOutputs, rootNode);
    rootNode->make_to_root();
    gameNodes.push_back(rootNode);
}


void MCTSAgent::apply_move_to_tree(Move move, bool ownMove)
{
    if (!reusedFullTree) {
        if (ownMove) {
            opponentsNextRoot = pick_next_node(move, rootNode);
            if (opponentsNextRoot != nullptr && opponentsNextRoot->has_nn_results()) {
                cout << "info string apply move to tree" << endl;
                gameNodes.push_back(opponentsNextRoot);
            }
        }
        else {
            ownNextRoot = pick_next_node(move, opponentsNextRoot);
            if (ownNextRoot != nullptr && ownNextRoot->has_nn_results()) {
                cout << "info string apply move to tree" << endl;
                gameNodes.push_back(ownNextRoot);
            }
        }
    }
}

void MCTSAgent::clear_game_history()
{
    for (Node* node: gameNodes) {
        delete node;
    }
    gameNodes.clear();
    hashTable->clear();
    oldestRootNode = nullptr;
    rootNode = nullptr;
    lastValueEval = -1.0f;
}

void MCTSAgent::evalute_board_state(Board *pos, EvalInfo& evalInfo)
{
    size_t nodesPreSearch = init_root_node(pos);

    if (rootNode->get_number_child_nodes() == 1) {
        cout << "info string Only single move available -> early stopping" << endl;
    }
    else if (rootNode->get_checkmate_node() != nullptr) {
        cout << "info string Checkmate in one -> early stopping" << endl;
    }
    else if (rootNode->get_number_child_nodes() == 0) {
        cout << "info string The given position has no legal moves" << endl;
    }
    else {
        cout << "info string apply dirichlet TODO" << endl;
//        rootNode->apply_dirichlet_noise_to_prior_policy();
        run_mcts_search();
    }

    evalInfo.childNumberVisits.resize(rootNode->get_number_child_nodes());
    evalInfo.childNumberVisits = rootNode->get_visits();

    evalInfo.policyProbSmall.resize(rootNode->get_number_child_nodes());
	
    get_mcts_policy(rootNode, searchSettings->qValueWeight,
                    get_current_q_thresh(searchSettings, rootNode->get_visits()),
                    evalInfo.policyProbSmall);
    size_t bestIdx = argmax(evalInfo.policyProbSmall);
//    if (bestIdx != argmax(rootNode->childNumberVisits)) {
//        cout << "info string Select different move due to higher Q-value" << endl;
//    }
    evalInfo.centipawns = value_to_centipawn(updated_value(rootNode));
    lastValueEval = updated_value(rootNode);
    evalInfo.legalMoves = retrieve_legal_moves(rootNode->get_child_nodes());
//    this->rootNode->get_principal_variation(evalInfo.pv); // TODO
    evalInfo.pv = {rootNode->candidate_child_node()->get_move()};
    evalInfo.depth = evalInfo.pv.size();
    evalInfo.is_chess960 = pos->is_chess960();
    evalInfo.nodes = rootNode->get_visits();
    evalInfo.nodesPreSearch = nodesPreSearch;
}

void MCTSAgent::run_mcts_search()
{
    thread** threads = new thread*[searchSettings->threads];
    for (size_t i = 0; i < searchSettings->threads; ++i) {
        searchThreads[i]->set_root_node(rootNode);
        searchThreads[i]->set_search_limits(searchLimits);
        threads[i] = new thread(go, searchThreads[i]);
    }
    stop_search_based_on_limits();

    for (size_t i = 0; i < searchSettings->threads; ++i) {
        threads[i]->join();
    }
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        cout << "info string You must do a search before you can print the root node statistics" << endl;
        return;
    }
    rootNode->print_node_statistics();
}


