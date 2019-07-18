/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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
#include "../blazeutil.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "uci.h"
#include "../statesmanager.h"
#include "../node.h"

using namespace mxnet::cpp;

MCTSAgent::MCTSAgent(NeuralNetAPI *netSingle, NeuralNetAPI** netBatches,
                     SearchSettings searchSettings, PlaySettings playSettings,
                     StatesManager *states
                     ):
    Agent(playSettings.temperature, playSettings.temperatureMoves, true),
    netSingle(netSingle),
    netBatches(netBatches),
    searchSettings(searchSettings),
    playSettings(playSettings),
    rootNode(nullptr),
    oldestRootNode(nullptr),
    states(states)
{
    hashTable = new unordered_map<Key, Node*>;
    hashTable->reserve(1e6);

    for (auto i = 0; i < searchSettings.threads; ++i) {
        cout << "searchSettings.batchSize" << searchSettings.batchSize << endl;
        searchThreads.push_back(new SearchThread(netBatches[i], searchSettings, hashTable));
    }

    valueOutput = new NDArray(Shape(1, 1), Context::cpu());

    if (netSingle->getSelectPolicyFromPlane()) {
        probOutputs = new NDArray(Shape(1, NB_LABELS_POLICY_MAP), Context::cpu());
    } else {
        probOutputs = new NDArray(Shape(1, NB_LABELS), Context::cpu());
    }

    potentialRoots.resize(2);
    timeManager = new TimeManager();
}

void MCTSAgent::expand_root_node_multiple_moves(const Board *pos)
{
    board_to_planes(pos, 0, true, begin(input_planes)); //input_planes_start);
}

size_t MCTSAgent::init_root_node(Board *pos)
{
    size_t nodesPreSearch;
    Node* newRoot = get_root_node_from_tree(pos);

    if (newRoot != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        states->swap_states();
        nodesPreSearch = rootNode->numberVisits;
        sync_cout << "info string reuse the tree with " << nodesPreSearch << " nodes" << sync_endl;
    } else {
        Board* newPos = new Board(*pos);
        newPos->setStateInfo(new StateInfo(*(pos->getStateInfo())));
        if (oldestRootNode != nullptr) {
            sync_cout << "info string delete the old tree " << sync_endl;
            Node::delete_subtree(oldestRootNode, hashTable);
        }
        sync_cout << "info string create new tree" << sync_endl;
        rootNode = new Node(newPos, nullptr, 0, &searchSettings);
        oldestRootNode = rootNode;
        board_to_planes(pos, 0, true, begin(input_planes));
        netSingle->predict(input_planes, *valueOutput, *probOutputs);
        get_probs_of_move_list(0, probOutputs, rootNode->legalMoves, newPos->side_to_move(),
                               !netSingle->getSelectPolicyFromPlane(), rootNode->policyProbSmall, netSingle->getSelectPolicyFromPlane());
        rootNode->enhance_checks();
        nodesPreSearch = 0;
//        hashTable->insert({rootNode->pos->hash_key(), rootNode});
    }

    potentialRoots.clear();

    // TODO: Use individual deletion of entries instead to maintain positions after reusing the tree
    hashTable->clear();
    return nodesPreSearch;
}

Node *MCTSAgent::get_root_node_from_tree(Board *pos)
{
    Node* newRoot = nullptr;

    if (rootNode != nullptr && rootNode->hash_key() == pos->hash_key()) {
        sync_cout << "info string reuse the full tree" << sync_endl;
        newRoot = rootNode;
    }
    else {
        for (Node* node : potentialRoots) {
            if (node != nullptr && node->hash_key() == pos->hash_key()) {
                newRoot = node;
                break;
            }
        }

        if (rootNode != nullptr and newRoot != nullptr) {
            sync_cout << "info string delete unused subtrees" << sync_endl;
            size_t i = 0;
            for (Node *childNode: rootNode->childNodes) {
                if (childNode != nullptr and childNode != newRoot->parentNode) {
                    Node::delete_subtree(childNode, hashTable);
                    rootNode->childNodes[i] = nullptr;
                }
                ++i;
            }

            rootNode = newRoot;
            rootNode->make_to_root();
        }
    }

    return newRoot;
}

void MCTSAgent::stop_search_based_on_limits()
{
    int curMovetime = timeManager->get_time_for_move(searchLimits, rootNode->pos->side_to_move(), rootNode->pos->plies_from_null()/2);
    sync_cout << "string info movetime " << curMovetime << sync_endl;
    this_thread::sleep_for(chrono::milliseconds(curMovetime/2));
    if (early_stopping()) {
        stop_search();
    } else {
        this_thread::sleep_for(chrono::milliseconds(curMovetime/2));
        stop_search();
    }
}

void MCTSAgent::stop_search()
{
    for (size_t i = 0; i < searchSettings.threads; ++i) {
        searchThreads[i]->stop();
    }
}

bool MCTSAgent::early_stopping()
{
    if (max(rootNode->childNumberVisits) > 0.9f * rootNode->numberVisits) {
        sync_cout << "string info Early stopping." << sync_endl;
        return true;
    }
    return false;
}

void MCTSAgent::apply_move_to_tree(Move move, bool ownMove)
{
    Node* parentNode;
    if (ownMove) {
        parentNode = rootNode;
    } else {
        parentNode = potentialRoots[0];
    }

    if (parentNode != nullptr) {
        size_t idx = 0;
        int foundIdx = -1;
        for (Move childMove : parentNode->legalMoves) {
            if (childMove == move) {
                foundIdx = idx;
                break;
            }
            ++idx;
        }

        if (foundIdx != -1 && parentNode->childNodes[foundIdx] != nullptr) {
            potentialRoots.push_back(parentNode->childNodes[foundIdx]);
        }
    }

}

EvalInfo MCTSAgent::evalute_board_state(Board *pos)
{
    size_t nodesPreSearch = init_root_node(pos);

    if (rootNode->nbDirectChildNodes == 1) {
        sync_cout << "info string Only single move available -> early stopping" << sync_endl;
    }
    else if (rootNode->nbDirectChildNodes == 0) {
        sync_cout << "info string The given position has no legal moves" << sync_endl;
    }
    else {
        sync_cout << "info string apply dirichlet" << sync_endl;
        rootNode->apply_dirichlet_noise_to_prior_policy();
        run_mcts_search();
    }

    float qValueFac = searchSettings.qValueWeight;
    float qValueThresh = 0.7;

    DynamicVector<float> mctsPolicy(rootNode->nbDirectChildNodes);
    rootNode->get_mcts_policy(qValueFac, qValueThresh, mctsPolicy);

    size_t best_idx = argmax(mctsPolicy);

    EvalInfo evalInfo;
    evalInfo.centipawns = value_to_centipawn(this->rootNode->getQValues()[best_idx]);
    evalInfo.legalMoves = this->rootNode->getLegalMoves();
    this->rootNode->get_principal_variation(evalInfo.pv);
    evalInfo.depth = evalInfo.pv.size();
    evalInfo.is_chess960 = pos->is_chess960();
    evalInfo.nodes = rootNode->numberVisits;
    evalInfo.nodesPreSearch = nodesPreSearch;

    return evalInfo;
}

void MCTSAgent::run_mcts_search()
{
    thread** threads = new thread*[searchSettings.threads];
    for (size_t i = 0; i < searchSettings.threads; ++i) {
        searchThreads[i]->setRootNode(rootNode);
        searchThreads[i]->set_search_limits(searchLimits);
        threads[i] = new thread(go, searchThreads[i]);
    }

    stop_search_based_on_limits();

    for (size_t i = 0; i < searchSettings.threads; ++i) {
        threads[i]->join();
    }
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        sync_cout << "info string You must do a search before you can print the root node statistics" << sync_endl;
        return;
    }
    sync_cout << rootNode << sync_endl;
}





