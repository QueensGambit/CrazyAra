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
    states(states)
{
    hashTable = new unordered_map<Key, Node*>;
    hashTable->reserve(10e6);

    for (auto i = 0; i < searchSettings.threads; ++i) {
        cout << "searchSettings.batchSize" << searchSettings.batchSize << endl;
        searchThreads.push_back(new SearchThread(netBatches[i], searchSettings.batchSize, searchSettings.virtualLoss, hashTable));
    }

    valueOutput = new NDArray(Shape(1, 1), Context::cpu());

    if (netSingle->getSelectPolicyFromPlane()) {
        probOutputs = new NDArray(Shape(1, NB_LABELS_POLICY_MAP), Context::cpu());
    } else {
        probOutputs = new NDArray(Shape(1, NB_LABELS), Context::cpu());
    }

    potentialRoots.resize(2);
}

void MCTSAgent::expand_root_node_multiple_moves(const Board *pos)
{

    board_to_planes(pos, 0, true, begin(input_planes)); //input_planes_start);

}

size_t MCTSAgent::reuse_tree(Board *pos)
{
    size_t nodesPreSearch;
    Node* newRoot = get_new_root_node(pos);

    if (newRoot != nullptr) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        states->swap_states();
        nodesPreSearch = rootNode->numberVisits;
        sync_cout << "info string reuse the tree with " << nodesPreSearch << " nodes" << sync_endl;
    } else {
        Board* newPos = new Board(*pos);
        newPos->setStateInfo(new StateInfo(*(pos->getStateInfo())));
        if (rootNode != nullptr) {
            sync_cout << "info string delete the old tree " << sync_endl;
            Node::delete_subtree(rootNode, hashTable);
        }
        sync_cout << "info string create new tree" << sync_endl;
        rootNode = new Node(newPos, nullptr, 0);
        board_to_planes(pos, 0, true, begin(input_planes));
        netSingle->predict(input_planes, *valueOutput, *probOutputs);
//        cout << "valueOutput: " << valueOutput << endl;
        get_probs_of_move_list(0, probOutputs, rootNode->legalMoves, newPos->side_to_move(),
                               !netSingle->getSelectPolicyFromPlane(), rootNode->policyProbSmall, netSingle->getSelectPolicyFromPlane());
//        cout << "policyProbSmall: " << rootNode->policyProbSmall << endl;
        rootNode->enhance_checks();
        nodesPreSearch = 0;
//        hashTable->insert({rootNode->pos->hash_key(), rootNode});
    }

    potentialRoots.clear();
    return nodesPreSearch;
}

Node *MCTSAgent::get_new_root_node(Board *pos)
{
    Node* newRoot = nullptr;

    if (rootNode != nullptr and rootNode->hash_key() == pos->hash_key()) {
        sync_cout << "info string reuse the full tree" << sync_endl;
        newRoot = rootNode;
    }
    else {
        for (Node* node : potentialRoots) {
            if (node != nullptr and node->hash_key() == pos->hash_key()) {
                newRoot = node;
                break;
            }
        }
        if (rootNode != nullptr and newRoot != nullptr) {
            sync_cout << "info string delete unused subtrees" << sync_endl;
            for (Node *childNode: rootNode->childNodes) {
                if (childNode != nullptr and childNode != newRoot->parentNode) {
                    Node::delete_subtree(childNode, hashTable);
                }
            }
            // delete the old rootNode
//            hashTable->erase(rootNode->hash_key());
//            delete rootNode;

//            // delete the new root parent
//            hashTable->erase(newRoot->parentNode->hash_key());
//            delete newRoot->parentNode;

            rootNode = newRoot;
            rootNode->make_to_root();
        }
    }

    return newRoot;
}

void MCTSAgent::apply_move_to_tree(Move m, bool ownMove)
{
    Node* parentNode;
    if (ownMove) {
        parentNode = rootNode;
    } else {
        parentNode = potentialRoots[0];
    }

    if (parentNode != nullptr) {
        size_t idx = 0;
        for (Move childMove : parentNode->legalMoves) {
            if (childMove == m) {
                break;
            }
            ++idx;
        }

        if (parentNode->childNodes[idx] != nullptr) {
            potentialRoots.push_back(parentNode->childNodes[idx]);
        }
    }

}


EvalInfo MCTSAgent::evalute_board_state(Board *pos)
{
    size_t nodesPreSearch = reuse_tree(pos);

    cout << "info string apply dirichlet" << endl;
    rootNode->apply_dirichlet_noise_to_prior_policy(0.25, 0.2);
    run_mcts_search();

    float qValueFac = 0.0; //0.5; //0.5;
    float qValueThresh = 0.7;

    DynamicVector<float> mctsPolicy(rootNode->nbDirectChildNodes);
    rootNode->get_mcts_policy(qValueFac, qValueThresh, mctsPolicy);

    size_t best_idx = argmax(mctsPolicy);

    EvalInfo evalInfo;
    evalInfo.centipawns = value_to_centipawn(this->rootNode->getQValues()[best_idx]);
    evalInfo.depth = 42;
    evalInfo.legalMoves = this->rootNode->getLegalMoves();
    this->rootNode->get_principal_variation(evalInfo.pv);
//    evalInfo.pv = {this->rootNode->getLegalMoves()[best_idx]};
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





