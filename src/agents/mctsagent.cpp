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

void MCTSAgent::run_single_playout() //Board *pos) //, int i) //Node *rootNode)
{
    std::cout << "hello :) " << std::endl;
}

void MCTSAgent::expand_root_node_multiple_moves(const Board *pos)
{

    board_to_planes(pos, 0, true, begin(input_planes)); //input_planes_start);

}

void MCTSAgent::select_node_to_extend()
{

}

size_t MCTSAgent::reuse_tree(Board *pos)
{
    size_t nodesPreSearch;

    auto it = hashTable->find(pos->hash_key());
    if(it != hashTable->end()) {
        // swap the states because now the old states are used
        // This way the memory won't be freed for the next new move
        states->swap_states();

        if (rootNode == it->second) {
            sync_cout << "info string reuse the full tree" << sync_endl;
        }
        else {
            if (rootNode != nullptr) {
                sync_cout << "info string delete unused subtrees" << sync_endl;
                for (Node *childNode: rootNode->childNodes) {
                    if (childNode != nullptr and childNode != it->second->parentNode) {
                        Node::delete_subtree(childNode);
                    }
                }
            }
            rootNode = it->second;
            rootNode->make_to_root();
        }
        nodesPreSearch = rootNode->numberVisits;
        sync_cout << "info string reuse the tree with " << nodesPreSearch << " nodes" << sync_endl;
    }
    else {
        if (rootNode != nullptr) {
            sync_cout << "info string delete the old tree " << sync_endl;
            Node::delete_subtree(rootNode);
        }
        sync_cout << "info string create new tree" << sync_endl;
        rootNode = new Node(new Board(*pos), nullptr, 0);
        board_to_planes(pos, 0, true, begin(input_planes));
        netSingle->predict(input_planes, valueOutput, probOutputs);
        get_probs_of_move_list(0, probOutputs, rootNode->legalMoves, pos->side_to_move(), true, rootNode->policyProbSmall);
        rootNode->enhance_checks();
        nodesPreSearch = 0;
    }

    return nodesPreSearch;
}


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


}

EvalInfo MCTSAgent::evalute_board_state(Board *pos)
{
    size_t nodesPreSearch = reuse_tree(pos);

    hashTable->clear();
    hashTable->insert({rootNode->pos->hash_key(), rootNode});

    cout << "info string apply dirichlet" << endl;
    rootNode->apply_dirichlet_noise_to_prior_policy(0.25, 0.2);
    run_mcts_search();

    float qValueFac = 0.7; //0.5; //0.5;
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
    for (size_t i = 0; i < 2; ++i) {
        searchThreads[i]->setRootNode(rootNode);
        searchThreads[i]->set_search_limits(searchLimits);
    }

    thread thread1(go, searchThreads[0]);
    thread thread2(go, searchThreads[1]);

    thread1.join();
    thread2.join();
}

void MCTSAgent::print_root_node()
{
    if (rootNode == nullptr) {
        sync_cout << "info string You must do a search before you can print the root node statistics" << sync_endl;
        return;
    }
    sync_cout << rootNode << sync_endl;
}





