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

using namespace mxnet::cpp;

void MCTSAgent::run_single_playout() //Board &pos) //, int i) //Node *rootNode)
{
    std::cout << "hello :) " << std::endl;
}

void MCTSAgent::expand_root_node_multiple_moves(const Board &pos)
{

    board_to_planes(pos, 0, true, begin(input_planes)); //input_planes_start);

}

void MCTSAgent::select_node_to_extend()
{

}


MCTSAgent::MCTSAgent(NeuralNetAPI *netSingle, NeuralNetAPI *netBatch,
                     SearchSettings searchSettings, SearchLimits searchLimits, PlaySettings playSettings //,
//                     unordered_map<Key, Node*> *hashTable
                     ):
    Agent(playSettings.temperature, playSettings.temperatureMoves, true),
    netSingle(netSingle),
    netBatch(netBatch),
    searchSettings(searchSettings),
    searchLimits(searchLimits),
    playSettings(playSettings) //,
//    hashTable(hashTable)
{
    hashTable = new unordered_map<Key, Node*>;
    hashTable->reserve(10e6);

    for (auto i = 0; i < searchSettings.threads; ++i) {
        cout << "searchSettings.batchSize" << searchSettings.batchSize << endl;
        searchThreads.push_back(new SearchThread(netBatch, searchSettings.batchSize, searchSettings.virtualLoss, hashTable));
    }


}

EvalInfo MCTSAgent::evalute_board_state(const Board &pos)
{
//    if (legalMoves.size() > 1) {
//        expand_root_node_multiple_moves(pos, legalMoves);
//    }

    size_t nodesPreSearch;

    auto it = hashTable->find(pos.key());
    if(it != hashTable->end()) {
       rootNode = it->second;
       rootNode->make_to_root();
       nodesPreSearch = rootNode->numberVisits;
       cout << "found root node in tree with " << nodesPreSearch << " nodes" << endl;
    }
    else {
        cout << "create new tree" << endl;
        rootNode = new Node(pos, nullptr, 0);
        hashTable->insert({rootNode->pos.key(), rootNode});

        board_to_planes(pos, 0, true, begin(input_planes));
        netSingle->predict(input_planes, valueOutput, probOutputs);
        get_probs_of_move_list(0, probOutputs, rootNode->legalMoves, pos.side_to_move(), true, rootNode->policyProbSmall);

        nodesPreSearch = 0;
    }
    cout << "apply dirichlet" << endl;
    rootNode->apply_dirichlet_noise_to_prior_policy(0.25, 0.2);
    run_mcts_search(pos);

    Constants::init();

    float qValueFac = 0; //0.5;
    float qValueThresh = 0.7;

    DynamicVector<float> mctsPolicy(rootNode->nbDirectChildNodes);
    rootNode->get_mcts_policy(qValueFac, qValueThresh, mctsPolicy);

//    size_t best_idx = argmax(this->rootNode->getPolicyProbSmall());
    size_t best_idx = argmax(mctsPolicy);

    EvalInfo evalInfo;
    evalInfo.centipawns = value_to_centipawn(this->rootNode->getQValues()[best_idx]);
    evalInfo.depth = 42;
    evalInfo.legalMoves = this->rootNode->getLegalMoves();
    evalInfo.pv = {this->rootNode->getLegalMoves()[best_idx]};
    evalInfo.is_chess960 = pos.is_chess960();
    evalInfo.nodes = rootNode->numberVisits;
    evalInfo.nodesPreSearch = nodesPreSearch;
//    eval_info.policyProbSmall = this->rootNode->getPVecSmall();

    return evalInfo;
}

void MCTSAgent::run_mcts_search(const Board &pos)
{
//    const int num_threads = 32;
//    std::thread threads[num_threads];

    searchThreads[0]->setRootNode(rootNode);
    searchThreads[1]->setRootNode(rootNode);

    thread thread1(go, searchThreads[0]);
    thread1.join();

//    for (int i = 0; i < num_threads; ++i) {
////        go();
//        threads[i] = std::thread(run_single_playout); //, pos); //, 3); //this->rootNode);
//    }

//    for (int i = 0; i < num_threads; ++i) {
//        threads[i].join();
//    }
    
}





