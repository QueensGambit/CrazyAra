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
 * @file: mctsagent.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
 * The final move is chosen according to the visit count of each direct child node.
 * One playout is defined as expanding one new node in the tree.
 * In the case of chess this means evaluating a new board position.
 * If the evaluation for one move takes too long on your hardware you can decrease the value for:
 * nb_playouts_empty_pockets and nb_playouts_filled_pockets.
 * For more details and the mathematical equations please take a look at src/domain/agent/README.md as well as the
 * official DeepMind-papers.
 */

#ifndef MCTSAGENT_H
#define MCTSAGENT_H

#include <thread>
#include "position.h"
#include "agent.h"
#include "../evalinfo.h"
#include "../node.h"
#include "../board.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "../searchthread.h"

class MCTSAgent : public Agent
{
private:
    NeuralNetAPI *netSingle;
    NeuralNetAPI **netBatches;

    SearchSettings searchSettings;
    PlaySettings playSettings;

    std::vector<SearchThread*> searchThreads;

    float input_planes[NB_VALUES_TOTAL];
    NDArray valueOutput = NDArray(Shape(1, 1), Context::cpu());
    NDArray probOutputs = NDArray(Shape(1, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH), Context::cpu());

    Node *rootNode;
    unordered_map<Key, Node*> *hashTable;

    void expand_root_node_multiple_moves(const Board &pos);
    static void run_single_playout(); //Board &pos); //, int i); //Node *rootNode);
    void select_node_to_extend();

    /**
     * @brief select_node Selects the best child node from a given parent node based on the q and u value
     * @param parentNode Reference to the node object which has been selected
                    If this node hasn't been expanded yet, None will be returned
            move - The move which leads to the selected child node from the given parent node on forward
            node_idx - Integer idx value indicating the index for the selected child of the parent node
    */
    void select_node(Node &parentNode);

public:

    MCTSAgent(NeuralNetAPI *netSingle,
              NeuralNetAPI** netBatches,
              SearchSettings searchSettings,
              PlaySettings playSettings); //,
//              unordered_map<Key, Node*> *hashTable);

    EvalInfo evalute_board_state(const Board &pos);
    void run_mcts_search(const Board &pos);
};

#endif // MCTSAGENT_H
