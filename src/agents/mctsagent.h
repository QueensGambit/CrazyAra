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
#include "../statesmanager.h"


class MCTSAgent : public Agent
{
private:
    NeuralNetAPI* netSingle;
    NeuralNetAPI** netBatches;

    SearchSettings searchSettings;
    PlaySettings playSettings;

    std::vector<SearchThread*> searchThreads;

    float input_planes[NB_VALUES_TOTAL];
    NDArray* valueOutput;
    NDArray* probOutputs;
//    NDArray valueOutput= NDArray(Shape(1, 1), Context::cpu());
//    NDArray probOutputs = NDArray(Shape(1, NB_LABELS), Context::cpu());

    Node* rootNode;
    vector<Node*> potentialRoots;
    unordered_map<Key, Node*>* hashTable;
    StatesManager* states;

    void expand_root_node_multiple_moves(const Board *pos);

    /**
     * @brief select_node Selects the best child node from a given parent node based on the q and u value
     * @param parentNode Reference to the node object which has been selected
                    If this node hasn't been expanded yet, None will be returned
            move - The move which leads to the selected child node from the given parent node on forward
            node_idx - Integer idx value indicating the index for the selected child of the parent node
    */
    void select_node(Node &parentNode);

    /**
     * @brief reuse_tree Checks if the postion is know and if the tree or parts of the tree can be reused.
     * The old tree or former subtrees will be freed from memory.
     * @param pos Requested board position
     * @return Number of nodes that have already been explored before the serach
     */
    inline size_t reuse_tree(Board* pos);

    /**
     * @brief get_new_root_node Returns the pointer of the new root node for the given position in the case
     * it was either the old root node or an element of the potential root node list.
     * Otherwise a nullptr will be returned.
     * @param pos Requested board position
     * @return Pointer to root node or nullptr
     */
    inline Node* get_new_root_node(Board* pos);

    /**
     * @brief stop_search_based_on_limits Checks for the search limit condition and possible early break-ups
     * and stops all running search threads accordingly
     */
    inline void stop_search_based_on_limits();

    /**
     * @brief stop_search Stops all search threads
     */
    inline void stop_search();

public:

    MCTSAgent(NeuralNetAPI* netSingle,
              NeuralNetAPI** netBatches,
              SearchSettings searchSettings,
              PlaySettings playSettings,
              StatesManager* states); //,
//              unordered_map<Key, Node*> *hashTable);

    EvalInfo evalute_board_state(Board *pos);
    void run_mcts_search();

    /**
     * @brief print_root_node Prints out the root node statistics (visits, q-value, u-value)
     *  by calling the stdout operator for the Node class
     * @param ownMove Defines if the move shall be applied to the current root or the first potential root
     */
    void print_root_node();

    void apply_move_to_tree(Move m, bool ownMove);
};

#endif // MCTSAGENT_H
