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
 * @file: searchthread.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Handles the functionality of a single search thread in the tree.
 */

#ifndef SEARCHTHREAD_H
#define SEARCHTHREAD_H

#include "node.h"
#include "constants.h"
#include "neuralnetapi.h"
#include "config/searchlimits.h"

// wrapper for unordered_map with a mutex for thread safe access
struct MapWithMutex {
    mutex mtx;
    unordered_map<Key, Node*>* hashTable;
    ~MapWithMutex() {
        delete hashTable;
    }
};


class SearchThread
{
private:
    Node* rootNode;
    NeuralNetAPI* netBatch;

    // inputPlanes stores the plane representation of all newly expanded nodes of a single mini-batch
    float* inputPlanes;

    // list of all node objects which have been selected for expansion
    vector<Node*> newNodes;
    vector<Node*> transpositionNodes;
    vector<Node*> collisionNodes;
    vector<Node*> terminalNodes;

    // stores the corresponding value-Outputs and probability-Outputs of the nodes stored in the vector "newNodes"
    // sufficient memory according to the batch-size will be allocated in the constructor
    NDArray* valueOutputs;
    NDArray* probOutputs;

    bool isRunning;

    MapWithMutex* mapWithMutex;
    SearchSettings* searchSettings;
    SearchLimits* searchLimits;

    /**
     * @brief set_nn_results_to_child_nodes Sets the neural network value evaluation and policy prediction vector for every newly expanded nodes
     */
    void set_nn_results_to_child_nodes();

    /**
     * @brief backup_value_outputs Backpropagates all newly received value evaluations from the neural network accross the visited search paths
     */
    void backup_value_outputs();

    /**
     * @brief backup_collisions Reverts the applied virtual loss for all rollouts which ended in a collision event
     */
    void backup_collisions();

public:
    /**
     * @brief SearchThread
     * @param netBatch Network API object which provides the prediction of the neural network
     * @param searchSettings Given settings for this search run
     * @param MapWithMutex Handle to the hash table
     */
    SearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex);
    ~SearchThread();

    /**
     * @brief create_mini_batch Creates a mini-batch of new unexplored nodes.
     * Terminal node are immediatly backpropagated without requesting the NN.
     * If the node was found in the hash-table it's value is backpropagated without requesting the NN.
     * If a collision occurs (the same node was selected multiple times), it will be added to the collisionNodes vector
     */
    void create_mini_batch();

    /**
     * @brief thread_iteration Runs multiple mcts-rollouts as long as a new batch is filled
     */
    void thread_iteration();

    /**
     * @brief nodes_limits_ok Checks if the searchLimits based on the amount of nodes to search has been reached.
     * In the case the number of nodes is set to zero the limit condition is ignored
     * @return bool
     */
    inline bool nodes_limits_ok();

    /**
     * @brief stop Stops the rollouts of the current thread
     */
    void stop();

    // Getter, setter functions
    void set_search_limits(SearchLimits *s);
    Node* get_root_node() const;
    SearchLimits *get_search_limits() const;
    void set_root_node(Node *value);
    bool get_is_running() const;
    void set_is_running(bool value);

    void add_new_node_to_tree(Node* parentNode, size_t childIdx);
};

void go(SearchThread *t);

struct NodeDescription
{
    // flag signaling a collision event, same node was selected multiple time
    bool isCollision;
    // flag signaling a terminal state
    bool isTerminal;
    // depth which was reached on this rollout
    size_t depth;
};

/**
 * @brief get_new_child_to_evaluate Traverses the search tree beginning from the root node and returns the prarent node and child index for the next node to expand.
 * @param rootNode Root node where all simulations start
 * @param useTranspositionTable Flag if the transposition table shall be used
 * @param hashTable Pointer to the hashTable
 * @param description Output struct which holds information what type of node it is
 * @return Pointer to next child to evaluate (can also be terminal or tranposition node in which case no NN eval is required)
 */
Node* get_new_child_to_evaluate(Node* rootNode, size_t& childIdx, NodeDescription& description);

void backup_values(vector<Node*>& nodes);

void fill_nn_results(size_t batchIdx, bool is_policy_map, NDArray* valueOutputs, NDArray* probOutputs, Node *node);

bool is_transposition_verified(const unordered_map<Key,Node*>::const_iterator& it, const StateInfo* stateInfo);

#endif // SEARCHTHREAD_H
