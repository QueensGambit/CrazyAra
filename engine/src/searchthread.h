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
#include "util/fixedvector.h"
#include "nn/neuralnetapiuser.h"


// wrapper for unordered_map with a mutex for thread safe access
struct MapWithMutex {
    mutex mtx;
    unordered_map<Key, Node*> hashTable;
    ~MapWithMutex() {
    }
};

enum NodeBackup : uint8_t {
    NODE_COLLISION,
    NODE_TERMINAL,
    NODE_TRANSPOSITION,
    NODE_NEW_NODE
};

struct NodeDescription
{
    NodeBackup type;
    // depth which was reached on this rollout
    size_t depth;
};

class SearchThread : NeuralNetAPIUser
{
private:
    Node* rootNode;
    StateObj* rootState;
    unique_ptr<StateObj> newState;

    // list of all node objects which have been selected for expansion
    unique_ptr<FixedVector<Node*>> newNodes;
    unique_ptr<FixedVector<SideToMove>> newNodeSideToMove;
    unique_ptr<FixedVector<Node*>> transpositionNodes;
    unique_ptr<FixedVector<Node*>> collisionNodes;

    vector<vector<size_t>> newTrajectories;
    vector<vector<size_t>> transpositionTrajectories;
    vector<vector<size_t>> collisionTrajectories;

    bool isRunning;

    MapWithMutex* mapWithMutex;
    SearchSettings* searchSettings;
    SearchLimits* searchLimits;
    size_t tbHits;
    size_t depthSum;
    size_t depthMax;
    size_t visitsPreSearch;

public:
    /**
     * @brief SearchThread
     * @param netBatch Network API object which provides the prediction of the neural network
     * @param searchSettings Given settings for this search run
     * @param MapWithMutex Handle to the hash table
     */
    SearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex);

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
     * @brief is_root_node_unsolved Checks if the root node result is still unsolved (not a forced win, draw or loss)
     * @return true for unsolved, else false
     */
    inline bool is_root_node_unsolved();

    /**
     * @brief stop Stops the rollouts of the current thread
     */
    void stop();

    // Getter, setter functions
    void set_search_limits(SearchLimits *s);
    Node* get_root_node() const;
    SearchLimits *get_search_limits() const;
    void set_root_node(Node *value);
    bool is_running() const;
    void set_is_running(bool value);

    /**
     * @brief add_new_node_to_tree Adds a new node to the search by either creating a new node or duplicating an exisiting node in case of transposition usage
     * @param newPos Board position of the new node
     * @param parentNode Parent node for the now
     * @param childIdx Respective index for the new node
     * @param inCheck Defines if the current position sets a player in check
     * @return Returns NODE_TRANSPOSITION if a tranpsosition node was added and NODE_NEW_NODE otherwise
     */
    NodeBackup add_new_node_to_tree(StateObj* newPos, Node* parentNode, size_t childIdx, bool inCheck);

    /**
     * @brief reset_tb_hits Sets the number of table hits to 0
     */
    void reset_stats();

    void set_root_state(StateObj* value);
    size_t get_tb_hits() const;

    size_t get_avg_depth();

    size_t get_max_depth() const;

private:
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

    /**
     * @brief get_new_child_to_evaluate Traverses the search tree beginning from the root node and returns the prarent node and child index for the next node to expand.
     * @param pos Temporary position which is initialized as the root position and will result in the final new node position when the function returns
     * @param rootNode Root node where all simulations start
     * @param useTranspositionTable Flag if the transposition table shall be used
     * @param hashTable Pointer to the hashTable
     * @param description Output struct which holds information what type of node it is
     * @param states States list which is used for 3-fold-repetition detection
     * @return Pointer to next child to evaluate (can also be terminal or tranposition node in which case no NN eval is required)
     */
    Node* get_new_child_to_evaluate(StateObj* state, size_t& childIdx, NodeDescription& description, vector<size_t>& trajectory);

    void backup_values(FixedVector<Node*>* nodes, vector<vector<size_t>>& trajectories);
    void backup_transposition_values(FixedVector<Node*>* nodes, vector<vector<size_t>>& trajectories);
};

void run_search_thread(SearchThread *t);

void fill_nn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings);
void node_post_process_policy(Node *node, float temperature, bool isPolicyMap, const SearchSettings* searchSettings);
void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx);

bool is_transposition_verified(const unordered_map<Key,Node*>::const_iterator& it, const StateObj* state);

/**
 * @brief random_root_playout Uses random move exploreation from the ROOT
 * @param description Serach description struct
 * @param currentNode Current node during trajectory
 * @param childIdx Return child index (maybe unchanged)
 */
inline void random_root_playout(NodeDescription& description, Node* currentNode, size_t& childIdx);

#endif // SEARCHTHREAD_H
