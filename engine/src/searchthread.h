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


enum NodeBackup : uint8_t {
    NODE_COLLISION,
    NODE_TERMINAL,
    NODE_TRANSPOSITION,
    NODE_NEW_NODE,
    NODE_UNKNOWN,
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
    unique_ptr<FixedVector<float>> transpositionValues;

    vector<Trajectory> newTrajectories;
    vector<Trajectory> transpositionTrajectories;
    vector<Trajectory> collisionTrajectories;

    Trajectory trajectoryBuffer;
    vector<Action> actionsBuffer;

    bool isRunning;

    MapWithMutex* mapWithMutex;
    const SearchSettings* searchSettings;
    SearchLimits* searchLimits;
    size_t tbHits;
    size_t depthSum;
    size_t depthMax;
    size_t visitsPreSearch;
    const uint_fast32_t terminalNodeCache;
    bool reachedTablebases;
public:
    /**
     * @brief SearchThread
     * @param netBatch Network API object which provides the prediction of the neural network
     * @param searchSettings Given settings for this search run
     * @param MapWithMutex Handle to the hash table
     */
    SearchThread(NeuralNetAPI* netBatch, const SearchSettings* searchSettings, MapWithMutex* mapWithMutex);

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
    void set_reached_tablebases(bool value);

    /**
     * @brief add_new_node_to_tree Adds a new node to the search by either creating a new node or duplicating an exisiting node in case of transposition usage
     * @param newPos Board position of the new node
     * @param parentNode Parent node for the now
     * @param childIdx Respective index for the new node
     * @param nodeBackup Returns NODE_TRANSPOSITION if a tranpsosition node was added and NODE_NEW_NODE otherwise
     * @return The newly added node
     */
    Node* add_new_node_to_tree(StateObj* newPos, Node* parentNode, ChildIdx childIdx, NodeBackup& nodeBackup);

    /**
     * @brief reset_tb_hits Sets the number of table hits to 0
     */
    void reset_stats();

    void set_root_state(StateObj* value);
    size_t get_tb_hits() const;

    size_t get_avg_depth();

    size_t get_max_depth() const;

    Node* get_starting_node(Node* currentNode, NodeDescription& description, ChildIdx& childIdx);

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
     * @param description Output struct which holds information what type of node it is
     * @return Pointer to next child to evaluate (can also be terminal or tranposition node in which case no NN eval is required)
     */
    Node* get_new_child_to_evaluate(NodeDescription& description);

    void backup_values(FixedVector<Node*>& nodes, vector<Trajectory>& trajectories);
    void backup_values(FixedVector<float>* values, vector<Trajectory>& trajectories);

    /**
     * @brief select_enhanced_move Selects an enhanced move (e.g. checking move) which has not been explored under given conditions.
     * @param currentNode Current node during forward simulation
     * @return uint_16_t(-1) for no action else custom idx
     */
    ChildIdx select_enhanced_move(Node* currentNode) const;
};

void run_search_thread(SearchThread *t);

void fill_nn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, const float* auxiliaryOutputs, Node *node, size_t& tbHits, bool mirrorPolicy, const SearchSettings* searchSettings, bool isRootNodeTB);
void node_post_process_policy(Node *node, float temperature, const SearchSettings* searchSettings);
void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx, bool isRootNodeTB);

/**
 * @brief random_root_playout Uses random move exploration (epsilon greedy) from the given position. The probability for doing a random move decays by depth.
 * @param currentNode Current node during trajectory
 * @param childIdx Return child index (maybe unchanged)
 */
inline void random_playout(Node* currentNode, ChildIdx& childIdx);

/**
 * @brief get_random_depth
 * Example: drawing a random number from a uniform distribution in [0, 100]
 * DEPTH 0: 0 - 50
 * DEPTH 1: 51 - 75
 * DEPTH 2: 76 - 77
 * DEPTH 3: 77 - 94
 * DEPTH 4: 95 - 97
 * DEPTH 5: 98 - 99
 * DEPTH 6: 100
 * @return random depth while the probability of choosing higher depths decreases exponetially
 */
size_t get_random_depth();

#endif // SEARCHTHREAD_H
