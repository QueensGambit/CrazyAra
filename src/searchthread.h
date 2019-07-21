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

class SearchThread
{
private:
    Node* rootNode;
    NeuralNetAPI *netBatch;

    float *inputPlanes;

    // list of all node objects which have been selected for expansion
    std::vector<Node*> newNodes;
    std::vector<Node*> transpositionNodes;
    std::vector<Node*> collisionNodes;
    std::vector<Node*> terminalNodes;

    // stores the corresponding value-Outputs and probability-Outputs of the nodes stored in the vector "newNodes"
    // sufficient memory according to the batch-size will be allocated in the constructor
    NDArray* valueOutputs;
    NDArray* probOutputs;

    bool isRunning;

    unordered_map<Key, Node*> *hashTable;
    SearchSettings searchSettings;
    SearchLimits* searchLimits;

    /**
     * @brief get_new_child_to_evaluate Traverses the search tree beginning from the root node and returns the prarent node and child index for the next node to expand.
     * In the case a collision event occured the isCollision flag will be set and for a terminal node the isTerminal flag is set.
     * @param childIdx Move index for the parent node in order to expand the next node
     * @param isCollision Flag signaling a collision event, same node was selected multiple time
     * @param isTerminal Flag signaling a terminal state
     * @param depth Depth which was reached on this rollout
     * @return
     */
    inline Node* get_new_child_to_evaluate(unsigned int &childIdx, bool &isCollision,  bool &isTerminal, size_t &depth);

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
     * @brief create_new_node Creates a new node which will be added to the tree
     * @param newPos Board position which belongs to the node
     * @param parentNode Parent node of the new node
     * @param childIdx Index on how to visit the child node from its parent
     * @param numberNewNodes Index of the new node in the current batch
     */
    inline void create_new_node(Board* newPos, Node* parentNode, size_t childIdx, size_t numberNewNodes);

    /**
     * @brief copy_node Copies the node with the NN evaluation based on a preexisting node
     * @param it Iterator which from the hash table
     * @param newPos Board position which belongs to the node
     * @param parentNode Parent node of the new node
     * @param childIdx Index on how to visit the child node from its parent
     */
    inline void copy_node(const unordered_map<Key,Node*>::const_iterator &it, Board* newPos, Node* parentNode, size_t childIdx);

public:
    /**
     * @brief SearchThread
     * @param netBatch Network API object which provides the prediction of the neural network
     * @param searchSettings Given settings for this search run
     * @param hashTable Handle to the hash table
     */
    SearchThread(NeuralNetAPI* netBatch, SearchSettings searchSettings, unordered_map<Key, Node*>* hashTable);

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
};

void go(SearchThread *t);


#endif // SEARCHTHREAD_H
