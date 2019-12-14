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
 * @file: node.h
 * Created on 28.08.2019
 * @author: queensgambit
 *
 * Class which stores the statistics of all nodes and in the search tree.
 */

#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <mutex>
#include <unordered_map>

#include <blaze/Math.h>

#include "position.h"
#include "movegen.h"
#include "board.h"

#include "agents/config/searchsettings.h"
#include "constants.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;

class Node
{
private:
    mutex mtx;
    Board* pos;
    Node* parentNode;

    // singular values
    float value;
    float visits;

    DynamicVector<float> policyProbSmall;
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;

    size_t numberChildNodes;
    size_t noVisitIdx;

    vector<Node*> childNodes;
    std::vector<Move> legalMoves;
    bool isTerminal;
    size_t childIdxForParent;
    bool hasNNResults;
    bool isFullyExpanded;        // is true if every child node has at least 1 visit

    float uParentFactor;        // stores all parts of the u-value as there a observable by the parent node
    float uDivisorSummand;       // summand which is added to the divisor of the u-divisor

    int checkmateIdx;

    SearchSettings* searchSettings;

    inline void check_for_terminal();

    /**
     * @brief fill_child_node_moves Generates the legal moves and save them in the list
     */
    void fill_child_node_moves();
public:
    /**
     * @brief Node Primary constructor which is used when expanding a node during search
     * @param parentNode Pointer to parent node
     * @param move Move which led to current board state
     * @param searchSettings Pointer to the searchSettings
     */
    Node(Board *pos,
         Node *parentNode,
         size_t childIdxForParent,
         SearchSettings* searchSettings);

    /**
     * @brief Node Copy constructor which copies the value evaluation, board position, prior policy and checkmateIdx.
     * The qValues, actionValues and visits aren't copied over.
     * @param b Node from which the stats will be copied
     */
    Node(const Node& b);

    /**
     * @brief ~Node Destructor which frees memory and the board position
     */
    ~Node();

    /**
     * @brief get_current_u_values Calucates and returns the current u-values for this node
     * @return DynamicVector<float>
     */
    DynamicVector<float> get_current_u_values();

    /**
     * @brief get_child_node Returns the child node at the given index.
     * A nullptr is returned if the child node wasn't expanded yet and no check is done if the childIdx is smaller than
     * @param childIdx Index for the next child node to select
     * @return child node
     */
    Node* get_child_node(size_t childIdx);

    size_t select_child_node();

    /**
     * @brief backup_value Iteratively backpropagates a value prediction across all of the parents for this node.
     * The value is flipped at every ply.
     * @param value Value evaluation to backup, this is the NN eval in the general case or can be from a terminal node
     */
    void backup_value(size_t childIdx, float value);

    /**
     * @brief revert_virtual_loss_and_update Revert the virtual loss effect and apply the backpropagated value of its child node
     * @param childIdx Index to the child node to update
     * @param value Specifies the value evaluation to backpropagate
     */
    void revert_virtual_loss_and_update(size_t childIdx, float value);

    /**
     * @brief backup_collision Iteratively removes the virtual loss of the collision event that occured
     * @param childIdx Index to the child node to update
     */
    void backup_collision(size_t childIdx);

    /**
     * @brief revert_virtual_loss Reverts the virtual loss for a target node
     * @param childIdx Index to the child node to update
     */
    void revert_virtual_loss(size_t childIdx);

    Move get_move(size_t childIdx) const;
    vector<Node*> get_child_nodes() const;
    bool is_terminal() const;
    bool has_nn_results() const;
    Color side_to_move() const;
    Board* get_pos() const;
    float get_value() const;

    void apply_virtual_loss_to_child(size_t childIdx);

    void revert_virtual_loss_and_update(float value);
    Node* get_parent_node() const;
    void increment_visits();
    void increment_no_visit_idx();
    Key hash_key() const;

    size_t get_number_child_nodes() const;

    /**
     * @brief sort_nodes_by_probabilities Sorts all child nodes in ascending order based on their probability value
     */
    void sort_moves_by_probabilities();

    /**
     * @brief calibrate_child_node_order Applies a partial sort for the previously updated nodes depending on nodeIdxUpdate
     */
    void calibrate_child_node_order();

    /**
     * @brief make_to_root Makes the node to the current root node by setting its parent to a nullptr
     */
    void make_to_root();

    float get_visits() const;

    /**
     * @brief get_current_u_divisor Calculates the current u-initialization-divisor factor for this node based on the total node visits
     * @return float
     */
    void update_u_divisor();

    void update_u_parent_factor();

    float compute_current_u_value() const;
    float compute_q_plus_u() const;

    float get_u_parent_factor() const;
    float get_u_divisor_summand() const;

    void lock();
    void unlock();

    /**
     * @brief apply_dirichlet_noise_to_prior_policy Applies dirichlet noise of strength searchSettings->dirichletEpsilon with
     * alpha value searchSettings->dirichletAlpha to the prior policy of the root node. This encourages exploration of nodes with initially low
     * low activations.
     */
    void apply_dirichlet_noise_to_prior_policy();

    /**
     * @brief apply_temperature_to_prior_policy Applies a given temperature value on the root nodes policy distribution.
     * For a temperature < 1, the distribution is "sharpened" and
     * for a temperature > 1, the distribution is "flattened"
     * @param temperature Temperature value (should be non-zero positive value)
     */
    void apply_temperature_to_prior_policy(float temperature);

    float get_action_value() const;
    SearchSettings* get_search_settings() const;

    /**
     * @brief set_parent_node Sets the parent node of this node. This is required when operator()= is used because this operator
     * doesn't set the value for the parent node itself.
     * @param value
     */
    void set_parent_node(Node* value);
    size_t get_no_visit_idx() const;

    bool is_fully_expanded() const;

    /**
     * @brief get_current_cput Calculates the current cpuct value factor for this node based on the total node visits
     * @return float
     */
    inline float get_current_cput();

    /**
     * @brief get_current_u_divisor Calculates the current u-initialization-divisor factor for this node based on the total node visits
     * @return float
     */
    inline float get_current_u_divisor();


    DynamicVector<float>& get_policy_prob_small();

    void set_probabilities_for_moves(const float *data, unordered_map<Move, size_t>& moveLookup);

    void apply_softmax_to_policy();

    /**
     * @brief enhance_moves Calls enhance_checks & enchance captures if the searchSetting suggests it and applies a renormilization afterwards
     */
    void enhance_moves();

    void set_value(float value);
    size_t get_child_idx_for_parent() const;

    void add_new_child_node(Node* newNode, size_t childIdx);

    /**
     * @brief add_transposition_child_node Copies the node with the NN evaluation based on a preexisting node
     * @param it Iterator which from the hash table
     * @param newPos Board position which belongs to the node
     * @param parentNode Parent node of the new node
     * @param childIdx Index on how to visit the child node from its parent
     */
    void add_transposition_child_node(Node* newNode, Board* newPos, size_t childIdx);

    /**
     * @brief max_prob Returns the maximum policy value
     * @return float
     */
    float max_policy_prob();

    /**
     * @brief max_q_child Returns the child index with the highest Q-value
     * @return size_t
     */
    size_t max_q_child();

    /**
     * @brief update_value_eval Returns the updated state evaluation based on the Q-value of the most visited child node
     * @return float
     */
    float updated_value_eval();
    std::vector<Move> get_legal_moves() const;
    int get_checkmate_idx() const;

    /**
     * @brief get_mcts_policy Returns the final policy after the mcts search which is used for move selection, in most cases argmax(mctsPolicy).
     * Depending on the searchSettings, Q-values will be taken into account for creating this.
     * @param node Node for which the mcts policy should be calculated
     * @param childNumberVisits Number of visits for each child node after search
     * @param mctsPolicy Output of the final mcts policy after search
     */
    void get_mcts_policy(DynamicVector<float>& mctsPolicy) const;

    /**
     * @brief get_principal_variation Traverses the tree using the get_mcts_policy() function until a leaf or terminal node is found.
     * The moves a are pushed into the pv vector.
     * @param pv Vector in which moves will be pushed.
     */
    void get_principal_variation(vector<Move>& pv) const;

    /**
     * @brief operator << Overload of stdout operator. Prints move, number visits, probability Value and Q-value
     * @param os ostream handle
     * @param node Node object to print
     * @return ostream
     */
    friend std::ostream& operator<<(std::ostream& os, const Node* node);
    DynamicVector<float> get_child_number_visits() const;
    void enable_has_nn_results();
};

// https://stackoverflow.com/questions/6339970/c-using-function-as-parameter
typedef bool (* vFunctionMoveType)(const Board* pos, Move move);
inline bool isCheck(const Board* pos, Move move);
inline bool isCapture(const Board* pos, Move move);

/**
 * @brief enhance_checks Enhances all possible checking moves below threshCheck by incrementCheck and returns true if a modification
 * was applied. This signals that a renormizalition should be applied afterwards.
 * @param increment_check Constant factor which is added to the checks below threshCheck
 * @param threshCheck Probability threshold for checking moves
 * @return bool
*/
inline bool enhance_move_type(float increment, float thresh, const Board* pos, const vector<Move>& legalMoves,
                              vFunctionMoveType func, DynamicVector<float>& policyProbSmall);

Node* select_child_node(Node* node);

/**
 * @brief delete_subtree Deletes the node itself and its pointer in the hashtable as well as all existing nodes in its subtree.
 * @param node Node of the subtree to delete
 * @param hashTable Pointer to the hashTable which stores a pointer to all active nodes
 */
void delete_subtree_and_hash_entries(Node *node, unordered_map<Key, Node*>* hashTable);

/**
 * @brief delete_sibling_subtrees Deletes all subtrees from all simbling nodes, deletes their hash table entry and sets the visit access to nullptr
 * @param hashTable Pointer to the hashTables
 */
void delete_sibling_subtrees(Node* node, unordered_map<Key, Node*>* hashTable);

typedef float (* vFunctionValue)(Node* node);
DynamicVector<float> retrieve_dynamic_vector(const vector<Node*>& childNodes, vFunctionValue func);

/**
 * @brief get_current_q_thresh Calculates the current q-thresh factor which is used to disable the effect of the q-value for low visited nodes
 * for the final move selection after the search
 * @return float
 */
float get_current_q_thresh(const SearchSettings* searchSettings, int numberVisits);

/**
 * @brief get_current_cput Calculates the current cpuct value factor for this node based on the total node visits
 * @return float
 */
double get_current_cput(float numberVisits, float cpuctBase, float cpuctInit);

/**
 * @brief get_current_u_divisor Calculates the current u-initialization-divisor factor for this node based on the total node visits
 * @return float
 */
float get_current_u_divisor(float numberVisits, float uMin, float uInit, float uBase);

/**
 * @brief print_node_statistics Prints all node statistics of the child nodes to stdout
 */
void print_node_statistics(const Node* node);

/**
 * @brief get_terminal_node_result Returns the game result of the terminal.
 * This function assumes the node to be a terminal node.
 * @param terminalNode Terminal node
 * @return Game result, either DRAWN, WHITE_WIN, BLACK_WIN
 */
Result get_terminal_node_result(const Node* terminalNode);

#endif // NODE_H
