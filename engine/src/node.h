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
#include "stateobj.h"

#include "agents/config/searchsettings.h"
#include "nodedata.h"
#include "agents/util/gcthread.h"


using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;

struct ParentNode {
    Node* node;
    uint32_t visits;
    double qSum;
    uint16_t childIdxForParent;
    bool isDead = false;
};

class Node
{
private:
    mutex mtx;

    DynamicVector<float> policyProbSmall;
    vector<Action> legalActions;
    //    DynamicVector<bool> isCheck;
    //    DynamicVector<bool> isCapture;

    vector<ParentNode> parentNodes;
    Key key;

    // singular values
    float value;
    unique_ptr<NodeData> d;

    // identifiers
    uint16_t pliesFromNull;

    bool isTerminal;
    bool isTablebase;
    bool hasNNResults;
    bool sorted;

public:
    /**
     * @brief Node Primary constructor which is used when expanding a node during search
     * @param parentNode Pointer to parent node
     * @param move Move which led to current board state
     * @param searchSettings Pointer to the searchSettings
     */
    Node(StateObj *state,
         bool inCheck,
         Node *parentNode,
         size_t childIdxForParent,
         const SearchSettings* searchSettings);

    /**
     * @brief ~Node Destructor which frees memory and the board position
     */
    ~Node();

    /**
     * @brief get_current_u_values Calucates and returns the current u-values for this node
     * @return DynamicVector<float>
     */
    DynamicVector<float> get_current_u_values(const SearchSettings* searchSettings);

    /**
     * @brief get_child_node Returns the child node at the given index.
     * A nullptr is returned if the child node wasn't expanded yet and no check is done if the childIdx is smaller than
     * @param childIdx Index for the next child node to select
     * @return child node
     */
    Node* get_child_node(size_t childIdx);

    size_t select_child_node(const SearchSettings* searchSettings);

    /**
     * @brief revert_virtual_loss_and_update Revert the virtual loss effect and apply the backpropagated value of its child node
     * @param childIdx Index to the child node to update
     * @param value Specifies the value evaluation to backpropagate
     */
    void revert_virtual_loss_and_update(size_t childIdx, float value, float virtualLoss);

    /**
     * @brief revert_virtual_loss Reverts the virtual loss for a target node
     * @param childIdx Index to the child node to update
     */
    void revert_virtual_loss(size_t childIdx, float virtualLoss);

    bool is_playout_node() const;

    /**
     * @brief is_blank_root_node Returns true if the node is a blank root node with no visits
     * @return True if initialized but no visits else false
     */
    bool is_blank_root_node() const;
    bool is_solved() const;
    bool has_forced_win() const;

    Action get_action(size_t childIdx) const;
    Node* get_child_node(size_t childIdx) const;

    Action get_best_action() const;

    /**
     * @brief get_ponder_moves Returns a list for possible ponder moves
     * @return vector of moves
     */
    vector<Action> get_ponder_moves() const;

    vector<Node*> get_child_nodes() const;
    bool is_terminal() const;
    bool has_nn_results() const;
    float get_value() const;

    void apply_virtual_loss_to_child(size_t childIdx, float virtualLoss);

    /**
     * @brief revert_virtual_loss_and_update Reverts the virtual loss and updates the Q-value and visits
     * @param value New value to update Q
     *
     * Example:
     * Q-value update on 2nd iteration
     * 0. Starting point: Initialized with e.g. 0.5 after first backup, vl = virtual loss
     *   Q_0 = 0.5, n_0 = 1; vl = 1
     * 1. Apply virtual loss
     *   Q_1 = (Q_0 * n_0 - vl) / (n_0 + vl)
     *       = (0.5 * 1 - 1) / (1 + 1)
     *       = - 0.25
     * 2. Increase visits by virtual loss
     *   n_1 = n_0 + vl
     *       = 1 + 1
     *       = 2
     * 3. Revert virtual loss
     *   Q_2 = (Q_1 * n_1 + vl) / (n_1 - vl)
     *       = (-0.25 * 2 + 1) / (2 - 1)
     *       = 0.5
     * 4. Update Q-value by new value (e.g. val = 0.7)
     *   Q_3 = (Q_2 * (n_1 - vl) + val) / (n_1)
     *       = (0.5 * (2 - 1) + 0.7) / 2
     *       = 0.6
     *
     * Note step 3. & 4. ca be expressed as a single update based on Q_1:
     * 3. & 4.: Revert value and update
     *   Q_3 = (Q_1 * n_1 + vl + val) / n_1
     *       = (-0.25 * 2 + 1 + 0.7) / 2
     *       = 0.6
     *
     */
    void revert_virtual_loss_and_update(float value);

    Node* main_parent_node() const;
    Node* get_parent_node(uint8_t parentIdx) const;
    uint16_t get_child_idx_for_parent(uint8_t parentIdx)  const;
    void increment_no_visit_idx();
    void fully_expand_node();

    Key hash_key() const;

    size_t get_number_child_nodes() const;
    uint8_t get_number_parent_nodes() const;

    void prepare_node_for_visits();

    /**
     * @brief sort_nodes_by_probabilities Sorts all child nodes in ascending order based on their probability value
     */
    void sort_moves_by_probabilities();

    /**
     * @brief make_to_root Makes the node to the current root node by setting its parent to a nullptr
     */
    void make_to_root();

    /**
     * @brief get_visits Returns the sum of all visited child nodes with virtual loss applied
     * @return uint32_t
     */
    uint32_t get_visits() const;

    /**
     * @brief get_real_visits Returns visits for given child idx without virtual loss applied
     * @param childIdx Child index
     * @return uint32_t
     */
    uint32_t get_real_visits(uint16_t childIdx) const;

    void lock();
    void unlock();

    /**
     * @brief apply_dirichlet_noise_to_prior_policy Applies dirichlet noise of strength searchSettings->dirichletEpsilon with
     * alpha value searchSettings->dirichletAlpha to the prior policy of the root node. This encourages exploration of nodes with initially low
     * low activations.
     */
    void apply_dirichlet_noise_to_prior_policy(const SearchSettings* searchSettings);

    /**
     * @brief apply_temperature_to_prior_policy Applies a given temperature value on the root nodes policy distribution.
     * For a temperature < 1, the distribution is "sharpened" and
     * for a temperature > 1, the distribution is "flattened"
     * @param temperature Temperature value (should be non-zero positive value)
     */
    void apply_temperature_to_prior_policy(float temperature);

    float get_action_value() const;
    SearchSettings* get_search_settings() const;

    size_t get_no_visit_idx() const;

    bool is_fully_expanded() const;

    DynamicVector<float>& get_policy_prob_small();

    void set_probabilities_for_moves(const float *data, SideToMove sideToMove);

    void apply_softmax_to_policy();

    /**
     * @brief enhance_moves Calls enhance_checks & enhance captures if the searchSetting suggests it and applies a renormilization afterwards
     * @param pos Current board position
     */
    void enhance_moves(const SearchSettings* searchSettings);

    void set_value(float value);
    uint16_t main_child_idx_for_parent() const;

    void add_new_child_node(Node* newNode, size_t childIdx);

    void add_transposition_parent_node(Node* newNode, uint16_t childIdx);

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
     * @brief max_visits_child Returns the child index with the most visits
     * @return size_t
     */
    size_t max_visits_child();

    /**
     * @brief update_value_eval Returns the updated state evaluation based on the Q-value of the most visited child node
     * @return float
     */
    float updated_value_eval() const;
    std::vector<Action> get_legal_actions() const;
    int get_checkmate_idx() const;

    /**
     * @brief get_mcts_policy Returns the final policy after the mcts search which is used for move selection, in most cases argmax(mctsPolicy).
     * Depending on the searchSettings, Q-values will be taken into account for creating this.
     * @param node Node for which the mcts policy should be calculated
     * @param childNumberVisits Number of visits for each child node after search
     * @param mctsPolicy Output of the final mcts policy after search
     */
    void get_mcts_policy(DynamicVector<float>& mctsPolicy, size_t& bestMoveIdx, float qValueWeight = 1) const;

    /**
     * @brief get_principal_variation Traverses the tree using the get_mcts_policy() function until a leaf or terminal node is found.
     * The moves a are pushed into the pv vector.
     * @param pv Vector in which moves will be pushed.
     */
    void get_principal_variation(vector<Action>& pv) const;

    /**
     * @brief mark_nodes_as_fully_expanded Sets the noVisitIdx to be the number of child nodes.
     * This method should be called for instance after applying dirichlet noise,
     * when the node ordering is not guaranteed to be correct anymore.
     */
    void mark_nodes_as_fully_expanded();

    /**
     * @brief is_root_node Checks if the current node is the root node
     * @return true if root node else false
     */
    bool is_root_node() const;

    DynamicVector<uint32_t> get_child_number_visits() const;
    uint32_t get_child_number_visits(uint16_t childIdx) const;

    void enable_has_nn_results();
    uint16_t plies_from_null() const;
    bool is_tablebase() const;
    uint8_t get_node_type() const;
    uint16_t get_end_in_ply() const;
    uint32_t get_terminal_visits() const;

    void init_node_data(size_t numberNodes);
    void init_node_data();

    void mark_as_terminal();

    bool is_sorted() const;

    /**
     * @brief get_q_value Returns the Q-value for the given child index
     * @param idx Child Index
     * @return Q-value
     */
    float get_q_value(size_t idx) const;

    /**
     * @brief get_q_values Returns the Q-values for all child nodes
     * @return Q-values
     */
    DynamicVector<float> get_q_values();

    /**
     * @brief set_q_value Sets a Q-value for a given child index
     * @param idx Child index
     * @param value value to set
     */
    void set_q_value(size_t idx, float value);

    /**
     * @brief get_best_q_idx Return the child index with the highest Q-value
     * @return maximum Q-value
     */
    size_t get_best_q_idx() const;

    /**
     * @brief get_q_idx_over_thresh Returns all child node which coresponding Q-values are greater than qThresh
     * @param qThresh Threshold
     * @return vector of child indices
     */
    vector<size_t> get_q_idx_over_thresh(float qThresh);

    /**
     * @brief print_node_statistics Prints all node statistics of the child nodes to stdout
     */
    void print_node_statistics(const StateObj* pos) const;

    /**
     * @brief get_nodes Returns the number of nodes in the subtree of this node
     * @return uint32_t
     */
    uint32_t get_nodes();

    float main_real_q_value(float virtualLoss);

    bool is_transposition() const;

    void kill_parent_node(const Node* parentNode);

    uint32_t max_parent_visits() const;
    ParentNode* parent_with_most_visits();

    bool only_dead_parents() const;

    double get_q_sum(uint16_t childIdx, float virtualLoss) const;

    template<bool increment>
    void update_virtual_loss_counter(uint16_t childIdx);

    uint8_t get_virtual_loss_counter(uint16_t childIdx) const;

    bool has_transposition_child_node();

    bool is_transposition_return(uint16_t childIdx, float virtualLoss, uint32_t& masterVisits, double& masterQsum) const;

private:

    uint32_t get_real_visits_for_parent(const ParentNode& parent) const;

    double get_q_sum_for_parent(const ParentNode& parent, float virtualLoss) const;

    /**
     * @brief reserve_full_memory Reserves memory for all available child nodes
     */
    void reserve_full_memory();

    /**
     * @brief check_for_terminal Checks if the given board position is a terminal node and updates isTerminal
     * @param state Current board position for this node
     * @param inCheck Boolean indicating if the king is in check
     */
    void check_for_terminal(StateObj* state, bool inCheck);

    /**
     * @brief check_for_tablebase_wdl Checks if the given board position is a tablebase position and
     *  updates isTerminal and the value evaluation
     * @param state Current board position for this node
     */
    void check_for_tablebase_wdl(StateObj* state);

    /**
     * @brief solve_for_terminal Tries to solve the current node to be a forced win, loss or draw.
     * The main idea is based on the paper "Exact-Win Strategy for Overcoming AlphaZero" by Chen et al.
     * https://www.researchgate.net/publication/331216459_Exact-Win_Strategy_for_Overcoming_AlphaZero
     * The solver uses the current backpropagating child node as well as all available child nodes.
     * @param childNode Child nodes which backpropagates the value
     */
    void solve_for_terminal(const Node* childNode);

    /**
     * @brief solved_win Checks if the current node is a solved win based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for SOLVE_WIN else false
     */
    bool solved_win(const Node* childNode) const;

    /**
     * @brief solved_draw Checks if the current node is a solved draw based on the given child node
     * and all available child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for SOLVED_DRAW else false
     */
    bool solved_draw(const Node* childNode) const;

    /**
     * @brief at_least_one_drawn_child Checks if this node has only DRAWN or WON child nodes and at least one DRAWN child
     * @return true if one DRAWN child exits and other child nodes are either won or DRAWN else false
     */
    bool at_least_one_drawn_child() const;

    /**
     * @brief only_won_child_nodws Checks if this node has only WON child nodes
     * @return true if only WON child nodes exist else false
     */
    bool only_won_child_nodes() const;

    /**
     * @brief solved_loss Checks if the current node is a solved loss based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for SOLVED_LOSS else false
     */
    bool solved_loss(const Node* childNode) const;

    /**
     * @brief mark_as_loss Marks the current node as a loss
     */
    void mark_as_loss();

    /**
     * @brief mark_as_draw Marks the current node as a draw
     */
    void mark_as_draw();

    /**
     * @brief mark_as_win Marks the current node as a winning node
     */
    void mark_as_win();

    /**
     * @brief define_end_ply_for_solved_terminal Calculates the number of plies in which the terminal will be reached.
     * The solving is based on the current backpropagating child nodes as well as all available child nodes.
     * @param childNode Child nodes which backpropagates the value
     */
    void define_end_ply_for_solved_terminal(const Node* childNode);

    /**
     * @brief update_solved_terminal Updates member variables for a solved terminal node
     * @param childNode Child nodes which backpropagates the value
     * @param targetValue Target value which will be set to be the new node value
     */
    template <int targetValue>
    void update_solved_terminal(const Node* childNode);

    /**
     * @brief mcts_policy_based_on_wins Sets all known winning moves in a given policy to 1 and all
     * remaining moves to 0. Afterwards the policy is renormalized.
     * @param mctsPolicy MCTS policy which will be set
     */
    void mcts_policy_based_on_wins(DynamicVector<float>& mctsPolicy) const;

    /**
     * @brief prune_losses_in_mcts_policy Sets all known losing moves in a given policy to 0 in case
     * the node is not known to be losing.
     * @param mctsPolicy MCTS policy which will be set
     */
    void prune_losses_in_mcts_policy(DynamicVector<float>& mctsPolicy) const;

    /**
     * @brief mcts_policy_based_on_q_n Creates the MCTS policy based on visits and Q-values
     * @param mctsPolicy MCTS policy which will be set
     */
    void mcts_policy_based_on_q_n(DynamicVector<float>& mctsPolicy, float qValueWeight) const;

//    /**
//     * @brief mark_enhaned_moves Fills the isCheck and isCapture vector according to the legal moves
//     * @param pos Current board positions
//     */
//    void mark_enhanced_moves(const Board* pos, const SearchSettings* searchSettings);

    /**
     * @brief disable_move Disables a given move for futher visits by setting the corresponding Q-value to -INT_MAX
     * and the move probability to 0.
     * @param childIdxForParent Index for the move which will be disabled
     */
    void disable_action(size_t childIdxForParent);
    void set_checkmate_idx(const Node* childNode) const;
};

/**
 * @brief get_best_action_index Returns the best move index of all available moves based on the mcts policy
 * or solved wins / draws / losses.
 * @param curNode Current node
 * @param fast If true, then the argmax(childNumberVisits) is returned for unsolved nodes
 * @return Index for best move and child node
 */
size_t get_best_action_index(const Node* curNode, bool fast);

/**
 * @brief delete_subtree Deletes the node itself and its pointer in the hashtable as well as all existing nodes in its subtree.
 * @param node Node of the subtree to delete
 * @param hashTable Pointer to the hashTable which stores a pointer to all active nodes
 * @param gcThread Reference to the garbadge collector object
 */
void delete_subtree_and_hash_entries(Node *node, unordered_map<Key, Node*>& hashTable, GCThread<Node>& gcThread);

/**
 * @brief delete_sibling_subtrees Deletes all subtrees from all simbling nodes, deletes their hash table entry and sets the visit access to nullptr
 * @param hashTable Pointer to the hashTables
 */
void delete_sibling_subtrees(Node* parentNode, Node* node, unordered_map<Key, Node*>& hashTable, GCThread<Node>& gcThread);

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
float get_current_cput(float visits, const SearchSettings* searchSettings);

/**
 * @brief get_current_u_divisor Calculates the current u-initialization-divisor factor for this node based on the total node visits
 * @return float
 */
float get_current_u_divisor(float numberVisits, float uMin, float uInit, float uBase);

/**
 * @brief node_type_to_string Returns a const char* representation for the enum nodeType
 * @param nodeType Node type
 * @return const char*
 */
const char* node_type_to_string(enum NodeType nodeType);

/**
 * @brief flip_node_type Flips the node type value (e.g. SOLVED_WIN into SOLVED_LOSS)
 * @param nodeType Node type
 * @return flipped node type
 */
NodeType flip_node_type(const enum NodeType nodeType);

/**
 * @brief is_terminal_value Checks if the given value corresponds to a WIN, DRAW or LOSS
 * @param value Node value evaluation
 * @return bool
 */
bool is_terminal_value(float value);

/**
 * @brief get_node_count Returns the number of nodes in the tree without counting terminal simulations
 * @param node Given node
 * @return Number of subnodes for thhe given node
 */
size_t get_node_count(const Node* node);

/**
 * @brief backup_collision Iteratively removes the virtual loss of the collision event that occurred
 * @param rootNode Root node of the tree
 * @param virtualLoss Virtual loss value
 * @param trajectory Trajectory on how to get to the given collision
 */
void backup_collision(Node* rootNode, float virtualLoss, const vector<MoveIdx>& trajectory);

/**
 * @brief backup_value Iteratively backpropagates a value prediction across all of the parents for this node.
 * The value is flipped at every ply.
 * @param rootNode Root node of the tree
 * @param value Value evaluation to backup, this is the NN eval in the general case or can be from a terminal node
 * @param virtualLoss Virtual loss value
 * @param trajectory Trajectory on how to get to the given value eval
 */
void backup_value(Node* rootNode, float value, float virtualLoss, const vector<MoveIdx>& trajectory);

#endif // NODE_H
