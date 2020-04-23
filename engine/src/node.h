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

#include "position.h"
#include "movegen.h"
#include "board.h"

#include "agents/config/searchsettings.h"
#include "constants.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;

enum NodeType : uint8_t {
    SOLVED_WIN,
    SOLVED_DRAW,
    SOLVED_LOSS,
    UNSOLVED
};

class Node
{
private:
    mutex mtx;
    // identifiers
    Key key;
    uint16_t pliesFromNull;
    Color sideToMove;

    Node* parentNode;

    // singular values
    float value;
    float visits;
    float terminalVisits;

    DynamicVector<float> policyProbSmall;
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;
    DynamicVector<bool> isCheck;
    DynamicVector<bool> isCapture;

    NodeType nodeType;
    uint16_t endInPly;
    uint16_t noVisitIdx;
    uint16_t numberChildNodes;
    uint16_t numberUnsolvedChildNodes;

    vector<Node*> childNodes;
    vector<Move> legalMoves;
    bool isTerminal;
    bool isTablebase;
    uint16_t childIdxForParent;
    bool hasNNResults;
    bool isFullyExpanded;        // is true if every child node has at least 1 visit
    int checkmateIdx;

    SearchSettings* searchSettings;

    /**
     * @brief check_for_terminal Checks if the given board position is a terminal node and updates isTerminal
     * @param pos Current board position for this node
     * @param inCheck Boolean indicating if the king is in check
     */
    void check_for_terminal(Board* pos, bool inCheck);

    /**
     * @brief check_for_tablebase_wdl Checks if the given board position is a tablebase position and
     *  updates isTerminal and the value evaluation
     * @param pos Current board position for this node
     */
    void check_for_tablebase_wdl(Board* pos);

    /**
     * @brief fill_child_node_moves Generates the legal moves and save them in the list
     * @param pos Current node position
     */
    void fill_child_node_moves(Board* pos);

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
     * @brief mark_as_loss Marks the current node as a loss and its parent node as a win
     */
    void mark_as_loss();

    /**
     * @brief mark_as_draw Marks the current node as a draw and informs its parent node
     */
    void mark_as_draw();

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
    void update_solved_terminal(const Node* childNode, int targetValue);

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
    void mcts_policy_based_on_q_n(DynamicVector<float>& mctsPolicy) const;

    /**
     * @brief mark_enhaned_moves Fills the isCheck and isCapture vector according to the legal moves
     * @param pos Current board positions
     */
    void mark_enhanced_moves(const Board* pos);

    /**
     * @brief disable_move Disables a given move for futher visits by setting the corresponding Q-value to -INT_MAX
     * and the move probability to 0.
     * @param childIdxForParent Index for the move which will be disabled
     */
    void disable_move(size_t childIdxForParent);

public:
    /**
     * @brief Node Primary constructor which is used when expanding a node during search
     * @param parentNode Pointer to parent node
     * @param move Move which led to current board state
     * @param searchSettings Pointer to the searchSettings
     */
    Node(Board *pos,
         bool inCheck,
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
     * @brief make_to_root Makes the node to the current root node by setting its parent to a nullptr
     */
    void make_to_root();

    float get_visits() const;

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

    DynamicVector<float>& get_policy_prob_small();

    void set_probabilities_for_moves(const float *data, unordered_map<Move, size_t>& moveLookup);

    void apply_softmax_to_policy();

    /**
     * @brief enhance_moves Calls enhance_checks & enhance captures if the searchSetting suggests it and applies a renormilization afterwards
     * @param pos Current board position
     */
    void enhance_moves();

    void set_value(float value);
    size_t get_child_idx_for_parent() const;

    void add_new_child_node(Node* newNode, size_t childIdx);

    /**
     * @brief add_transposition_child_node Copies the node with the NN evaluation based on a preexisting node
     * @param it Iterator which from the hash table
     * @param parentNode Parent node of the new node
     * @param childIdx Index on how to visit the child node from its parent
     */
    void add_transposition_child_node(Node* newNode, size_t childIdx);

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
    float updated_value_eval() const;
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

    /**
     * @brief operator << Overload of stdout operator. Prints move, number visits, probability Value and Q-value
     * @param os ostream handle
     * @param node Node object to print
     * @return ostream
     */
    friend std::ostream& operator<<(std::ostream& os, const Node* node);
    DynamicVector<float> get_child_number_visits() const;
    void enable_has_nn_results();
    int plies_from_null() const;
    Color side_to_move() const;
    bool is_tablebase() const;
    uint8_t get_node_type() const;
    uint16_t get_end_in_ply() const;
    float get_terminal_visits() const;
};

/**
 * @brief get_best_move_index Returns the best move index of all available moves based on the mcts policy
 * or solved wins / draws / losses.
 * @param curNode Current node
 * @return Index for best move and child node
 */
size_t get_best_move_index(const Node* curNode);

/**
 * @brief generate_dtz_values Generates the DTZ values for a given position and all legal moves.
 * This function assumes that the given position is a TB entry.
 * Warning: The DTZ values do not return the fastest way to win but the distance to zeroing (50 move rule counter reset)
 * @param legalMoves Legal moves
 * @param pos Current position
 * @param dtzValues Returned dtz-Values in the view of the current player to use
 */
void generate_dtz_values(const vector<Move> legalMoves, Board& pos, DynamicVector<int>& dtzValues);

// https://stackoverflow.com/questions/6339970/c-using-function-as-parameter
typedef bool (* vFunctionMoveType)(const Board* pos, Move move);
inline bool is_check(const Board* pos, Move move);
inline bool is_capture(const Board* pos, Move move);

/**
 * @brief enhance_checks Enhances all possible checking moves below threshCheck by incrementCheck and returns true if a modification
 * was applied. This signals that a renormizalition should be applied afterwards.
 * @param increment_check Constant factor which is added to the checks below threshCheck
 * @param threshCheck Probability threshold for checking moves
 * @return bool
*/
inline bool enhance_move_type(float increment, float thresh, const vector<Move>& legalMoves,
                              const DynamicVector<bool>& moveType, DynamicVector<float>& policyProbSmall);

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
float get_node_count(const Node* node);

#endif // NODE_H
