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
    Move move;
    float value;
    float probValue;
    double qValue;
    double actionValue;
    unsigned int visits;
    int virtualLossCounter;

    bool isTerminal;
    size_t numberChildNodes;

    vector<Node*> childNodes;

    bool isExpanded;
    bool hasNNResults;
    bool isCalibrated;  // determines if the nodes are ordered
    bool areChildNodesSorted;

    // if checkmateNode is != nullptr it will always be preferred over all other nodes
    Node* checkmateNode;

    SearchSettings* searchSettings;

    inline void check_for_terminal();

public:
    Node(Node* parentNode, Move move);
    /**
     * @brief Node Constructor used for creating a new root node
     * @param parentNode Pointer to parent node
     * @param move Move which led to current board state
     * @param pos Current board position
     */
    Node(Board *pos, Node *parentNode, Move move);

    void expand();
    Move get_move() const;
    vector<Node*> get_child_nodes() const;
    bool is_terminal() const;
    bool has_nn_results() const;
    Color side_to_move() const;
    Board* get_pos() const;
    void set_prob_value(float value);
    float get_value() const;

    void set_nn_results(float nn_value, const DynamicVector<float>& policyProbSmall);
    void apply_virtual_loss();
    void revert_virtual_loss();
    void revert_virtual_loss_and_update(float value);
    Node* get_parent_node() const;
    void increment_visits();
    bool is_expanded() const;
    bool are_child_nodes_sorted() const;
    bool is_calibrated() const;
    Node* first_child_node() const;
    Node* second_child_node() const;
    Key hash_key() const;

    size_t get_number_child_nodes() const;
    Node* get_checkmate_node() const;

    /**
     * @brief sort_nodes_by_probabilities Sorts all child nodes in ascending order based on their probability value
     */
    void sort_child_nodes_by_probabilities();

    /**
     * @brief make_to_root Makes the node to the current root node by setting its parent to a nullptr
     */
    void make_to_root();

    unsigned int get_visits() const;
    float get_prob_value() const;
    double get_q_value() const;

    /**
     * @brief init_board Initializes the board position given the previous position and move to play
     * @param parentPos Previous board position
     * @param move Move to apply
     * @param pos New position which will be set
     */
    void init_board();

};

// generate the legal moves and save them in the list

/**
 * @brief backup_value Iteratively backpropagates a value prediction across all of the parents for this node.
 * The value is flipped at every ply.
 * @param value Value evaluation to backup, this is the NN eval in the general case or can be from a terminal node
 */
void backup_value(Node* currentNode, float value);

void backup_collision(Node* currentNode);

/**
 * @brief retrieve_legal_moves Retrieves each move from the child nodes
 * @param childNodes List of child nodes
 * @return Legal move list
 */
vector<Move> retrieve_legal_moves(const vector<Node*>& childNodes);

/**
 * @brief create_child_nodes Generates a new child node for every legal move
 * @param parentNode Parent node to which all child nodes will be connected with
 * @param pos Board position
 * @param childNodes Empty vector which will be filled for every legal move
 */
inline void create_child_nodes(Node* parentNode, const Board* pos, vector<Node*> &childNodes);

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

/**
  * @brief enhance_moves Calls enhance_checks & enchance captures if the searchSetting suggests it and applies a renormilization afterwards
  * @param threshCheck Threshold probability for checking moves
  * @param checkFactor Factor based on the maximum probability with which checks will be increased
  * @param threshCapture Threshold probability for capture moves
  * @param captureFactor Factor based on the maximum probability with which captures will be increased
  */
void enhance_moves(const SearchSettings* searchSettings, const Board* pos, const vector<Move>& legalMoves, DynamicVector<float>& policyProbSmall);

Node* select_child_node(Node* node);

/**
 * @brief operator << Overload of stdout operator. Prints move, number visits, probability Value and Q-value
 * @param os ostream handle
 * @param node Node object to print
 * @return ostream
 */
extern std::ostream& operator<<(std::ostream& os, Node* node);

/**
 * @brief print_node_statistics Prints all node statistics of the child nodes to stdout
 */
void print_node_statistics(Node* node);

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

float get_visits(Node* node);
float get_q_value(Node* node);
typedef float (* vFunctionValue)(Node* node);
DynamicVector<float> retrieve_dynamic_vector(const vector<Node*>& childNodes, vFunctionValue func);

DynamicVector<float> retrieve_visits(const Node* node);
DynamicVector<float> retrieve_q_values(const Node* node);

/**
 * @brief get_mcts_policy Returns the final policy after the mcts search which is used for move selection, in most cases argmax(mctsPolicy).
 * Depending on the searchSettings, Q-values will be taken into account for creating this.
 * @param mctsPolicy Output of the final mcts policy after search
 */
void get_mcts_policy(const Node *node, const float qValueWeight, const float qThresh, DynamicVector<float> &mctsPolicy);

/**
 * @brief get_current_q_thresh Calculates the current q-thresh factor which is used to disable the effect of the q-value for low visited nodes
 * for the final move selection after the search
 * @return float
 */
float get_current_q_thresh(SearchSettings* searchSettings, int numberVisits);

double updated_value(const Node* node);

#endif // NODE_H
