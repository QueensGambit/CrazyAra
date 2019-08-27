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
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Class which stores the statistics of all nodes and in the search tree.
 */

#ifndef NODE_H
#define NODE_H

#include <mutex>
#include "position.h"
#include "movegen.h"
#include "board.h"

#include <blaze/Math.h>

using blaze::HybridVector;
using blaze::DynamicVector;
#include <unordered_map>
using namespace std;
#include <iostream>
#include "agents/config/searchsettings.h"

class Node
{
private:
    std::mutex mtx;
    float value;
    Board* pos;
    DynamicVector<float> policyProbSmall;
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;

    // singular values
    double probValue;
    double qValue;
    double actionValue;
    double visits;
    Move move;

    std::vector<Move> legalMoves;
    bool isTerminal;
    unsigned int nbDirectChildNodes;

    float initialValue;
    int numberVisits = 0;
    std::vector<Node*> childNodes;

    Node* canidateNode;
    Node* secondCandidateNode;
    Node* parentNode;
    unsigned int childIdxForParent;
    bool hasNNResults;

    // if checkMateIdx is != -1 it will always be preferred over all other nodes
    int checkmateIdx;

    SearchSettings* searchSettings;

    /**
     * @brief check_for_terminal Checks if the currect node is a terminal node and updates the checkmateIdx for its parent in case of a checkmate terminal
     */
    inline void check_for_terminal();

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

    /**
     * @brief get_current_q_thresh Calculates the current q-thresh factor which is used to disable the effect of the q-value for low visited nodes
     * for the final move selection after the search
     * @return float
     */
    inline float get_current_q_thresh();

    /**
     * @brief get_current_u_values Calucates and returns the current u-values for this node
     * @return DynamicVector<float>
     */
    DynamicVector<float> get_current_u_values();

    /**
     * @brief get_current_u_values Calucates anCalucates and returns the current u-values for this node
     * @return DynamicVector<float>
     */
    inline double get_current_u_value() const;

    /**
      * @brief enhance_checks Enhances all possible checking moves below threshCheck by incrementCheck and returns true if a modification
      * was applied. This signals that a renormizalition should be applied afterwards.
      * @param increment_check Constant factor which is added to the checks below threshCheck
      * @param threshCheck Probability threshold for checking moves
      * @return bool
      */
     inline bool enhance_checks(const float incrementCheck, float threshCheck);

     /**
       * @brief enhance_captures Enhances all possible capture moves below threshCapture by incrementCapture and returns true if a modification
       * was applied. This signals that a renormizalition should be applied afterwards.
       * @param incrementCapture Constant factor which is added to the checks below threshCheck
       * @param threshCapture Probability threshold for capture moves
       * @return bool
       */
     inline bool enhance_captures(const float incrementCapture, float threshCapture);

public:

    Node(Board *pos,
         Node *parentNode,
         unsigned int childIdxForParent,
         SearchSettings* searchSettings);

    /**
     * @brief Node Copy constructor which copies the value evaluation, board position, prior policy and checkmateIdx.
     * The qValues, actionValues and visits aren't copied over.
     * @param b Node from which the stats will be copied
     */
    Node(const Node& b);
    ~Node();

    DynamicVector<float> getPVecSmall() const;
    void setPVecSmall(const DynamicVector<float> &value);
    std::vector<Move> getLegalMoves() const;
    void setLegalMoves(const std::vector<Move> &value);
    void apply_virtual_loss_to_child(unsigned int childIdx);
    float getValue() const;
    void setValue(float value);
    size_t select_child_node();
    Node* select_node();

    /**
     * @brief get_child_node Returns the child node at the given index.
     * A nullptr is returned if the child node wasn't expanded yet and no check is done if the childIdx is smaller than
     * @param childIdx Index for the next child node to select
     * @return child node
     */
    Node* get_child_node(size_t childIdx);

    /**
     * @brief backup_value Iteratively backpropagates a value prediction across all of the parents for this node.
     * The value is flipped at every ply.
     * @param value Value evaluation to backup, this is the NN eval in the general case or can be from a terminal node
     */
    void backup_value(unsigned int childIdx, float value);

    /**
     * @brief revert_virtual_loss_and_update Revert the virtual loss effect and apply the backpropagated value of its child node
     * @param child_idx Index to the child node to update
     * @param value Specifies the value evaluation to backpropagate
     */
    void revert_virtual_loss_and_update(unsigned int child_idx, float value);

    /**
     * @brief backup_collision Iteratively removes the virtual loss of the collision event that occured
     * @param childIdx Index to the child node to update
     */
    void backup_collision(unsigned int childIdx);

    /**
     * @brief revert_virtual_loss Reverts the virtual loss for a target node
     * @param child_idx Index to the child node to update
     */
    void revert_virtual_loss(unsigned int childIdx);

    /**
     * @brief make_to_root Makes the node to the current root node by setting its parent to a nullptr
     */
    void make_to_root();

    /**
      * @brief enhance_moves Calls enhance_checks & enchance captures if the searchSetting suggests it and applies a renormilization afterwards
      * @param threshCheck Threshold probability for checking moves
      * @param checkFactor Factor based on the maximum probability with which checks will be increased
      * @param threshCapture Threshold probability for capture moves
      * @param captureFactor Factor based on the maximum probability with which captures will be increased
      */
    void enhance_moves(const float threshCheck = 0.1f, const float checkFactor=0.5f, const float threshCapture = 0.1f, const float captureFactor=0.05f);

    friend class SearchThread;
    friend class MCTSAgent;
    friend bool operator> (const Node& n1, const Node& n2);
//    friend bool operator<= (const Node& n1, const Node& n2);

//    friend bool operator< (const Node& n1, const Node& n2);
//    friend bool operator>= (const Node& n1, const Node& n2);
    inline double get_score_value() const;

    DynamicVector<float> getPolicyProbSmall();
    void setPolicyProbSmall(const DynamicVector<float> &value);

    /**
     * @brief get_mcts_policy Returns the final policy after the mcts search which is used for move selection, in most cases argmax(mctsPolicy).
     * Depending on the searchSettings, Q-values will be taken into account for creating this.
     * @param mctsPolicy Output of the final mcts policy after search
     */
    void get_mcts_policy(DynamicVector<float>& mctsPolicy);
    DynamicVector<float> getQValues() const;

    /**
     * @brief apply_dirichlet_noise_to_prior_policy Applies dirichlet noise of strength searchSettings->dirichletEpsilon with
     * alpha value searchSettings->dirichletAlpha to the prior policy of the root node. This encourages exploration of nodes with initially low
     * low activations.
     */
    void apply_dirichlet_noise_to_prior_policy();

    void setQValues(const DynamicVector<float> &value);
    DynamicVector<float> getChildNumberVisits() const;
    unsigned int getNbDirectChildNodes() const;
    Board* getPos();

    /**
     * @brief delete_subtree Deletes the node itself and all existing nodes in its subtree.
     * @param node ode of the subtree to delete
     */
    static void delete_subtree(Node *node);

    /**
     * @brief delete_subtree Deletes the node itself and its pointer in the hashtable as well as all existing nodes in its subtree.
     * @param node Node of the subtree to delete
     * @param hashTable Pointer to the hashTable which stores a pointer to all active nodes
     */
    static void delete_subtree_and_hash_entries(Node *node, unordered_map<Key, Node*>* hashTable);


    /**
     * @brief delete_sibling_subtrees Deletes all subtrees from all simbling nodes, deletes their hash table entry and sets the visit access to nullptr
     * @param hashTable Pointer to the hashTables
     */
    void delete_sibling_subtrees(unordered_map<Key, Node*>* hashTable);

    int getNumberVisits() const;

    /**
     * @brief get_principal_variation Traverses the tree using the get_mcts_policy() function until a leaf or terminal node is found.
     * The moves a are pushed into the pv vector.
     * @param pv Vector in which moves will be pushed.
     */
    void get_principal_variation(std::vector<Move>& pv);

    /**
     * @brief hash_key Returns the hash key of its corresponding position
     * @return
     */
    Key hash_key();
    static void setSearchSettings(SearchSettings *value);

    /**
     * @brief find_move_idx Returns the index of the required move in its move list.
     * -1 is returned if the move wasn't found.
     * @param m Move that is searched for
     * @return Respective child index for the given move in its move list
     */
    int find_move_idx(Move m);

    /**
     * @brief sort_nodes_by_probabilities Sorts all child nodes in ascending order based on their probability value
     */
    void sort_nodes_by_probabilities();

    void assign_values_to_child_nodes();

    std::vector<Node *> getChildNodes() const;
    double getProbValue() const;
    double getQValue() const;
    double getActionValue() const;
    double getVisits() const;
    Move getMove() const;

    /**
     * @brief print_node_statistics Prints all node statistics of the child nodes to stdout
     */
    void print_node_statistics();

    void fill_child_node_moves();

};

/**
 * @brief operator << Overload of stdout operator. Prints move, number visits, probability Value and Q-value
 * @param os ostream handle
 * @param node Node object to print
 * @return ostream
 */
extern std::ostream& operator<<(std::ostream& os, Node* node);

#endif // NODE_H
