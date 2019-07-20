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
    Board *pos;
    DynamicVector<float> policyProbSmall;
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;

    // dummy
    DynamicVector<float> ones;

    std::vector<Move> legalMoves;
    bool isTerminal;
    unsigned int nbDirectChildNodes;

    float initialValue;
    int numberVisits = 0;
    std::vector<Node*> childNodes;

    Node *parentNode;
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
     * @brief get_current_u_values Calucates anCalucates and returns the current u-values for this node
     * @return DynamicVector<float>
     */
    DynamicVector<float> get_current_u_values();

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
    Node* get_child_node(size_t childIdx);
    void set_child_node(size_t childIdx, Node *newNode);

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
     * @brief enhance_checks Enhances all possible checking moves by min(0.1, 0.5 * max(policyProbSmall)) and applies a renormalization afterwards
     */
     void enhance_checks();

    friend class SearchThread;
    friend class MCTSAgent;

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
     * @brief delete_subtree Deletes the node itself and its pointer in the hashtable as well as all existing nodes in its subtree.
     * @param node Node of the subtree to delete
     * @param hashTable Pointer to the hashTable which stores a pointer to all active nodes
     */
    static void delete_subtree(Node *node, unordered_map<Key, Node*>* hashTable);

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
};

extern std::ostream& operator<<(std::ostream& os, const Node *node);

#endif // NODE_H
