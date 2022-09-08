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


using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;
using ChildIdx = uint_fast16_t;
using Budget = uint_fast16_t;

struct NodeAndIdx {
    Node* node;
    uint16_t childIdx;
    NodeAndIdx(Node* node, uint16_t childIdx) :
        node(node), childIdx(childIdx) {}
};
using Trajectory = vector<NodeAndIdx>;
using HashMap = unordered_map<Key, weak_ptr<Node>> ;
// wrapper for unordered_map with a mutex for thread safe access
struct MapWithMutex {
    mutex mtx;
    HashMap hashTable;
    ~MapWithMutex() {
    }
};


struct NodeSplit {
    ChildIdx firstArg;
    ChildIdx secondArg;
    Budget firstBudget;
    Budget secondBudget;

    inline void only_first(ChildIdx firstArg, uint_fast16_t budget) {
        this->firstArg = firstArg;
        firstBudget = budget;
        secondBudget = 0;
    }
};

struct NodeAndBudget {
    Node* node;
    uint_fast16_t budget;
    StateObj* curState;
    Trajectory curTrajectory;
    NodeAndBudget(Node* node, uint_fast16_t budget, StateObj* state) :
        node(node), budget(budget), curState(state) {}
};

class Node
{
private:
    mutex mtx;

    DynamicVector<float> policyProbSmall;
    vector<Action> legalActions;
    Key key;

    // singular values
    // valueSum stores the sum of all incoming value evaluations
    double valueSum;

    unique_ptr<NodeData> d;
#ifdef MCTS_STORE_STATES
    unique_ptr<StateObj> state;
#endif

    uint32_t realVisitsSum;

    // identifiers
    uint16_t pliesFromNull;

    uint16_t numberParentNodes;
    bool isTerminal;
    bool isTablebase;
    bool hasNNResults;
    bool sorted;

public:
    /**
     * @brief Node Primary constructor which is used when expanding a node during search
     * @param State Corresponding state object
     * @param searchSettings Pointer to the searchSettings
     */
    Node(StateObj *state,
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
    Node* get_child_node(ChildIdx childIdx);

    ChildIdx select_child_node(const SearchSettings* searchSettings);

    /**
     * @brief select_child_nodes Selects multiple nodes at once
     * @param searchSettings Search settings struct
     * @param budget How many simulations are still available
     * @return Struct on how the selection was split
     */
    NodeSplit select_child_nodes(const SearchSettings* searchSettings, uint_fast16_t budget);

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
     * @param childIdx Index to the child node to update
     * @param value Specifies the value evaluation to backpropagate
     * @param solveForTerminal Decides if the terminal solver will be used
     */
    template<bool freeBackup>
    void revert_virtual_loss_and_update(ChildIdx childIdx, float value, float virtualLoss, bool solveForTerminal)
    {
        lock();
        // decrement virtual loss counter
        update_virtual_loss_counter<false>(childIdx, virtualLoss);

        valueSum += value;
        ++realVisitsSum;

        if (d->childNumberVisits[childIdx] == virtualLoss) {
            // set new Q-value based on return
            // (the initialization of the Q-value was by Q_INIT which we don't want to recover.)
            d->qValues[childIdx] = value;
        }
        else {
            // revert virtual loss and update the Q-value
            assert(d->childNumberVisits[childIdx] != 0);
            d->qValues[childIdx] = (double(d->qValues[childIdx]) * d->childNumberVisits[childIdx] + virtualLoss + value) / d->childNumberVisits[childIdx];
            assert(!isnan(d->qValues[childIdx]));
        }

        if (virtualLoss != 1) {
            d->childNumberVisits[childIdx] -= size_t(virtualLoss) - 1;
            d->visitSum -= size_t(virtualLoss) - 1;
        }
        if (freeBackup) {
            ++d->freeVisits;
        }
        if (solveForTerminal) {
            solve_for_terminal(childIdx);
        }
        unlock();
    }

    /**
     * @brief revert_virtual_loss Reverts the virtual loss for a target node
     * @param childIdx Index to the child node to update
     */
    void revert_virtual_loss(ChildIdx childIdx, float virtualLoss);

    bool is_playout_node() const;

    /**
     * @brief is_blank_root_node Returns true if the node is a blank root node with no visits
     * @return True if initialized but no visits else false
     */
    bool is_blank_root_node() const;
    bool is_solved() const;
    bool has_forced_win() const;

    Action get_action(ChildIdx childIdx) const;
    Node* get_child_node(ChildIdx childIdx) const;
    shared_ptr<Node> get_child_node_shared(ChildIdx childIdx) const;

    vector<shared_ptr<Node>>::const_iterator get_node_it_begin() const;
    vector<shared_ptr<Node>>::const_iterator get_node_it_end() const;


    bool is_terminal() const;
    bool has_nn_results() const;
    float get_value() const;

    /**
     * @brief get_value_display Return value evaluation which can be used for logging
     * Warning: Must be called with d != nullptr
     * @return value() or pre-defined constant
     */
    float get_value_display() const;

    double get_value_sum() const;
    uint32_t get_real_visits() const;

    void apply_virtual_loss_to_child(ChildIdx childIdx, uint_fast32_t virtualLoss);

    void increment_no_visit_idx();
    void fully_expand_node();

    Key hash_key() const;

    size_t get_number_child_nodes() const;

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
    uint32_t get_real_visits(ChildIdx childIdx) const;

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

    uint16_t get_no_visit_idx() const;

    bool is_fully_expanded() const;

    DynamicVector<float>& get_policy_prob_small();

    void set_probabilities_for_moves(const float *data, bool mirrorPolicy);

    void apply_softmax_to_policy();

    /**
     * @brief enhance_moves Calls enhance_checks & enhance captures if the searchSetting suggests it and applies a renormilization afterwards
     * @param pos Current board position
     */
    void enhance_moves(const SearchSettings* searchSettings);

    void set_value(float value);
    uint16_t main_child_idx_for_parent() const;

    /**
     * @brief add_new_node_to_tree Checks if the given position already exists in the Hash map.
     * If so, connect the parent to this node. Otherwise create a new node.
     * @param mapWithMutex Hash map with mutex
     * @param newState Corresponding state
     * @param childIdx Child index
     * @param searchSettings Search Settings struct
     * @param transposition Return true, if the transposition request was successfull, else false, i.e. a new node was added
     * @return the newly added node
     */
    Node* add_new_node_to_tree(MapWithMutex* mapWithMutex, StateObj* newState, ChildIdx childIdx, const SearchSettings* searchSettings, bool& transposition);

    void add_transposition_parent_node();

    /**
     * @brief max_prob Returns the maximum policy value
     * @return float
     */
    float max_policy_prob();

    /**
     * @brief max_q_child Returns the child index with the highest Q-value
     * @return size_t
     */
    ChildIdx max_q_child() const;

    /**
     * @brief max_visits_child Returns the child index with the most visits
     * @return size_t
     */
    ChildIdx max_visits_child() const;

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
     * @param mctsPolicy Output of the final mcts policy after search
     * @param bestMoveIdx Index for the best move
     * @param qValueWeight Decides if Q-values are taken into account
     * @param qVetoDelta Describes how much better the highest Q-Value has to be to replace the candidate move with the highest visit count
     */
     void get_mcts_policy(DynamicVector<double>& mctsPolicy, ChildIdx& bestMoveIdx, float qValueWeight, float qVetoDelta) const;

    /**
     * @brief get_principal_variation Traverses the tree using the get_mcts_policy() function until a leaf or terminal node is found.
     * The moves a are pushed into the pv vector.
     * @param pv Vector in which moves will be pushed.
     * @param qValueWeight Decides if Q-values are taken into account
     * @param qVetoDelta Describes how much better the highest Q-Value has to be to replace the candidate move with the highest visit count
     */
     void get_principal_variation(vector<Action>& pv, float qValueWeight, float qVetoDelta);

    /**
     * @brief is_root_node Checks if the current node is the root node
     * @return true if root node else false
     */
    bool is_root_node() const;

    DynamicVector<uint32_t> get_child_number_visits() const;
    uint32_t get_child_number_visits(ChildIdx childIdx) const;

    void enable_has_nn_results();
    uint16_t plies_from_null() const;
    bool is_tablebase() const;
    NodeType get_node_type() const;
    uint16_t get_end_in_ply() const;
    uint32_t get_free_visits() const;

    void init_node_data(size_t numberNodes);
    void init_node_data();

    void mark_as_terminal();

    bool is_sorted() const;

    /**
     * @brief get_q_value Returns the Q-value for the given child index
     * @param idx Child Index
     * @return Q-value
     */
    float get_q_value(ChildIdx idx) const;

    /**
     * @brief get_q_values Returns the Q-values for all child nodes
     * @return Q-values
     */
    DynamicVector<float> get_q_values() const;

    /**
     * @brief set_q_value Sets a Q-value for a given child index
     * @param idx Child index
     * @param value value to set
     */
    void set_q_value(ChildIdx idx, float value);

    /**
     * @brief get_best_q_idx Return the child index with the highest Q-value
     * @return Index of child with maximum Q-value
     */
    ChildIdx get_best_q_idx() const;

    /**
     * @brief get_q_idx_over_thresh Returns all child node which coresponding Q-values are greater than qThresh
     * @param qThresh Threshold
     * @return vector of child indices
     */
    vector<ChildIdx> get_q_idx_over_thresh(float qThresh);

    /**
     * @brief print_node_statistics
     * @param pos Position object related to the current position.
     *  If the position is given as "nulltptr" the moves will be displayed in UCI notation instead of SAN.
     * @param customOrdering Optional custom ordering of how the moves shall be displayed (e.g. according to the MCTS policy after search).
     *  If an empty vector is given, it will use the current ordering of the child nodes (by default according to the prior policy).
     */
    void print_node_statistics(const StateObj* pos, const vector<size_t>& customOrdering) const;

    /**
     * @brief get_node_count Returns the number of nodes in the subgraph of this nodes without counting terminal simulations
     * @return uint32_t
     */
    uint32_t get_node_count() const;

    bool is_transposition() const;

    /**
     * @brief decrement_number_parents Decrements the counter of the number of parent nodes.
     * This is needed for memory clearing to avoid double free.
     */
    void decrement_number_parents();

    double get_q_sum(ChildIdx childIdx, float virtualLoss) const;

    template<bool increment>
    void update_virtual_loss_counter(ChildIdx childIdx, float virtualLoss)
    {
        if (increment) {
            d->virtualLossCounter[childIdx] += virtualLoss;
        }
        else {
            assert(d->virtualLossCounter[childIdx] != 0);
            d->virtualLossCounter[childIdx] -= virtualLoss;
        }
    }

    uint8_t get_virtual_loss_counter(ChildIdx childIdx) const;

    bool has_transposition_child_node();

    bool is_transposition_return(double myQvalue) const;

    void set_checkmate_idx(ChildIdx childIdx) const;

    /**
     * @brief was_inspected Returns true if the node has already been inspected for e.g. checks.
     * @return bool
     */
    bool was_inspected();

    /**
     * @brief set_as_inspected Sets the inspected variable to true
     */
    void set_as_inspected();

#ifdef MCTS_STORE_STATES
    StateObj* get_state() const;

    /**
     * @brief set_auxiliary_outputs Sets the auxiliary outputs of the neural network to the state object
     * @param auxiliaryOutputs Auxiliary outputs of the neural network for the corresponding state
     */
    void set_auxiliary_outputs(const float* auxiliaryOutputs);
#endif

    uint32_t get_number_of_nodes() const;

private:
    /**
     * @brief reserve_full_memory Reserves memory for all available child nodes
     */
    void reserve_full_memory();

    /**
     * @brief check_for_terminal Checks if the given board position is a terminal node and updates isTerminal
     * @param state Current board position for this node
     */
    void check_for_terminal(StateObj* state);

#ifdef MCTS_TB_SUPPORT
    /**
     * @brief check_for_tablebase_wdl Checks if the given board position is a tablebase position and
     *  updates isTerminal and the value evaluation
     * @param state Current board position for this node
     */
    void check_for_tablebase_wdl(StateObj* state);

    void mark_as_tablebase();
#endif

    /**
     * @brief solve_for_terminal Tries to solve the current node to be a forced win, loss or draw.
     * The main idea is based on the paper "Exact-Win Strategy for Overcoming AlphaZero" by Chen et al.
     * https://www.researchgate.net/publication/331216459_Exact-Win_Strategy_for_Overcoming_AlphaZero
     * The solver uses the current backpropagating child node as well as all available child nodes.
     * @param childNode Child nodes which backpropagates the value
     * @return true, if the node type of the current node was modified
     */
    bool solve_for_terminal(ChildIdx childIdx);

    /**
     * @brief solved_win Checks if the current node is a solved win based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for WIN else false
     */
    bool solved_win(const Node* childNode) const;

    /**
     * @brief solved_draw Checks if the current node is a solved draw based on the given child node
     * and all available child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for DRAW else false
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
     * @brief only_child_nodes_of_one_kind Check if all expanded child nodes are of the same kind.
     * @return true if only child nodes of type <nodeType> exist else false
     */
    template <NodeType nodeType>
    bool only_child_nodes_of_one_kind() const
    {
        for (auto it = d->childNodes.begin(); it != d->childNodes.end(); ++it) {
            const Node* childNode = it->get();
            if (childNode->d->nodeType != nodeType) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief solved_loss Checks if the current node is a solved loss based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for LOSS else false
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

#ifdef MCTS_TB_SUPPORT
    /**
     * @brief solve_tb_win Checks if the current node is a solved tablebase win based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for TB_WIN else false
     */
    bool solve_tb_win(const Node* childNode) const;

    /**
     * @brief solved_tb_draw Checks if the current node is a solved tablebase draw based on the given child node
     * and all available child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for TB_DRAW else false
     */
    bool solved_tb_draw(const Node* childNode) const;

    /**
     * @brief solved_tb_loss Checks if the current node is a solved tablebase loss based on the given child node
     * @param childNode Child nodes which backpropagates the value
     * @return true for TB_LOSS else false
     */
    bool solved_tb_loss(const Node* childNode) const;

    /**
     * @brief only_won_tb_child_nodws Checks if this node has only WON child nodes
     * @return true if only WIN_TB child nodes exist else false
     */
    bool only_won_tb_child_nodes() const;

    /**
     * @brief mark_as_tb_loss Marks the current node as a tablebase loss
     */
    void mark_as_tb_loss();

    /**
     * @brief mark_as_tb_draw Marks the current node as a tablebase draw
     */
    void mark_as_tb_draw();

    /**
     * @brief mark_as_tb_win Marks the current node as a tablebase win
     */
    void mark_as_tb_win();
#endif

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
    void update_solved_terminal(const Node* childNode, uint_fast16_t childIdx);

    /**
     * @brief mcts_policy_based_on_wins Sets all known winning moves in a given policy to 1 and all
     * remaining moves to 0.
     * @param mctsPolicy MCTS policy which will be set
     */
    void mcts_policy_based_on_wins(DynamicVector<double>& mctsPolicy) const;

    /**
     * @brief mcts_policy_based_on_losses Sets the policy entry which delays the mate the longest to 1 and remaining values to 0.
     * @param mctsPolicy MCTS policy which will be set
     */
    void mcts_policy_based_on_losses(DynamicVector<double>& mctsPolicy) const;

    /**
     * @brief prune_losses_in_mcts_policy Sets all known losing moves in a given policy to 0 in case
     * the node is not known to be losing.
     * @param mctsPolicy MCTS policy which will be set
     */
    void prune_losses_in_mcts_policy(DynamicVector<double>& mctsPolicy) const;

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
};

/**
 * @brief get_best_action_index Returns the best move index of all available moves based on the mcts policy
 * or solved wins / draws / losses.
 * @param curNode Current node
 * @param fast If true, then the argmax(childNumberVisits) is returned for unsolved nodes
 * @param qValueWeight Decides if qValues are taken into account
 * @param qVetoDelta Describes how much better the highest Q-Value has to be to replace the candidate move with the highest visit count
 * @return Index for best move and child node
 */
 size_t get_best_action_index(const Node* curNode, bool fast, float qValueWeight, float qVetoDelto);

typedef float (* vFunctionValue)(Node* node);
DynamicVector<float> retrieve_dynamic_vector(const vector<Node*>& childNodes, vFunctionValue func);

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
 * @brief flip_node_type Flips the node type value (e.g. WIN into LOSS)
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
 * @brief backup_collision Iteratively removes the virtual loss of the collision event that occurred
 * @param rootNode Root node of the tree
 * @param virtualLoss Virtual loss value
 * @param trajectory Trajectory on how to get to the given collision
 */
void backup_collision(float virtualLoss, const Trajectory& trajectory);

float get_transposition_q_value(uint_fast32_t transposVisits, double transposQValue, double masterQValue);

/**
 * @brief backup_value Iteratively backpropagates a value prediction across all of the parents for this node.
 * The value is flipped at every ply.
 * @param rootNode Root node of the tree
 * @param value Value evaluation to backup, this is the NN eval in the general case or can be from a terminal node
 * @param virtualLoss Virtual loss value
 * @param trajectory Trajectory on how to get to the given value eval
 * @param solveForTerminal Decides if the terminal solver will be used
 */
template <bool freeBackup>
void backup_value(float value, float virtualLoss, const Trajectory& trajectory, bool solveForTerminal) {
    double targetQValue = 0;
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        if (targetQValue != 0) {
            const uint_fast32_t transposVisits = it->node->get_real_visits(it->childIdx);
            if (transposVisits != 0) {
                const double transposQValue = -it->node->get_q_sum(it->childIdx, virtualLoss) / transposVisits;
                value = get_transposition_q_value(transposVisits, transposQValue, targetQValue);
            }
        }
#ifndef MCTS_SINGLE_PLAYER
        value = -value;
#endif
        freeBackup ? it->node->revert_virtual_loss_and_update<true>(it->childIdx, value, virtualLoss, solveForTerminal) :
                   it->node->revert_virtual_loss_and_update<false>(it->childIdx, value, virtualLoss, solveForTerminal);

        if (it->node->is_transposition()) {
            targetQValue = it->node->get_value();
        }
        else {
            targetQValue = 0;
        }
    }
}

/**
 * @brief is_transposition_verified Checks if the node and state object are a verified position, i.e. same move counter and node has nn results
 * @param node Node object
 * @param state State object
 * @return True, for verification, else false
 */
bool is_transposition_verified(const Node* node, const StateObj* state);


#endif // NODE_H
