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
 * @file: mctsagent.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
 * The final move is chosen according to the visit count of each direct child node and optionally taken Q-values into account.
 * One playout is defined as expanding one new node in the tree.
 * In the case of chess this means evaluating a new board position.
 * For more details and the mathematical equations please refer to our Journal Paper:
 * https://arxiv.org/abs/1908.06660 as well as the official DeepMind-papers.
 */

#ifndef MCTSAGENT_H
#define MCTSAGENT_H

#include <thread>
#include "position.h"
#include "agent.h"
#include "../evalinfo.h"
#include "../node.h"
#include "../board.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "../searchthread.h"
#include "../manager/statesmanager.h"
#include "../manager/timemanager.h"

class MCTSAgent : public Agent
{
private:
    NeuralNetAPI* netSingle;
    NeuralNetAPI** netBatches;

    SearchSettings* searchSettings;
    std::vector<SearchThread*> searchThreads;

    float inputPlanes[NB_VALUES_TOTAL];
    NDArray* valueOutput;
    NDArray* probOutputs;

    TimeManager* timeManager;

    Node* rootNode;
    // The oldes root node stores a reference to the node with with the current root nodes is based on.
    // This is used in the case of tree reusage. The old subtree cannot be cleared immediatly because of
    // stateInfos for 3-fold repetition, but can be cleared as soon as the tree cannot be reused anymore.
    Node* oldestRootNode;

    // stores the pointer to the root node which will become the new root
    Node* ownNextRoot;
    // stores the pointer to the root node which will become the new root for opponents turn
    Node* opponentsNextRoot;

    // this vector contains all nodes which have been played during a game
    vector<Node*> gameNodes;

    MapWithMutex* mapWithMutex;
    StatesManager* states;
    float lastValueEval;

    // boolean which indicates if the same node was requested twice for analysis
    bool reusedFullTree;

    /**
     * @brief reuse_tree Checks if the postion is know and if the tree or parts of the tree can be reused.
     * The old tree or former subtrees will be freed from memory.
     * @param pos Requested board position
     * @return Number of nodes that have already been explored before the serach
     */
    inline size_t init_root_node(Board* pos);

    /**
     * @brief get_new_root_node Returns the pointer of the new root node for the given position in the case
     * it was either the old root node or an element of the potential root node list.
     * Otherwise a nullptr will be returned. The old tree is deleted except the game nodes.
     * @param pos Requested board position
     * @return Pointer to root node or nullptr
     */
    inline Node* get_root_node_from_tree(Board* pos);

    /**
     * @brief stop_search_based_on_limits Checks for the search limit condition and possible early break-ups
     * and stops all running search threads accordingly
     */
    inline void stop_search_based_on_limits();

    /**
     * @brief stop_search Stops all search threads
     */
    inline void stop_search();

    /**
     * @brief check_early_stopping Checks if the search can be ended prematurely based on the current tree statistics (visits & Q-values)
     * @return True, if early stopping is recommended
     */
    inline bool early_stopping();

    /**
     * @brief continue_search Checks if the search should which is based on the initial value prediciton
     * @return True, if search extension is recommend
     */
    inline bool continue_search();

    /**
     * @brief create_new_root_node Creates a new root node for the given board position and requests the neural network for evaluation
     * @param pos Board position
     */
    inline void create_new_root_node(Board *pos);

    /**
     * @brief delete_old_tree Clear the old tree except the gameNodes (rootNode, opponentNextRoot)
     */
    void delete_old_tree();

    /**
     * @brief delete_game_nodes Delete all gameNodes and their hashTable item
     */
    void delete_game_nodes();

public:
    MCTSAgent(NeuralNetAPI* netSingle,
              NeuralNetAPI** netBatches,
              SearchSettings* searchSettings,
              PlaySettings* playSettings_,
              StatesManager* states);

    ~MCTSAgent();

    void evaluate_board_state(Board *pos, EvalInfo& evalInfo);

    /**
     * @brief run_mcts_search Starts the MCTS serach using all available search threads
     */
    void run_mcts_search();

    /**
     * @brief print_root_node Prints out the root node statistics (visits, q-value, u-value)
     *  by calling the stdout operator for the Node class
     * @param ownMove Defines if the move shall be applied to the current root or the first potential root
     */
    void print_root_node();

    /**
     * @brief apply_move_to_tree Applies the given move to the search tree by adding the expanded node to the candidate list
     * @param m Move
     * @param ownMove Boolean indicating if it was CrazyAra's move
     * @param pos Board position that will occur after move is applied
     */
    void apply_move_to_tree(Move move, bool ownMove, Board* pos);

    /**
     * @brief clear_game_history Traverses all root positions for the game and calls clear_subtree() for each of them
     */
    void clear_game_history();

    /**
     * @brief is_policy_map Checks if the current loaded network uses policy map representation.
     * @return True, if policy map else false
     */
    bool is_policy_map();

    /**
     * @brief get_name Returns the name specification of the MCTSAgent using the CrazyAra version ID and loaded neural net
     * @return
     */
    string get_name() const;

    Node *get_opponents_next_root() const;

    Node* get_root_node() const;

    string get_device_name() const;

    float get_dirichlet_noise() const;

    float get_q_value_weight() const;

    /**
     * @brief update_q_value_weight Updates the Q-value weights for the search (used for quick search)
     * @param value New value to set
     */
    void update_q_value_weight(float value);

    /**
     * @brief update_dirichlet_epsilon Updates the amount of dirichlet noise (used for quick search)
     * @param value New value to set
     */
    void update_dirichlet_epsilon(float value);
};

#endif // MCTSAGENT_H
