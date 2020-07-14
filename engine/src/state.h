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
  GNU General Public License fåor more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: state.h
 * Created on 13.07.2020
 * @author: queensgambit
 *
 * State is an abstract class which is used in the MCTS as a generic interface for various environments.
 */

#ifndef GAMESTATE_H
#define GAMESTATE_H

#include <vector>
#include <string>
#include "types.h"


using namespace std;

typedef uint64_t Key;
typedef int Action;
const int ACTION_NONE = 0;

enum TerminalType {
    TERMINAL_LOSS,
    TERMINAL_DRAW,
    TERMINAL_WIN,
    TERMINAL_NONE
};

class State
{
public:

    /**
     * @brief legal_actions Returns all legal actions as a vector list
     * @return vector of legal actions
     */
    virtual vector<Action> legal_actions() const = 0;

    /**
     * @brief set Sets a new states and modifies the current state.
     * @param fenStr String description about the state
     * @param isChess960 If true 960 mode will be active
     * @param variant Variant which the position corresponds to.
     * @return An alias to the updated state
     */
    virtual State& set(const string& fenStr, bool isChess960, int variant) = 0;

    /**
     * @brief get_state_planes Returns the state plane representation of the current state which can be used for NN inference.
     * @param normalize If true thw normalized represnetation should be returned, otherwise the raw representation
     * @param inputPlanes Pointer to the memory array where to set the state plane representation. It is assumed that the memory has already been allocated
     */
    virtual void get_state_planes(bool normalize, float* inputPlanes) const = 0;

    /**
     * @brief steps_from_null Number of steps form the initial position (e.g. starting position)
     * @return number of steps
     */
    virtual unsigned int steps_from_null() const = 0;

    /**
     * @brief is_chess960 Returns true if the position is a 960 random position, else false
     * @return bool
     */
    virtual bool is_chess960() const = 0;

    /**
     * @brief fen Returns the fen or string description of the current state
     * @return string
     */
    virtual string fen() const = 0;

    /**
     * @brief do_action Applies a given action to the current state
     * @param action Type of action to apply. It is assumed that the action is discrete and integer format
     */
    virtual void do_action(Action action) = 0;

    /**
     * @brief number_repetitions Returns the number of times this state has already occured in the current episode
     * @return int
     */
    virtual unsigned int number_repetitions() const = 0;

    /**
     * @brief side_to_move Returns the side to move (e.g. Color: WHITE or BLACK) in chess
     * @return int
     */
    virtual int side_to_move() const = 0;

    /**
     * @brief hash_key Returns a uique identifier for the current position which can be used for accessing the hash table
     * @return
     */
    virtual Key hash_key() const = 0;

    /**
     * @brief flip Flips the state along the x-axis
     */
    virtual void flip() = 0;

    /**
     * @brief uci_to_action Converts the given action in uci notation to an action object
     * @param uciStr uci specification for the action
     * @return Action
     */
    virtual Action uci_to_action(string& uciStr) const = 0;

    /**
     * @brief action_to_san Converts a given action to SAN (pgn move notation) usign the current position and legal moves
     * @param action Given action
     * @param legalActions List of legal moves for the current position
     * @return SAN string
     */
    virtual string action_to_san(Action action, const vector<Action>& legalActions) const = 0;

    /**
     * @brief is_terminal Returns the terminal type for the current state. If the state is a non terminal state,
     * then TERMINAL_NONE should be returned.
     * @param numberLegalMoves Number of legal moves in the current position
     * @param inCheck Boolean which defines if there is a check in the current position
     * @return TerminalType
     */
    virtual TerminalType is_terminal(size_t numberLegalMoves, bool inCheck) const = 0;

    /**
     * @brief gives_check Checks if the current action is a checking move
     * @param action Action
     * @return bool
     */
    virtual bool gives_check(Action action) const = 0;

    /**
     * @brief clone Clones the current state as a deep copy
     * @return deep copy
     */
    virtual unique_ptr<State> clone() const = 0;
};



#endif // GAMESTATE_H
