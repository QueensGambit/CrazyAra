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
 * @file: tests.h
 * Created on 18.07.2019
 * @author: queensgambit
 *
 * Main entry point to run all tests.
 */

#ifndef TESTS_H
#define TESTS_H


#ifdef BUILD_TESTS
#include <vector>
#include <string>
#include "stateobj.h"
#ifdef SF_DEPENDENCY
#include "environments/chess_related/board.h"
#endif

using namespace std;

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#ifndef MODE_STRATEGO
/**
 * @brief init Initializes bitboards, bitbases and position arrays
 */
void init();

/**
 * @brief get_planes_statistics Returns numeric statistics about the corresponding input planes of the board position
 * @param pos Board position
 * @param normalize True, if the plane should be normalized
 * @param sum Sum of all numerical values
 * @param maxNum Maximum value of the planes
 * @param key Unique identifier of the plane
 * @param argMax Index with the highest value
 */
void get_planes_statistics(const Board* pos, bool normalize, double& sum, double& maxNum, double& key, size_t& argMax);

/**
 * @brief apply_moves_to_board Applies a list of moves given in uci-notation to a given board
 * @param uciMoves List of UCI-moves (strings)
 * @param pos Board position on which the moves will be applied
 * @param states State position pointer
 */
void apply_moves_to_board(const vector<string>& uciMoves, Board& pos, StateListPtr& states);

/**
 * @brief are_all_entries_true Returns true if all entries of the vector uciMoves return true by calling the function foo.
 * @param uciMoves Vector storing moves in uci move notation
 * @return boolean
 */
bool are_all_entries_true(const vector<string>& uciMoves, bool (*foo)(Square, Square));

/**
 * @brief The GameInfo struct is a return type for apply_random_moves() which provides information about the game.
 */
struct GameInfo {
    uint nbAppliedMoves;
    bool reachedTerminal;
    GameInfo() :
        nbAppliedMoves(0),
        reachedTerminal(false)
        {};
};

/**
 * @brief apply_random_moves Applies random legal moves to the state object and returns a GameInfo struct.
 * If a terminal state is reached the loop stops and GameInfo.givesCheck == true is returned.
 * @param state Starting position
 * @param movesToApply Number of moves to apply.
 * @return GameInfo struct
 */
GameInfo apply_random_moves(StateObj& state, uint movesToApply);

/**
 * @brief apply_given_moves Applies a given set of uci-moves to the states
 * @param state State object
 * @param uciMoves Move to apply
 */
void apply_given_moves(StateObj& state, const std::vector<string>& uciMoves);

/**
 * @brief get_default_variant Returns the default variant for the used build mode
 * @return Variant
 */
Variant get_default_variant();

#ifdef SF_DEPENDENCY
#if !defined(MODE_XIANGQI) && !defined(MODE_BOARDGAMES)
/**
 * @brief is_uci_move_legal Check if a uci move, given as a string, is legal at a specific position
 * @return bool True, if the engine thinks the move is legal
 */
bool is_uci_move_legal(const BoardState& pos, const string& move, bool is960);

/**
 * @brief are_uci_moves_legal_bool Checks if all uci moves are either legal or not
 * @param equals Specifies if the moves have to be legal (true) or not (false) to return true.
 * @return bool True, if all moves equal the parameter 'equals', else false.
 */
bool are_uci_moves_legal_bool(const BoardState& pos, const vector<string>& uciMoves, bool equals, bool is960);

/**
 * @brief legal_actions_equal_ucimoves Checks if the given uci moves is a permutation of the legal actions
 */
bool legal_actions_equal_ucimoves(const BoardState& pos, const vector<string>& uciMoves, bool is960);
#endif
#endif

#endif
#endif

#endif // TESTS_H
