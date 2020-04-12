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

//#define BUILD_TESTS

#ifdef BUILD_TESTS
#include <vector>
#include <string>
#include "../board.h"

using namespace std;

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

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

#endif

#endif // TESTS_H
