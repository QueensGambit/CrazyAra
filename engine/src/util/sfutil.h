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
 * @file: sfutil.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Additional utility functions for the Stockfish library.
 */

#ifndef SFUTIL_H
#define SFUTIL_H

#include <string>
#include <cctype>
#include "types.h"

/**
 * @brief get_origin_square Returns the origin square for a valid uciMove
 * @param uciMove uci-Move in string notation
 * @return origin square
 */
Square get_origin_square(std::string& uciMove);

/**
 * @brief get_origin_square Returns the destination square for a valid uciMove
 * @param uciMove uci-Move in string notation
 * @return destination square
 */
Square get_destination_square(std::string& uciMove);

/**
 * @brief is_drop_move Checks if the given uciMove is a dropping move.
 * @param uciMove Valid uci string including crazyhouse dropping moves.
 * It's assumed that pawn drops have a `P` as a prefix.
 * @return Bool
 */
bool is_drop_move(std::string& uciMove);

/**
 * @brief is_promotion_move Checks if the given uciMove is a promition move based on the string length
 * @param uciMove Valid uci string including crazyhouse dropping moves.
 * @return Bool
 */
bool is_promotion_move(std::string& uciMove);

/**
 * @brief is_en_passent_candidate Return true, if the given uci-move might be an en-passent capture.
 * En-passent captures can occur from the 5th rank in the view of white and from the 4th rank for black
 * each square has two ways to capture except the border square with only one capture.
 * This function assumes that the given squares origin and destination are based on a valid uci-move.
 * @param origin Origin square of the move
 * @param destination Destination square of the move
 * @return Boolean
 */
bool is_en_passent_candidate(Square origin, Square destination);

/**
 * @brief make_move Creates a move in coordinate representation given an uci string.
 *                  Multiple sf moves are returned in case it's ambigious such as castling or en-passent moves.
 * @param uciMove Valid uci string including crazyhouse dropping moves
 * @return Move in coordinate representation
 */
std::vector<Move> make_move(std::string uciMove);

/**
 * @brief fill_en_passent_moves Fills the given vector with all en-passent candidate moves.
 * @param enPassentMoves Empty std vector which will be filled.
 */
void fill_en_passent_moves(std::vector<std::string> &enPassentMoves);

/**
 * https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipVertically
 * ! NOT USED ATM BECAUSE INDIVIDUAL BIT FLIPPING WAS SLIGHTLY FASTER !
 * Flip a bitboard vertically about the centre ranks.
 * Rank 1 is mapped to rank 8 and vice versa.
 * @param x any bitboard
 * @return bitboard x flipped vertically
 */
Bitboard flip_vertical(Bitboard x);

/**
 * @brief mirror_move Mirrors a move in uci representation
 * @param moveUCI String uci
 * @return Returns corresponding mirrored uci string with the rank flipped
 */
std::string mirror_move(std::string moveUCI);

/**
 * @brief make_move Creates a move in coordinate representation given an uci string.
 *                  Multiple sf moves are returned in case it's ambigious such as castling or en-passent moves.
 * @param uciMove Valid uci string including crazyhouse dropping moves
 * @return Move in coordinate representation
 */
std::vector<Move> make_move(std::string uciMove);

// "An 8x8 Board with a rank-file mapping, needs to perform an exclusive or with 56 (A8 in LERF)"
// https://www.chessprogramming.org/Vertical_Flipping
constexpr Square vertical_flip(Square s) {
  return Square(s ^ 56); // Vertical flip SQ_A1 -> SQ_A8
}

#endif // SFUTIL_H
