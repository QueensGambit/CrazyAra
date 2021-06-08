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

#ifndef MODE_POMMERMAN
#include <string>
#include <cctype>
#include "types.h"

using namespace std;

/**
 * @brief get_origin_square Returns the origin square for a valid uciMove
 * @param uciMove uci-Move in string notation
 * @return origin square
 */
Square get_origin_square(const string& uciMove);

/**
 * @brief get_origin_square Returns the destination square for a valid uciMove
 * @param uciMove uci-Move in string notation
 * @return destination square
 */
Square get_destination_square(const string& uciMove);

/**
 * @brief is_drop_move Checks if the given uciMove is a dropping move.
 * @param uciMove Valid uci string including crazyhouse dropping moves.
 * It's assumed that pawn drops have a `P` as a prefix.
 * @return Bool
 */
bool is_drop_move(const string& uciMove);

/**
 * @brief is_promotion_move Checks if the given uciMove is a promition move based on the string length
 * @param uciMove Valid uci string including crazyhouse dropping moves.
 * @return Bool
 */
bool is_promotion_move(const string& uciMove);

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
 * @brief fill_en_passent_moves Fills the given vector with all en-passent candidate moves.
 * @param enPassentMoves Empty std vector which will be filled.
 */
vector<string> create_en_passent_moves();

/**
 * https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipVertically
 * "Using the x86-64 _byteswap_uint64 or bswap64 intrinsics only takes one processor instruction in 64-bit mode.
 * Flip a bitboard vertically about the centre ranks.
 * Rank 1 is mapped to rank 8 and vice versa."
 * @param x any bitboard
 * @return bitboard x flipped vertically
 */
Bitboard flip_vertical(Bitboard x);

/**
 * @brief mirror_move Mirrors a move in uci representation
 * @param moveUCI String uci
 * @return Returns corresponding mirrored uci string with the rank flipped
 */
string mirror_move(const string& uciMove);

/**
 * @brief make_move Creates a move in coordinate representation given an uci string.
 *                  Multiple sf moves are returned in case it's ambigious such as castling or en-passent moves.
 * @param uciMove Valid uci string including crazyhouse dropping moves
 * @param is960 Indicates if the move is played in the 960 game type which triggers a different handling for castling moves
 * @return Move in coordinate representation
 */
vector<Move> make_move(const string& uciMove, bool is960);

/**
 * @brief create_castling_moves Creates a vector for all available castling moves in UCI representation
 * @param is960 Boolean indicating if castling moves for chess960 shall be generated
 * @return vector of castling moves in uci-string representation
 */
vector<string> create_castling_moves(bool is960);

/**
 * @brief handle_classical_castling_moves If the given uciMove is in {"e1g1", "e1c1", "e8g8" "e8c8"}, then the corresponding castling move
 *                                        will be added to the given move vector.
 * @param uciMove Valid uci string including crazyhouse dropping moves
 * @param moveVector Current move vector to which a move entry might be appended
 */
void handle_classical_castling_moves(const string& uciMove, vector<Move>& moveVector);

/**
 * @brief is_960_castling_candidate_move Checks if the given uci move is a potential castling move
 * @param origin Origin move square
 * @param destination Destination move square
 * @return boolean
 */
bool is_960_castling_candidate_move(Square origin, Square destination);

// "An 8x8 Board with a rank-file mapping, needs to perform an exclusive or with 56 (A8 in LERF)"
// https://www.chessprogramming.org/Vertical_Flipping
constexpr Square vertical_flip(Square s) {
  return Square(int(s) ^ 56); // Vertical flip SQ_A1 -> SQ_A8
}

#endif // SFUTIL_H
#endif
