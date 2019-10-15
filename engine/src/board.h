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
 * @file: board.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Extension of Stockfish's board presentation by introducing new functionality.
 */

#ifndef BOARD_H
#define BOARD_H

#include <position.h>

class Board : public Position
{
public:
    Board();
    Board(const Board& b);
    ~Board();

    Bitboard promoted_pieces() const;
    int get_pocket_count(Color c, PieceType pt) const;
    Key hash_key() const;
    void setStateInfo(StateInfo* st);
    StateInfo* getStateInfo() const;

    Board& operator=(const Board &b);
    int plies_from_null();

    /**
     * @brief total_move_cout Returns the current full move counter
     * @return Total move number
     */
    size_t total_move_cout();
};

/**
 * @brief is_pgn_move_ambiguous This method is used to detect possible ambiguities for a move for pgn conversion.
 * There are several cases of ambiguity in pgn motivation which need to be resolved.
 * The ambiguity occurs if a same possible legal destination squares exists for different pieces of the same type.
 * 1) different origin squares (common, e.g. knight moves, resolved by adding rank information)
 * 2) same origin rank but different origin files (common, e.g. rook moves, resolved by adding origin file information)
 * 3) same origin file but different origin ranks (rare, e.g. rook moves, resolved by adding origin rank information)
 * 4) same origin file and origin rank shared by at least three pieces of same type (very rare, e.g. queen with promoted queens,
 * resolved by adding full origin square)
 * @param m Move of interest
 * @param pos Board position
 * @param legalMoves List of legal moves for the board position
 * @param file_ambiguous Returns if the move is file ambiguous (two pieces share the same file with same destination square)
 * @param rank_ambiguous Returns if the move is rank ambiguous (two pieces share the same rank with same destination square)
 * @return True, in case of ambiguity else false
 */
bool is_pgn_move_ambiguous(Move m, const Board& pos, const std::vector<Move>& legalMoves, bool& file_ambiguous, bool& rank_ambiguous);

/**
 * @brief pgnMove Converts a given move into PGN move notation.
 * Passing the board position is needed to check for capture and possible ambiguity.
 * The ambiguity check requires the need to call generate_legal_moves()
 * @param m Move
 * @param chess960 True if 960 mode
 * @param pos Board position
 * @param legalMoves List of legal moves in the position (avoid regneration in case it has already been done)
 * @param leadsToTerminal True if the given move leads to a terminal state
 * @return String representation of move in PGN format
 */
std::string pgnMove(Move m, bool chess960, const Board& pos, const std::vector<Move>& legalMoves, bool leadsToTerminal=false);

#endif // BOARD_H
