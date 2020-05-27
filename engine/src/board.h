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
 * @file: board.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Extension of Stockfish's board presentation by introducing new functionality.
 */

#ifndef BOARD_H
#define BOARD_H

#include <position.h>
#include <deque>
#include "syzygy/tbprobe.h"
#include "domain/crazyhouse/constants.h"

class Board : public Position
{
private:
#ifdef MODE_CHESS
    // up to NB_LAST_MOVES are stored in a list, most recent moves first
    deque<Move> lastMoves;
    /**
     * @brief add_move_to_list Adds a given move to the move list and removes the
     * last element if the list exceeds NB_LAST_MOVES items
     * @param m Given Move
     */
    inline void add_move_to_list(Move m);
#endif
public:
    Board();
    Board(const Board& b);
    ~Board();

    Bitboard promoted_pieces() const;
    int get_pocket_count(Color c, PieceType pt) const;
    Key hash_key() const;
    void set_state_info(StateInfo* st);
    StateInfo* get_state_info() const;

    Board& operator=(const Board &b);
    int plies_from_null() const;

    /**
     * @brief total_move_cout Returns the current full move counter.
     * In the initial starting position and after first half move of whites, it returns 0
     * @return Total move number
     */
    size_t total_move_cout() const;

    /**
     * @brief number_repetitions Returns how often the position has already occured.
     * Only possible returned values are 0, 1, 2
     * @return number of repetitions
     */
    size_t number_repetitions() const;

    /**
     * @brief is_3fold_repetition Returns true, if the position is a 3-fold-repetition draw else false
     * @return bool
     */
    bool can_claim_3fold_repetition() const;

    /**
     * @brief is_50_move_rule_draw Returns true, if the positions is a draw due to 50 move rule.
     * Method is based on Position::is_draw(int ply).
     * is_50_move_rule_draw() always returns false for the Crazyhouse variant for improved speed
     * @return
     */
    bool is_50_move_rule_draw() const;

    /**
     * @brief is_terminal Checks if move is a terminal based on the number of legal moves
     * @return True for terminal, else false
     */
    bool is_terminal() const;

    /**
     * @brief draw_by_insufficient_material Checks for draws by insufficient material according to FIDE rules:
     * 1) KK
     * 2) KB vs K
     * 3) KN vs K
     * 4) KNN vs K
     * Other draws which are highly likely such as (KN vs KN, KB vs KN, KNN vs KB, KBN vs KB, KBN vs KR, ...)
     * are expected to be handled by tablebases.
     * Reference: https://www.chessprogramming.org/Material
     * @return True, if draws by insufficient material occured
     */
    bool draw_by_insufficient_material() const;

#ifdef MODE_CHESS
    // overloaded function which include a last move list update
    void do_move(Move m, StateInfo& newSt);
    void do_move(Move m, StateInfo& newSt, bool givesCheck);
    void undo_move(Move m);
    void set(const std::string& fenStr, bool isChess960, Variant v, StateInfo* si, Thread* th);
    void set(const std::string& code, Color c, Variant v, StateInfo* si);
    deque<Move> get_last_moves() const;
#endif
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
 * @param isFileAmbiguous Returns if the move is file ambiguous (two pieces share the same file with same destination square)
 * @param isRankAmbiguous Returns if the move is rank ambiguous (two pieces share the same rank with same destination square)
 * @return True, in case of ambiguity else false
 */
bool is_pgn_move_ambiguous(Move m, const Board& pos, const std::vector<Move>& legalMoves, bool& isFileAmbiguous, bool& isRankAmbiguous);

/**
 * @brief pgnMove Converts a given move into PGN move notation.
 * Passing the board position is needed to check for capture and possible ambiguity.
 * The ambiguity check requires the need to call generate_legal_moves()
 * @param m Move
 * @param chess960 True if 960 mode
 * @param pos Board position
 * @param legalMoves List of legal moves in the position (avoid regneration in case it has already been done)
 * @param leadsToWin True if the given move leads to a lost terminal state for the opponent
 * @param bookMove Appends " {book}" in case the move was a book move
 * @return String representation of move in PGN format
 */
std::string pgn_move(Move m, bool chess960, const Board& pos, const std::vector<Move>& legalMoves, bool leadsToWin=false, bool bookMove=false);

/**
 * @brief leads_to_terminal Checks if the next states is a terminal state if you would apply the given move
 * @param pos Given board position
 * @param m Move which is assumed to be legal
 * @return True, if the next state would be a terminal, else false
 */
bool leads_to_terminal(const Board& pos, Move m, StateListPtr& states);

/**
 * @brief get_result Returns the current game result. In case a normal position is given NO_RESULT is returned.
 * @param pos Board position
 * @param inCheck Determines if a king in the current position is in check (needed to differ between checkmate and stalemate).
 * It can be computed by `gives_check(<last-move-before-current-position>)`.
 * @return value in [DRAWN, WHITE_WIN, BLACK_WIN, NO_RESULT]
 */
Result get_result(const Board& pos, bool inCheck);

/**
 * @brief is_win Return true if the given result is a win, else false
 * @param res Result
 * @return Bool
 */
bool is_win(Result res);

/**
 * @brief probe_wdl Wrapper for probe_wdl(Position& pos, Tablebases::ProbeState* result)
 * @param pos Board position
 * @param result If result == FAIL then probe was unsuccessful
 * @return Returns WDL-score (-2 : loss, -1 : loss, but draw under 50-move rule, 0 : draw, 1 : win, but draw under 50-move rule, 2 : win)
 */
Tablebases::WDLScore probe_wdl(Board& pos, Tablebases::ProbeState* result);

/**
 * @brief probe_dtz Wrapper for int probe_dtz(Position &pos, Tablebases::ProbeState *result)
 * @param pos Board position
 * @param result If result == FAIL then probe was unsuccessful
 * @return Number of plies for a WIN (0 < n < 100) or LOSS (-100 < n < 0)
 */
int probe_dtz(Board& pos, Tablebases::ProbeState* result);

#endif // BOARD_H
