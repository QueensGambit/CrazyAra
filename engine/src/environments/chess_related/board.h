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

#ifndef MODE_POMMERMAN
#include <position.h>
#include <deque>
#include "syzygy/tbprobe.h"
#include "../constants.h"
#include <blaze/Math.h>
using blaze::StaticVector;
using blaze::DynamicVector;

class Board : public Position
{
private:
    // up to NB_LAST_MOVES are stored in a list, most recent moves first
    deque<Move> lastMoves;
    /**
     * @brief add_move_to_list Adds a given move to the move list and removes the
     * last element if the list exceeds NB_LAST_MOVES items
     * @param m Given Move
     */
    inline void add_move_to_list(Move m);

public:
    Board();
    Board(const Board& b);
    ~Board();

#ifdef CRAZYHOUSE
    Bitboard promoted_pieces() const;
    int get_pocket_count(Color c, PieceType pt) const;
#endif
    Key hash_key() const;
    void set_state_info(StateInfo* st);
    StateInfo* get_state_info() const;

    Board& operator=(const Board &b);
    int plies_from_null() const;

    /**
     * @brief legal_moves Return all legal moves
     * @return  Legal moves
     */
    vector<Action> legal_actions() const;

    /**
     * @brief total_move_cout Returns the current full move counter.
     * In the initial starting position and after first half move of whites, it returns 0
     * @return Total move number
     */
    size_t total_move_cout() const;

    /**
     * @brief number_repetitions Returns how often the position has already occurred.
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
     * @return True, if draws by insufficient material occurred
     */
    bool draw_by_insufficient_material() const;

    /**
     * @brief get_phase Returns the game phase of the current board state based on the total amount of phases and the chosen GamePhaseDefinition
     * Possible returned values are all integers from 0 to numPhases - 1
     * @param unsigned int numPhases
     * @param GamePhaseDefinition gamePhaseDefinition
     * @return Game phase as unsigned int
     */
    GamePhase get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const;

    // overloaded function which include a last move list update
    void do_move(Move m, StateInfo& newSt);
    void do_move(Move m, StateInfo& newSt, bool givesCheck);
    void undo_move(Move m);
    void set(const std::string& fenStr, bool isChess960, Variant v, StateInfo* si, Thread* th);
    void set(const std::string& code, Color c, Variant v, StateInfo* si);
    deque<Move> get_last_moves() const;

    /**
     * @brief count_board_piece Counts the number of board pieces of a particular piece of a certain color.
     * @param color Color
     * @param pieceType piece type
     * @return Number of pieces on the board of a given piece type and color
     */
    int get_board_piece_count(Color color, PieceType pieceType) const;
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
bool is_pgn_move_ambiguous(Move m, const Board& pos, const std::vector<Action>& legalMoves, bool& isFileAmbiguous, bool& isRankAmbiguous);

/**
 * @brief pgnMove Converts a given move into PGN move notation.
 * Passing the board position is needed to check for capture and possible ambiguity.
 * The ambiguity check requires the need to call generate_legal_moves()
 * @param m Move
 * @param chess960 True if 960 mode
 * @param pos Board position
 * @param legalMoves List of legal moves in the position (avoid regeneration in case it has already been done)
 * @param leadsToWin True if the given move leads to a lost terminal state for the opponent
 * @param bookMove Appends " {book}" in case the move was a book move
 * @return String representation of move in PGN format
 */
std::string pgn_move(Move m, bool chess960, const Board& pos, const std::vector<Action>& legalMoves, bool leadsToWin=false, bool bookMove=false);

/**
 * @brief leads_to_terminal Checks if the next states is a terminal state if you would apply the given move
 * @param pos Given board position
 * @param m Move which is assumed to be legal
 * @return True, if the next state would be a terminal, else false
 */
bool leads_to_terminal(const Board& pos, Move m, StateListPtr& states);

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

/**
 * @brief generate_dtz_values Generates the DTZ values for a given position and all legal moves.
 * This function assumes that the given position is a TB entry.
 * Warning: The DTZ values do not return the fastest way to win but the distance to zeroing (50 move rule counter reset)
 * @param legalMoves Legal moves
 * @param pos Current position
 * @param dtzValues Returned dtz-Values in the view of the current player to use
 */
void generate_dtz_values(const vector<Move>& legalMoves, Board& pos, DynamicVector<int>& dtzValues);

// https://stackoverflow.com/questions/6339970/c-using-function-as-parameter
typedef bool (* vFunctionMoveType)(const Board* pos, Move move);
inline bool is_check(const Board* pos, Move move);
inline bool is_capture(const Board* pos, Move move);

/**
 * @brief enhance_checks Enhances all possible checking moves below threshCheck by incrementCheck and returns true if a modification
 * was applied. This signals that a renormalization should be applied afterwards.
 * @param increment_check Constant factor which is added to the checks below threshCheck
 * @param threshCheck Probability threshold for checking moves
 * @param gcThread Reference to the garbage collector object
 * @return bool
*/
inline bool enhance_move_type(float increment, float thresh, const vector<Move>& legalMoves,
                              const DynamicVector<bool>& moveType, DynamicVector<float>& policyProbSmall);
/**
 * @brief get_majors_and_minors_count Returns the amount of majors and minors currently still on the board (both sides)
 * @param pos Given board position
 * @return Unsigned integer representing the amount of majors and minors left
 */
unsigned int get_majors_and_minors_count(const Board& pos);

/**
 * @brief is_backrank_sparse Checks whether the backrank of either side is sparse (three or less pieces)
 * @param pos Given board position
 * @return True if either the white or the black backrank is sparse
 */
bool is_backrank_sparse(const Board& pos);

/**
 * @brief score_region Calculates a mixedness score for a 2x2 subregion of the board
 * @param numWhitePiecesInRegion Amount of white pieces in the current region
 * @param numBlackPiecesInRegion Amount of black pieces in the current region
 * @param rank Rank of the current region
 * @return Integer representing the mixedness of the region
 */
int score_region(int numWhitePiecesInRegion, int numBlackPiecesInRegion, int rank);

/**
 * @brief get_mixedness Returns the mixedness of the given position as defined in https://github.com/lichess-org/scalachess/blob/master/src/main/scala/Divider.scala
 * @param pos Given board position
 * @return Integer representing the mixedness of the given position
 */
int get_mixedness(const Board& pos);

#endif // BOARD_H
#endif
