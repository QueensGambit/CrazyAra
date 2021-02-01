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
 * @file: board.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#ifndef MODE_POMMERMAN
#include "board.h"
#include "stateobj.h"
#include "uci.h"
#include <iostream>
#include "movegen.h"

using namespace std;

Board::Board()
{
}

Board::Board(const Board &b)
{
    // TODO: Change to usage of swap
    operator=(b);
}

Board::~Board()
{
}

#ifdef CRAZYHOUSE
Bitboard Board::promoted_pieces() const
{
    return promotedPieces;
}

int Board::get_pocket_count(Color c, PieceType pt) const
{
    return pieceCountInHand[c][pt];
}
#endif

Key Board::hash_key() const
{
    return st->key;
}

void Board::set_state_info(StateInfo *st)
{
    this->st = st;
}

StateInfo *Board::get_state_info() const
{
    return st;
}

Board& Board::operator=(const Board &b)
{
    std::copy(b.board, b.board+SQUARE_NB, this->board );
    std::copy(b.byTypeBB, b.byTypeBB+PIECE_TYPE_NB, this->byTypeBB);
    std::copy(b.byColorBB, b.byColorBB+COLOR_NB, this->byColorBB);
    std::copy(b.pieceCount, b.pieceCount+PIECE_NB, this->pieceCount);
#ifdef HORDE
    std::copy(&b.pieceList[0][0], &b.pieceList[0][0]+PIECE_NB*SQUARE_NB, &this->pieceList[0][0]);
#else
    std::copy(&b.pieceList[0][0], &b.pieceList[0][0]+PIECE_NB*16, &this->pieceList[0][0]);
#endif
#ifdef CRAZYHOUSE
    std::copy(&b.pieceCountInHand[0][0], &b.pieceCountInHand[0][0]+COLOR_NB*PIECE_TYPE_NB, &this->pieceCountInHand[0][0]);
    promotedPieces = b.promotedPieces;
#endif
    std::copy(b.index, b.index+SQUARE_NB, this->index);
    std::copy(b.castlingRightsMask, b.castlingRightsMask+SQUARE_NB, this->castlingRightsMask);
#if defined(GIVEAWAY) || defined(EXTINCTION) || defined(TWOKINGS)
    std::copy(b.castlingKingSquare, b.castlingKingSquare+COLOR_NB, this->castlingKingSquare);
#endif
    std::copy(b.castlingRookSquare, b.castlingRookSquare+CASTLING_RIGHT_NB, castlingRookSquare);
    std::copy(b.castlingPath, b.castlingPath+CASTLING_RIGHT_NB, this->castlingPath);
    this->gamePly = b.gamePly;
    sideToMove = b.sideToMove;
    psq = b.psq;
    thisThread = b.thisThread;
    st = b.st;
    chess960 = b.chess960;
    var = b.var;
    subvar = b.subvar;
#if defined(MODE_CHESS) || defined(MODE_LICHESS)
    lastMoves = b.lastMoves;  // vectors and deques are deeply copied by default
#endif
    return *this;
}

int Board::plies_from_null() const
{
    return st->pliesFromNull;
}

size_t Board::total_move_cout() const
{
    return size_t(gamePly / 2);
}

size_t Board::number_repetitions() const
{
    if (st->repetition == 0) {
        return 0;
    }
    else if (st->repetition) {
        return 1;
    }
    return 2;
}

bool Board::can_claim_3fold_repetition() const
{
    // The repetition info stores the ply distance to the next previous
    // occurrence of the same position.
    // It is negative in the 3-fold case, or zero if the position was not repeated.
    return st->repetition < 0;
}

bool Board::is_50_move_rule_draw() const
{
#ifdef CRAZYHOUSE
    if (is_house()) {} else
#endif
        if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size())) {
            return true;
        }
    return false;
}

bool Board::is_terminal() const
{
    // 3-fold-repetition and 50 move rul draw is handled outside move generation
    if (can_claim_3fold_repetition() || is_50_move_rule_draw() || draw_by_insufficient_material()) {
        return true;
    }

    for (const ExtMove move : MoveList<LEGAL>(*this)) {
        return false;
    }
    return true;
}

bool Board::draw_by_insufficient_material() const
{
#ifdef CRAZYHOUSE
    // fast return options (insufficient material can never by reached in crayhouse)
    if (is_house() || this->count<ALL_PIECES>() > 4) {
        return false;
    }
#endif

    return (this->count<ALL_PIECES>() == 2) ||                                      // 1) KK
           (this->count<ALL_PIECES>() == 3 && this->count<BISHOP>() == 1) ||        // 2) KB vs K
           (this->count<ALL_PIECES>() == 3 && this->count<KNIGHT>() == 1) ||        // 3) KN vs K
           (this->count<ALL_PIECES>() == 4 &&
            (this->count<KNIGHT>(WHITE) == 2 || this->count<KNIGHT>(BLACK) == 2));  // 4) KNN vs K

    return false;
}

#if defined(MODE_CHESS) || defined(MODE_LICHESS)
void Board::add_move_to_list(Move m)
{
    lastMoves.push_front(m);
    if (lastMoves.size() > StateConstants::NB_LAST_MOVES()) {
        lastMoves.pop_back();
    }
}

void Board::do_move(Move m, StateInfo &newSt)
{
    add_move_to_list(m);
    Position::do_move(m, newSt);
}

void Board::do_move(Move m, StateInfo &newSt, bool givesCheck)
{
    add_move_to_list(m);
    Position::do_move(m, newSt, givesCheck);
}

void Board::undo_move(Move m)
{
    lastMoves.pop_front();
    Position::undo_move(m);
}

deque<Move> Board::get_last_moves() const
{
    return lastMoves;
}

void Board::set(const string &fenStr, bool isChess960, Variant v, StateInfo *si, Thread *th)
{
    lastMoves.clear();
    Position::set(fenStr, isChess960, v, si, th);
}

void Board::set(const string &code, Color c, Variant v, StateInfo *si)
{
    lastMoves.clear();
    Position::set(code, c, v, si);
}
#endif

std::string pgn_move(Move m, bool chess960, const Board& pos, const std::vector<Action>& legalMoves, bool leadsToWin, bool bookMove)
{
    std::string move;

    const Square from = from_sq(m);
    const Square to = to_sq(m);

    if (m == MOVE_NONE) {
        return "(none)";
    }

    if (m == MOVE_NULL) {
        return "0000";
    }

    bool rank_ambiguous;
    bool file_ambiguous;

    string ambiguous = "";
    if (is_pgn_move_ambiguous(m, pos, legalMoves, file_ambiguous, rank_ambiguous)) {
        if (file_ambiguous && rank_ambiguous) {
            ambiguous = UCI::square(from);
        }
        else if (file_ambiguous) {
            ambiguous = char('1' + rank_of(from));
        }
        else {
            ambiguous = char('a' + file_of(from));
        }
    }
    else {
        ambiguous = "";
    }

    if (type_of(m) == CASTLING && !chess960) {
        if (file_of(to) == FILE_G || file_of(to) == FILE_H) {
            move = "O-O";
        }
        else {
            move = "O-O-O";
        }
    }
    else if (pos.capture(m)) {
        if (pos.piece_on(from) == W_PAWN || pos.piece_on(from) == B_PAWN) {
            move = std::string{"abcdefgh "[file_of(from)]} + "x";
        }
        else {
            move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + ambiguous + "x";
        }
        move += UCI::square(to);
    }

#ifdef CRAZYHOUSE
    else if (type_of(m) == DROP) {
        move = std::string{" PNBRQK  PNBRQK "[dropped_piece(m)], '@'} + UCI::square(to);
    }
#endif
    else {
        if (pos.piece_on(from) == W_PAWN || pos.piece_on(from) == B_PAWN) {
            move = UCI::square(to);
        }
        else {
            move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + ambiguous + UCI::square(to);
        }
    }

    if (type_of(m) == PROMOTION) {
        move += " PNBRQK"[promotion_type(m)];
    }

    if (pos.gives_check(m)) {
        if (leadsToWin) {
            move += "#";
        }
        else {
            move += "+";
        }
    }

    if (bookMove) {
        move += " {book}";
    }
    return move;
}


bool is_pgn_move_ambiguous(Move m, const Board& pos, const std::vector<Action> &legalMoves, bool &isFileAmbiguous, bool &isRankAmbiguous)
{
    bool ambiguous = false;
    isFileAmbiguous = false;
    isRankAmbiguous = false;
    const Square from = from_sq(m);
    const Square to = to_sq(m);

    for (Action move: legalMoves) {
        const Square cur_from = from_sq(Move(move));
        const Square cur_to = to_sq(Move(move));
        if (to == cur_to && from != cur_from && pos.piece_on(from) == pos.piece_on(cur_from)) {
            ambiguous = true;
            if (file_of(from) == file_of(cur_from)) {
                isFileAmbiguous = true;
            }
            if (rank_of(from) == rank_of(cur_from)) {
                isRankAmbiguous = true;
            }
        }
    }
    return ambiguous;
}

bool leads_to_terminal(const Board &pos, Move m, StateListPtr& states)
{
    Board posCheckTerminal = Board(pos);
    states->emplace_back();
    posCheckTerminal.do_move(m, states->back());
    return posCheckTerminal.is_terminal();
}

Result get_result(const Board& pos, bool inCheck)
{
    if (pos.is_terminal()) {
        if (!inCheck || pos.is_50_move_rule_draw() || pos.can_claim_3fold_repetition() || pos.draw_by_insufficient_material()) {
            return DRAWN;
        }
        if (pos.side_to_move() == BLACK) {
            return WHITE_WIN;
        }
        else {
            return BLACK_WIN;
        }
    }
    return NO_RESULT;
}

Tablebases::WDLScore probe_wdl(Board& pos, Tablebases::ProbeState* result)
{
    return Tablebases::probe_wdl(pos, result);
}

int probe_dtz(Board &pos, Tablebases::ProbeState *result)
{
    return Tablebases::probe_dtz(pos, result);
}

void generate_dtz_values(const vector<Move>& legalMoves, Board& pos, DynamicVector<int>& dtzValues) {
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(0));
    // fill dtz value vector
    for (size_t idx = 0; idx < legalMoves.size(); ++idx) {
        states->emplace_back();
        pos.do_move(legalMoves[idx], states->back());
        Tablebases::ProbeState result;
        int dtzValue = -probe_dtz(pos, &result);
        if (result != Tablebases::FAIL) {
            dtzValues[idx] = dtzValue;
        }
        else {
            cerr << "DTZ tablebase look-up failed!";
        }
        pos.undo_move(legalMoves[idx]);
    }
}

bool enhance_move_type(float increment, float thresh, const vector<Move>& legalMoves, const DynamicVector<bool>& moveType, DynamicVector<float>& policyProbSmall)
{
    bool update = false;
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        if (moveType[i] && policyProbSmall[i] < thresh) {
            policyProbSmall[i] += increment;
            update = true;
        }
    }
    return update;
}

bool is_check(const Board* pos, Move move)
{
    return pos->gives_check(move);
}

bool is_capture(const Board* pos, Move move)
{
    return pos->capture(move);
}

#endif
