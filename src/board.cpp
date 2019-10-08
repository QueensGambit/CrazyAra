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
 * @file: board.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "board.h"
#include "constants.h"
#include "uci.h"
#include <iostream>

using namespace std;

Board::Board()
{
    board;
}

Board::Board(const Board &b)
{
    // TODO: Change to usage of swap
    operator=(b);
}

Board::~Board()
{
    delete st;
}

Bitboard Board::promoted_pieces() const
{
    return promotedPieces;
}

int Board::get_pocket_count(Color c, PieceType pt) const
{
    return pieceCountInHand[c][pt];
}

Key Board::hash_key() const
{
    return st->key; // + size_t(st->pliesFromNull);
}

void Board::setStateInfo(StateInfo *st)
{
    this->st = st;
}

StateInfo *Board::getStateInfo() const
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
#if defined(ANTI) || defined(EXTINCTION) || defined(TWOKINGS)
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

    return *this;
}

int Board::plies_from_null()
{
    return st->pliesFromNull;
}

size_t Board::total_move_cout()
{
    return st->pliesFromNull / 2 + 1;
}

std::string pgnMove(Move m, bool chess960, const Board& pos, bool leadsToTerminal)
{
    std::string move;

    Square from = from_sq(m);
    Square to = to_sq(m);

    if (m == MOVE_NONE)
        return "(none)";

    if (m == MOVE_NULL)
        return "0000";

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
        move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + "x";
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
move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + UCI::square(to);
}
}

if (type_of(m) == PROMOTION) {
    move += " PNBRQK"[promotion_type(m)];
}

if (pos.gives_check(m)) {
    if (leadsToTerminal) {
        move += "#";
    }
    else {
        move += "+";
    }
}
return move;
}

