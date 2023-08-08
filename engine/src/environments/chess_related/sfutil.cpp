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
 * @file: sfutil.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#ifndef MODE_POMMERMAN
#include "sfutil.h"
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include "types.h"
#include "uci.h"

// https://stackoverflow.com/questions/41770887/cross-platform-definition-of-byteswap-uint64-and-byteswap-ulong
#ifdef _MSC_VER
#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)
#elif defined(__APPLE__)
// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)
#elif defined(__sun) || defined(sun)
#include <sys/byteorder.h>
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)
#elif defined(__FreeBSD__)
#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#elif defined(__OpenBSD__)
#include <sys/types.h>
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)
#elif defined(__NetBSD__)
#include <sys/types.h>
#include <machine/bswap.h>
#if defined(__BSWAP_RENAME) && !defined(__bswap_32)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#endif
#else
#include <byteswap.h>
#endif

const unordered_map<char, File> FILE_LOOKUP = {
    {'a', FILE_A},
    {'b', FILE_B},
    {'c', FILE_C},
    {'d', FILE_D},
    {'e', FILE_E},
    {'f', FILE_F},
    {'g', FILE_G},
    {'h', FILE_H}};

const unordered_map<char, Rank> RANK_LOOKUP = {
    {'1', RANK_1},
    {'2', RANK_2},
    {'3', RANK_3},
    {'4', RANK_4},
    {'5', RANK_5},
    {'6', RANK_6},
    {'7', RANK_7},
    {'8', RANK_8}};

const unordered_map<char, PieceType> PIECE_TYPE_LOOKUP = {
    {'p', PAWN},
    {'n', KNIGHT},
    {'b', BISHOP},
    {'r', ROOK},
    {'q', QUEEN},
    {'k', KING}};

const unordered_map<char, Piece> PIECE_LOOKUP = {
    {'P', W_PAWN},
    {'N', W_KNIGHT},
    {'B', W_BISHOP},
    {'R', W_ROOK},
    {'Q', W_QUEEN},
    {'K', W_KING},
    {'p', B_PAWN},
    {'n', B_KNIGHT},
    {'b', B_BISHOP},
    {'r', B_ROOK},
    {'q', B_QUEEN},
    {'k', B_KING}};

vector<string> create_en_passent_moves()
{
    vector<string> enPassentMoves;
    for (int color : {WHITE, BLACK}) {
        // white en-passent moves
        int from_rank = 5;
        int to_rank = 6;

        if (color == BLACK) {
            // black en-passent moves
            from_rank = 4;
            to_rank = 3;
        }

        for (char from_file = 'a'; from_file <= 'h'; ++from_file) {
            for (char to_file = char(max(int(from_file-1), int('a'))); to_file <= char(min(int(from_file+1), int('g'))); to_file+=2) {
                // correct the side-lines
                if (from_file == to_file) {
                    if (to_file == 'a') {
                        to_file += 1;
                    }
                    if (to_file == 'h') {
                        to_file -= 1;
                    }
                }
                string mv = from_file + to_string(from_rank) + to_file + to_string(to_rank);
                enPassentMoves.push_back(mv);
            }
        }
    }
    return enPassentMoves;
}

Square get_origin_square(const string& uciMove)
{
    File from_file = FILE_LOOKUP.at(uciMove[0]);
    Rank from_rank = RANK_LOOKUP.at(uciMove[1]);
    return make_square(from_file, from_rank);
}

Square get_destination_square(const string& uciMove)
{
    File to_file = FILE_LOOKUP.at(uciMove[2]);
    Rank to_rank = RANK_LOOKUP.at(uciMove[3]);
    return make_square(to_file, to_rank);
}

bool is_drop_move(const string& uciMove)
{
    return uciMove[1] == '@';
}

bool is_promotion_move(const string& uciMove)
{
    return uciMove.length() == 5;
}

bool is_en_passent_candidate(Square origin, Square destination)
{
    // en-passent move candidates for white & black
    if ((rank_of(origin) == RANK_5 && rank_of(destination) == RANK_6) || (rank_of(origin) == RANK_4 && rank_of(destination) == RANK_3)) {
        // diagonal pawn-captures
        if ((file_of(destination) == file_of(origin) - 1) || (file_of(destination) == file_of(origin) + 1)) {
            return true;
        }
    }
    return false;
}

Bitboard flip_vertical(Bitboard x)
{
    return bswap_64(x);
}

string mirror_move(const string& uciMove)
{
    // first copy the original move
    string moveMirrored = string(uciMove);

    // replace the rank with the mirrored rank
    for (unsigned int idx = 0; idx < uciMove.length(); ++idx) {
        if (isdigit(uciMove[idx])) {
            int rank = uciMove[idx] - '0';
            int rank_mirrored = 8 - rank + 1;
            moveMirrored[idx] = char(rank_mirrored + '0');
        }
    }
    return moveMirrored;
}

vector<Move> make_move(const string& uciMove, bool is960)
{
    vector<Move> sfMoves;
    Square to_sq = get_destination_square(uciMove);

    if (is_drop_move(uciMove)) {
#ifdef CRAZYHOUSE
        // dropping moves have a different id for black and white in Stockfish's move representation
        for (int color : {WHITE, BLACK}) {
            char piece = uciMove[0];
            if (color == BLACK) {
                piece = char(tolower(piece));
            }
            Piece pt = Piece(PIECE_LOOKUP.at(piece));
            sfMoves.push_back(make_drop(to_sq, pt));
        }
#endif
    }
    else {
        Square from_sq = get_origin_square(uciMove);
        // castling moves have a seperate flag in Stockfish's move representation
        if (is960) {
            if (is_960_castling_candidate_move(from_sq, to_sq)) {
                sfMoves.push_back(make<CASTLING>(from_sq, to_sq));
            }
        }
        else {
            handle_classical_castling_moves(uciMove, sfMoves);
        }

        if (is_en_passent_candidate(from_sq, to_sq)) {
            sfMoves.push_back(make<ENPASSANT>(from_sq, to_sq));
        }
        if (is_promotion_move(uciMove)) {
            PieceType pt = PIECE_TYPE_LOOKUP.at(uciMove[4]);
            sfMoves.push_back(make<PROMOTION>(from_sq, to_sq, pt));
        }
        else {
            sfMoves.push_back(make_move(from_sq, to_sq));
        }
    }
    return sfMoves;
}

void handle_classical_castling_moves(const string& uciMove, vector<Move>& moveVector)
{
    Square w_ksq = make_square(FILE_E, RANK_1);
    Square b_ksq = make_square(FILE_E, RANK_8);

    if (uciMove == "e1g1") {
        moveVector.push_back(make<CASTLING>(w_ksq, make_square(FILE_H, RANK_1)));
    }
    else if (uciMove == "e1c1") {
        moveVector.push_back(make<CASTLING>(w_ksq, make_square(FILE_A, RANK_1)));
    }
    else if (uciMove == "e8g8") {
        moveVector.push_back(make<CASTLING>(b_ksq, make_square(FILE_H, RANK_8)));
    }
    else if (uciMove == "e8c8") {
        moveVector.push_back(make<CASTLING>(b_ksq, make_square(FILE_A, RANK_8)));
    }
}

vector<string> create_castling_moves(bool is960)
{
    if (!is960) {
        return vector<string>{"e1g1", "e1c1", "e8g8" "e8c8"};
    }
    vector<string> castlingMoves;
    for (char rank = '1'; rank <= '8'; rank+=7) {
        for (char fileKing = 'b'; fileKing <= 'g'; ++fileKing) {
            for (char fileRook = 'a'; fileRook <= 'h'; ++fileRook) {
                if (fileKing != fileRook) {
                    castlingMoves.push_back(string{fileKing, rank, fileRook, rank});
                }
            }
        }
    }
    return castlingMoves;
}

bool is_960_castling_candidate_move(Square origin, Square destination)
{
    return rank_of(origin) == rank_of(destination) &&
           (rank_of(origin) == RANK_1 || rank_of(origin) == RANK_8) &&
           file_of(origin) != FILE_A && file_of(origin) != FILE_H;
}
#endif
