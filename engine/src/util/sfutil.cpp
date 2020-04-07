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

#include "sfutil.h"
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include "types.h"
#include "uci.h"

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

void fill_en_passent_moves(vector<string> &enPassentMoves) {

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
}

Square get_origin_square(string& uciMove)
{
    File from_file = FILE_LOOKUP.at(uciMove[0]);
    Rank from_rank = RANK_LOOKUP.at(uciMove[1]);
    return make_square(from_file, from_rank);
}

Square get_destination_square(string& uciMove)
{
    File to_file = FILE_LOOKUP.at(uciMove[2]);
    Rank to_rank = RANK_LOOKUP.at(uciMove[3]);
    return make_square(to_file, to_rank);
}

bool is_drop_move(string& uciMove)
{
    return uciMove[1] == '@';
}

bool is_promotion_move(string &uciMove)
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

void fill_castle_moves(vector<string> &castleMoves, bool is_960) {
    castleMoves.push_back("e1g1");
    castleMoves.push_back("e1c1");
    castleMoves.push_back("e8g8");
    castleMoves.push_back("e8c8");
    if (is_960) {
        throw invalid_argument( "960 castling isn't supported yet" );
    }
}

Bitboard flip_vertical(Bitboard x)
{
    return  ( (x << 56)                           ) |
            ( (x << 40) & 0x00ff000000000000 ) |
            ( (x << 24) & 0x0000ff0000000000 ) |
            ( (x <<  8) & 0x000000ff00000000 ) |
            ( (x >>  8) & 0x00000000ff000000 ) |
            ( (x >> 24) & 0x0000000000ff0000 ) |
            ( (x >> 40) & 0x000000000000ff00 ) |
            ( (x >> 56) );
}

string mirror_move(string& uciMove) {

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

vector<Move> make_move(string& uciMove, vector<string>& castlingMoves) {

    vector<Move> sfMoves;
    Square from_sq = get_origin_square(uciMove);
    Square to_sq = get_destination_square(uciMove);

    if (is_drop_move(uciMove)) {
        // in sf the dropping moves have a different id for black and white
        for (int color : {WHITE, BLACK}) {
            char piece = uciMove[0];
            if (color == BLACK) {
                piece = char(tolower(piece));
            }
            Piece pt = Piece(PIECE_LOOKUP.at(piece));
            sfMoves.push_back(make_drop(to_sq, pt));
        }
    }
    else {
        // castling moves have a seperate flag in sf
        if (find(castlingMoves.begin(), castlingMoves.end(), uciMove) != castlingMoves.end()) {
            sfMoves.push_back(make<CASTLING>(from_sq, to_sq));
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
