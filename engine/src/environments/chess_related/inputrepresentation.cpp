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
 * @file: inputrepresentation.cpp
 * Created on 27.05.2019
 * @author: queensgambit
 */

#include "inputrepresentation.h"
#include <iostream>
#include <deque>
#include "stateobj.h"
#include "sfutil.h"
using namespace std;

void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *inputPlanes, Color color) {
    size_t p = 0;
    // set the individual bits for the pieces
    // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
    while (bitboard != 0) {
        if (bitboard & 0x1) {
            if (color == WHITE) {
                inputPlanes[channel * StateConstants::NB_SQUARES() + p] = 1;
            }
            else {
                //                                         row            col
                inputPlanes[channel * StateConstants::NB_SQUARES() + (7 - (p / 8)) * 8 + (p % 8)] = 1;
            }
        }
        bitboard >>= 1;
        p++;
    }
}


void board_to_planes(const Board *pos, size_t boardRepetition, bool normalize, float *inputPlanes)
{

    // intialize the input_planes with 0
    std::fill(inputPlanes, inputPlanes+StateConstants::NB_VALUES_TOTAL(), 0.0f);

    // Fill in the piece positions
    // Iterate over both color starting with WHITE
    size_t current_channel = 0;
    Color me = pos->side_to_move();
    Color you = ~me;

    // (I) Set the pieces for both players
    for (Color color : {me, you}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            const Bitboard pieces = pos->pieces(color, piece);
            // set the individual bits for the pieces
            // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
            set_bits_from_bitmap(pieces, current_channel, inputPlanes, me);
            current_channel += 1;
        }
    }

    // (II) Fill in the Repetition Data
    // set how often the position has already occurred in the game (default 0 times)
    // this is used to check for claiming the 3 fold repetition rule
    // A game to test out if everything is working correctly is: https://lichess.org/jkItXBWy#73
    if (boardRepetition >= 1) {
        std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        if (boardRepetition >= 2) {
            std::fill(inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+2) * StateConstants::NB_SQUARES(), 1.0f);
        }
    }
    current_channel+= 2;

#ifndef MODE_CHESS
    // (III) Fill in the Prisoners / Pocket Pieces
    // iterate over all pieces except the king
    for (Color color : {me, you}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
            // unfortunately you can't use a loop over count_in_hand() PieceType because of template arguments
            int pocket_cnt = pos->get_pocket_count(color, piece);
            if (pocket_cnt > 0) {
                std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(),
                          normalize ? pocket_cnt / StateConstants::MAX_NB_PRISONERS() : pocket_cnt);
            }
            current_channel++;
        }
    }

    // (IV) Fill in the promoted pieces
    // iterate over all promoted pieces according to the mask and set the according bit
    set_bits_from_bitmap(pos->promoted_pieces() & pos->pieces(me), current_channel, inputPlanes, me);
    current_channel++;
    set_bits_from_bitmap(pos->promoted_pieces() & pos->pieces(you), current_channel, inputPlanes, me);
    current_channel++;
#endif

    // (V) En Passant Square
    // mark the square where an en-passant capture is possible
    if (pos->ep_square() != SQ_NONE) {
        unsigned int ep_square = me == WHITE ? int(pos->ep_square()) : int(vertical_flip(pos->ep_square()));
        inputPlanes[current_channel * StateConstants::NB_SQUARES() + ep_square] = 1.0f;
    }
    current_channel++;

    // (VI) Constant Value Inputs
    // (VI.1) Color
    if (me == WHITE) {
        std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
    }
    current_channel++;

    // (VI.2) Total Move Count
    std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(),
              // stockfish starts counting from 0, the full move counter starts at 1 in FEN
              normalize ? ((pos->game_ply()/2)+1) / StateConstants::MAX_FULL_MOVE_COUNTER() : ((pos->game_ply()/2)+1));
    current_channel++;

    // (IV.3) Castling Rights
    // check for King Side Castling
    if (me == WHITE) {
        if (pos->can_castle(WHITE_OO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(WHITE_OOO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(BLACK_OO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(BLACK_OOO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
    }   else {
        if (pos->can_castle(BLACK_OO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(BLACK_OOO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(WHITE_OO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;
        if (pos->can_castle(WHITE_OOO)) {
            std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
        }
        current_channel++;

    }

    // (VI.4) No Progress Count
    // define a no 'progress' counter
    // it gets incremented by 1 each move
    // however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    // halfmove_clock is an official metric in fen notation
    //  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(),
              normalize ? pos->rule50_count() / StateConstants::MAX_NB_NO_PROGRESS(): pos->rule50_count());
#ifndef MODE_CRAZYHOUSE
    current_channel++;
#endif

#ifdef MODE_LICHESS
    // set the remaining checks (only needed for "3check")
    if (pos->is_three_check()) {
        for (Color color : {me, you}) {
            if (pos->checks_given(color) != 0) {
                std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
                current_channel++;
                if (pos->checks_given(color) >= 2) {
                    std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
                }
                current_channel++;
            }
            else {
                current_channel += 2;
            }
        }
    }
    else {
        current_channel += 4;
    }

    // (V) Variants specification
    // set the is960 boolean flag when active
    if (pos->is_chess960()) {
        std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
    }

    // set the current active variant as a one-hot encoded entry
    current_channel += StateConstants::CHANNEL_MAPPING_VARIANTS().at(pos->variant());
    std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
#endif

#ifdef MODE_CHESS
    // (V) Variants specification
    // set the is960 boolean flag when active
    if (pos->is_chess960()) {
        std::fill(inputPlanes + current_channel * StateConstants::NB_SQUARES(), inputPlanes + (current_channel+1) * StateConstants::NB_SQUARES(), 1.0f);
    }
#endif

#if defined(MODE_CHESS) || defined(MODE_LICHESS)
    current_channel = StateConstants::NB_CHANNELS_TOTAL() - StateConstants::NB_CHANNELS_HISTORY();
    // (VI) Fill the bits of the last move planes
    for (const Move move : pos->get_last_moves()) {
        if (me == WHITE) {
            inputPlanes[current_channel++ * StateConstants::NB_SQUARES() + from_sq(move)] = 1.0f;
            inputPlanes[current_channel++ * StateConstants::NB_SQUARES() + to_sq(move)] = 1.0f;
        }
        else {
            inputPlanes[current_channel++ * StateConstants::NB_SQUARES() + int(vertical_flip(from_sq(move)))] = 1.0f;
            inputPlanes[current_channel++ * StateConstants::NB_SQUARES() + int(vertical_flip(to_sq(move)))] = 1.0f;
        }
    }
#endif
}

