/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: inputrepresentation.cpp
 * Created on 27.05.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "inputrepresentation.h"
//#include "board.h"
#include "constants.h"
#include <iostream>
using namespace std;

void fill_value(int pocket_cnt, int current_channel, float *input_planes) {
    if (pocket_cnt > 0) {
        std::fill(input_planes + (current_channel+1) * NB_SQUARES, input_planes + (current_channel+2) * NB_SQUARES, 1.0f);
    }
}

void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *input_planes, Color color) {
    size_t p;
    true ? p = 0 : p = 63;
    // set the individual bits for the pieces
    // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
    while (bitboard != 0) {
      if (bitboard & 0x1) {
          if (color == WHITE) {
            input_planes[channel * NB_SQUARES + p] = 1;
          }
          else {
              size_t row = p / 8;
              size_t col = p % 8;
              input_planes[channel * NB_SQUARES + (7 - row) * 8 + col] = 1;
          }
      }
      bitboard >>= 1;
      true ? p++ : p--;
    }
}


void board_to_planes(Board pos, int board_occ, bool normalize, float *input_planes) {

    // intialize the input_planes with 0
    std::fill(input_planes, input_planes+NB_VALUES_TOTAL, 0.0f);

    // Fill in the piece positions

    // Iterate over both color starting with WHITE
    size_t current_channel = 0;
    Color us = pos.side_to_move();
    Color them = ~us;

    // (I) Set the pieces for both players
    for (Color color : {us, them}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            Bitboard pieces = pos.pieces(color, piece);
//            size_t p = 0;
            // set the individual bits for the pieces
            // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
            set_bits_from_bitmap(pieces, current_channel, input_planes, us);
            current_channel += 1;
        }
    }

    // (II) Fill in the Repetition Data
    // set how often the position has already occurred in the game (default 0 times)
    // this is used to check for claiming the 3 fold repetition rule
    // A game to test out if everything is working correctly is: https://lichess.org/jkItXBWy#73
    if (board_occ >= 1) {
        std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        if (board_occ >= 2) {
            std::fill(input_planes + (current_channel+1) * NB_SQUARES, input_planes + (current_channel+2) * NB_SQUARES, 1.0f);
        }
    }
    current_channel+= 2;

    // (III) Fill in the Prisoners / Pocket Pieces
    // iterate over all pieces except the king
    for (Color color : {us, them}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
            // unfortunately you can't use a loop over count_in_hand() PieceType because of template arguments
            int pocket_cnt = pos.get_pocket_count(color, piece);
            if (pocket_cnt > 0) {
                std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES,
                          normalize ? pocket_cnt / MAX_NB_PRISONERS : pocket_cnt);
            }
            current_channel++;
        }
    }

    // (IV) Fill in the promoted pieces
    // iterate over all promoted pieces according to the mask and set the according bit
    set_bits_from_bitmap(pos.promoted_pieces() & pos.pieces(us), current_channel, input_planes, us);
    current_channel++;
    set_bits_from_bitmap(pos.promoted_pieces() & pos.pieces(them), current_channel, input_planes, us);
    current_channel++;

    // (V) En Passant Square
    // mark the square where an en-passant capture is possible
    if (pos.ep_square() != SQ_NONE) {
        unsigned int ep_square = us == WHITE ? int(pos.ep_square()) : 64-int(pos.ep_square());
        input_planes[current_channel * NB_SQUARES + ep_square] = 1.0f;
    }
    current_channel++;

    // (VI) Constant Value Inputs
    // (VI.1) Color
    if (us == WHITE) {
        std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
    }
    current_channel++;

//    std::cout << "pos.game_ply()" << pos.game_ply() << std::endl;
    // (VI.2) Total Move Count
    std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES,
              // stockfish starts counting from 0, the full move counter starts at 1 in FEN
              normalize ? ((pos.game_ply()/2)+1) / MAX_FULL_MOVE_COUNTER : ((pos.game_ply()/2)+1));
    current_channel++;

    // (IV.3) Castling Rights
    // check for King Side Castling
    if (us == WHITE) {
        if (pos.can_castle(WHITE_OO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(WHITE_OOO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(BLACK_OO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(BLACK_OOO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
    }   else {
        if (pos.can_castle(BLACK_OO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(BLACK_OOO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(WHITE_OO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;
        if (pos.can_castle(WHITE_OOO)) {
            std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES, 1.0f);
        }
        current_channel++;

    }
//    std::cout << "last channel" << current_channel << std::endl;
    // (VI.4) No Progress Count
    // define a no 'progress' counter
    // it gets incremented by 1 each move
    // however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    // halfmove_clock is an official metric in fen notation
    //  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    std::fill(input_planes + current_channel * NB_SQUARES, input_planes + (current_channel+1) * NB_SQUARES,
              normalize ? pos.rule50_count() / MAX_NB_NO_PROGRESS: pos.rule50_count());
    current_channel++;
}

