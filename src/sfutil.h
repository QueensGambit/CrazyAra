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
 * @file: sfutil.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Additional utility functions for the Stockfish library.
 */

#ifndef SFUTIL_H
#define SFUTIL_H

#include <string>
#include "types.h"

/**
 * @brief make_move Creates a move in coordinate representation given an uci string.
 *                  Multiple sf moves are returned in case it's ambigious such as castling or en-passent moves.
 * @param uciMove Valid uci string including crazyhouse dropping moves
 * @return Move in coordinate representation
 */
std::vector<Move> make_move(std::string uciMove);


/**
 * @brief fill_en_passent_moves
 */// en-passent capture can occur from the 5th rank in the view of white and from the 4th rank for black
// each square has two ways to capture except the border square with only one capture
void fill_en_passent_moves(std::vector<std::string> &enPassentMoves);

#endif // SFUTIL_H
