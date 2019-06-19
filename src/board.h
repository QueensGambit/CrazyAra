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
 * @file: board.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef BOARD_H
#define BOARD_H

#include <position.h>



class Board : public Position
{
public:
    Board();
    Board(const Board& b);

    Bitboard promoted_pieces() const;
    int get_pocket_count(Color c, PieceType pt) const;


};

#endif // BOARD_H
