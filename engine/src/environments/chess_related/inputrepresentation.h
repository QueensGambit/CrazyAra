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
 * @file: inputrepresentation.h
 * Created on 27.05.2019
 * @author: queensgambit
 *
 * Utility methods for defining the inputs of the neural network
 */

#ifndef INPUTREPRESENTATION_H
#define INPUTREPRESENTATION_H

#include "board.h"

/**
 * @brief board_to_planes Converts the given board representation into the plane representation.
 *                        The plane representation will be encoded on a flat float array.
 *                        No history and only the single current board position will be encoded.
 * @param pos Board position
 * @param boardRepetition Defines how often the board has already been repeated so far
 * @param normalize Flag, telling if the representation should be rescaled into the [0,1] range using the scaling constants from "constants.h"
 * @param input_planes Output where the plane representation will be stored.
 */
void board_to_planes(const Board *pos, size_t boardRepetition, bool normalize, float *inputPlanes);

/**
 * @brief set_bits_from_bitmap Sets the individual bits from a given bitboard on the given channel for the inputPlanes
 * @param bitboard Bitboard of a single 8x8 plane
 * @param channel Channel index on where to set the bits
 * @param input_planes Input planes encoded as flat vector
 * @param color Color of the side to move
 */
inline void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *inputPlanes, Color color);

#endif // INPUTREPRESENTATION_H
