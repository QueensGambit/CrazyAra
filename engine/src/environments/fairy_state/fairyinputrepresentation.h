#ifndef FAIRYINPUTREPRESENTATION_H
#define FAIRYINPUTREPRESENTATION_H

#include "fairyboard.h"

/**
 * @brief board_to_planes Converts the given board representation into the plane representation.
 * @param pos Board position
 * @param normalize Flag, telling if the representation should be rescaled into the [0,1] range
 * @param input_planes Output where the plane representation will be stored.
 */
void board_to_planes(const FairyBoard* pos, bool normalize, float *inputPlanes);

/**
 * @brief set_bits_from_bitmap Sets the individual bits from a given bitboard on the given channel for the inputPlanes
 * @param bitboard Bitboard of a single 8x8 plane
 * @param channel Channel index on where to set the bits
 * @param input_planes Input planes encoded as flat vector
 * @param color Color of the side to move
 */
inline void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *inputPlanes, Color color);

#endif // FAIRYINPUTREPRESENTATION_H
