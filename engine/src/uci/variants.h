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
 * @file: variants.h
 * Created on 12.07.2019
 * @author: queensgambit
 *
 * Constant definitions for available chess variants
 */

#ifndef VARIANTS_H
#define VARIANTS_H

#include "types.h"
using namespace std;

// list of all current available variants for CrazyAra
static vector<string> availableVariants = {
    "crazyhouse",
    "chess",
    "atomic",
    "giveaway",  // antichess
};

// FEN strings of the initial positions
const static string StartFENs[SUBVARIANT_NB] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #ifdef ANTI
    // "The king has no royal power and accordingly:
    // it may be captured like any other piece
    // there is no check or checkmate there is no castling"
    // -- https://en.wikipedia.org/wiki/Losing_chess
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    #endif
    #ifdef ATOMIC
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef CRAZYHOUSE
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef EXTINCTION
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef GRID
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef HORDE
    "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
    #endif
    #ifdef KOTH
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef LOSERS
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef RACE
    "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1",
    #endif
    #ifdef THREECHECK
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
    #endif
    #ifdef TWOKINGS
    "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
    #endif
    #ifdef SUICIDE
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    #endif
    #ifdef BUGHOUSE
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef DISPLACEDGRID
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef LOOP
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef PLACEMENT
    "8/pppppppp/8/8/8/8/PPPPPPPP/8[KQRRBBNNkqrrbbnn] w - -",
    #endif
    #ifdef SLIPPEDGRID
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef TWOKINGSSYMMETRIC
    "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
    #endif
};

#endif // VARIANTS_H
