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
 * @file: traindata.h
 * Created on 11.09.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef TRAINDATA_H
#define TRAINDATA_H

#include <inttypes.h>
#include <zlib.h>

#include "../domain/crazyhouse/constants.h"

using namespace std;

struct TrainDataExport {
    // Training sample information / inputs / x
    // -----------------------------------------
    // P1 defines the current player to move, (us/me)
    // P2 defines the opposing player (them/you)
    // Note: bool is exported as one byte and not one bit

    // Bitboard information
    // P1 {PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING}
    // P2 {PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING}
    // P1 Promoted Pawns
    // P2 Promoted Pawns
    // En-passant square
    Bitboard pieces[NB_PIECE_TYPES * NB_PLAYERS];

    // number of how often this position already occured
    uint8_t repetitions;

#ifdef CRAZYHOUSE
    // pocket pieces for player P1 followed by P2
    // the king is excluded
    uint8_t pocketCount[(NB_PIECE_TYPES-1) * NB_PLAYERS];
    // promoted pawns (is flipped for P=BLACK)
    Bitboard promotedPawns[NB_PLAYERS];
#endif
    // en-passent square (is flipped for sideToMove=BLACK)
    uint64_t enPassentSquare;

    // color/sideToMove: false for black and true for white
    bool color;
    // sets the full move count (FEN notation)
    uint8_t totalMoveCount;

    // castling rights
    // {P1_KING_SIDE, P1_QUEEN_SIDE, P2_KING_SIDE, P2_QUEEN_SIDE}
    bool castlingRights[NB_PLAYERS * 2];

    // No-progress count / FEN halfmove clock / rule50 count
    uint8_t noProgressCount;

    // Learning targets / output / y
    // -----------------------------
    float probabilities[NB_VALUES_TOTAL];
    int8_t gameResult;

    // misc
    uint32_t version;
    float moveQvalue;
    float highestQvalue;
};

class Exporter {
private:
    string filename = "/home/queensgambit/Desktop/Programming/C++/build-HelloQt-Desktop-Release/data.gz";
    gzFile fout;
public:
    Exporter();
    int export_training_sample(const TrainDataExport& trainData);
    void close_gz();
};

#endif // TRAINDATA_H
