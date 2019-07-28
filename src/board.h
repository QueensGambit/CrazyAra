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
 * @file: board.h
 * Created on 23.05.2019
 * @author: queensgambit
 *
 * Extension of Stockfish's board presentation by introducing new functionality.
 */

#ifndef BOARD_H
#define BOARD_H

#include <position.h>

class Board : public Position
{
public:
    Board();
    Board(const Board& b);
    ~Board();

    Bitboard promoted_pieces() const;
    int get_pocket_count(Color c, PieceType pt) const;
    Key hash_key() const;
    void setStateInfo(StateInfo* st);
    StateInfo* getStateInfo() const;

    Board& operator=(const Board &b);
    int plies_from_null();

    /**
     * @brief total_move_cout Returns the current full move counter
     * @return Total move number
     */
    size_t total_move_cout();
};

#endif // BOARD_H
