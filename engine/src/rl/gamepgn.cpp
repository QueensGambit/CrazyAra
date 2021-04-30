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
 * @file: gamepgn.cpp
 * Created on 16.09.2019
 * @author: queensgambit
 */

#include "gamepgn.h"

ostream &operator<<(ostream& os, const GamePGN& gamePGN)
{
    const size_t plyCount = gamePGN.gameMoves.size();
    os << "[Variant \"" << gamePGN.variant << "\"]" << endl
       << "[Event \"" << gamePGN.event << "\"]" << endl
       << "[Date \"" << gamePGN.date << "\"]" << endl
       << "[Site \"" << gamePGN.site << "\"]" << endl
       << "[Round \"" << gamePGN.round << "\"]" << endl
       << "[FEN \"" << gamePGN.fen << "\"]" << endl
       << "[White \"" << gamePGN.white << "\"]" << endl
       << "[Black \"" << gamePGN.black << "\"]" << endl
       << "[Result \"" << gamePGN.result << "\"]" << endl
       << "[PlyCount \"" << plyCount << "\"]" << endl
       << "[TimeControl \"" << gamePGN.timeControl << "\"]" << endl << endl;

    for (size_t ply = 0; ply < plyCount; ++ply) {
        if (ply % 2 == 0) {
            os << ply/2+1  << ". ";
        }
        os << gamePGN.gameMoves[ply] << " ";

        if ((ply+1) % 8 == 0) {
            os << endl;
        }
    }
    os << gamePGN.result << endl << endl;

    return os;
}

void GamePGN::new_game()
{
    gameMoves.clear();
    result = "?";
}
