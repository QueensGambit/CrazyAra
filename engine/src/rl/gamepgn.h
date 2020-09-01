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
 * @file: gamepgn.h
 * Created on 16.09.2019
 * @author: queensgambit
 *
 * Structure for exporting a game in PGN format
 */

#ifndef GAMEPGN_H
#define GAMEPGN_H

#include <string>
#include <vector>
#include <iostream>

using namespace std;

struct GamePGN
{
    string variant = "?";
    string fen = "?";
    string event = "?";
    string site = "?";
    string date = "?";
    string round = "?";
    string white = "?";
    string black = "?";
    string result = "?";
//    string plyCount = "?";  // will be computed with gameMoves.size()
    string timeControl = "?";
    vector<string> gameMoves;
    bool is960 = false;
    void new_game();
};

extern std::ostream& operator<<(std::ostream& os, const GamePGN& evalInfo);

#endif // GAMEPGN_H
