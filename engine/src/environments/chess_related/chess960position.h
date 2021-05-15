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
 * @file: chess960position.h
 * Created on 25.04.2020
 * @author: queensgambit
 *
 * http://rosettacode.org/wiki/Generate_Chess960_starting_position
 */

#ifndef CHESS960POSITION_H
#define CHESS960POSITION_H

class Chess960Position
{
public:
    Chess960Position();
};

#include <iostream>
#include <string>
#include <time.h>
#include <cstring>
using namespace std;

namespace
{
    void placeRandomly(char* p, char c)
    {
        int loc = rand() % 8;
        if (!p[loc])
            p[loc] = c;
        else
            placeRandomly(p, c);    // try again
    }
    int placeFirst(char* p, char c, int loc = 0)
    {
        while (p[loc]) ++loc;
        p[loc] = c;
        return loc;
    }

    string startPos()
    {
        char p[8]; memset( p, 0, 8 );

        // bishops on opposite color
        p[2 * (rand() % 4)] = 'B';
        p[2 * (rand() % 4) + 1] = 'B';

        // queen knight knight, anywhere
        for (char c : "QNN")
            placeRandomly(p, c);

        // rook king rook, in that order
        placeFirst(p, 'R', placeFirst(p, 'K', placeFirst(p, 'R')));

        return string(p, 8);
    }

    string chess960fen()
    {
        string firstRank = startPos();
        string lastRank = string(firstRank);
        std::transform(firstRank.begin(), firstRank.end(), firstRank.begin(), ::tolower);
        const string fen = firstRank + "/pppppppp/8/8/8/8/PPPPPPPP/" + lastRank +  " w KQkq - 0 1";
        return fen;
    }
}   // leave local

#endif // CHESS960POSITION_H
