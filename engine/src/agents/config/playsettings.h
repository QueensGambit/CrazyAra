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
 * @file: playsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Struct which defines setting about general playing behaviour
 */

#ifndef PLAYSETTINGS_H
#define PLAYSETTINGS_H

#include <stddef.h>

struct PlaySettings
{
public:
    float temperature;
    size_t temperatureMoves;
    bool useTimeManagement;
    size_t openingGuardMoves;
#ifdef USE_RL
    // mean value of an exponentional distribution about how many samples are directly sampled from the raw NN policy
    size_t meanInitPly;
    // maximum value for randomly sampled plys
    size_t maxInitPly;
#endif
    PlaySettings():
                 temperature(0.0),
                 temperatureMoves(4),
                 useTimeManagement(true),
                 openingGuardMoves(0) {}
};

#endif // PLAYSETTINGS_H
