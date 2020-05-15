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
 * @file: playsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Struct which defines setting about general playing behaviour
 */

#ifndef PLAYSETTINGS_H
#define PLAYSETTINGS_H

#include <stddef.h>
#include <math.h>

struct PlaySettings
{
public:
    double initTemperature;
    size_t temperatureMoves;
    bool useTimeManagement;
    // exponential decay factor which reduces the intial temperature on every move (every 2nd ply, must be in [0.0,1.0]])
    // plyTemp = initTemp * tempDecay^moveCount
    double temperatureDecayFactor;
    double quantileClipping;
#ifdef USE_RL
    // mean value of an exponentional distribution about how many samples are directly sampled from the raw NN policy
    size_t meanInitPly;
    // maximum value for randomly sampled plys
    size_t maxInitPly;
#endif
    PlaySettings():
                 initTemperature(0.0),
                 temperatureMoves(4),
                 useTimeManagement(true) {}
};

/**
 * @brief get_current_temperature
 * @param play
 * @param ply
 * @return
 */
double get_current_temperature(const PlaySettings& play, size_t moveNumber);


#endif // PLAYSETTINGS_H
