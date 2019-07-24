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
 * @file: timemanager.h
 * Created on 16.07.2019
 * @author: queensgambit
 *
 * The time manager calculates a movetime based on the search limits.
 * The movetime is only a recommendation and can be ignored in cased of early stopping.
 */

#ifndef TIMEMANAGER_H
#define TIMEMANAGER_H

#include "../agents/config/searchlimits.h"

class TimeManager
{
private:
    int curMovetime;
    int timeBuffer;

    int expectedGameLength;
    int threshMove;
    float moveFactor;
    float incrementFactor;
    int timeBufferFactor;
public:

    /**
     * @brief TimeManager
     * @param expectedGameLength Expected game length for the game in full moves
     * @param threshMove Threshold move on which the constant move regime will switch to a proportional one
     * @param moveFactor Portion of the current move time which will be used in the proportional movetime regime
     * @param timeBufferFactor Factor which is applied on the moveOverhead to calculate a time buffer for avoiding losing on time
     */
    TimeManager(int expectedGameLength=50, int threshMove=40, float moveFactor=0.05f, float incrementFactor=0.7f, int timeBufferFactor=30.0f);

    /**
     * @brief get_time_for_move Calculates the movetime based on the searchSettigs
     * It uses a constant movetime for the first moves until the ``threshMove`` is reached.
     * Afterwards it uses a portion of the remaining time as defined in ``moveFact`
     * @param searchLimits Limit specification for the current position
     * @param me Color of the current player
     * @param moveNumber Move number of the position (ply//2)
     * @return movetime in ms
     */
    int get_time_for_move(SearchLimits* searchLimits, Color me, int moveNumber);
};

#endif // TIMEMANAGER_H
