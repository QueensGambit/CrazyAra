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
#include "state.h"

class TimeManager
{
private:
    int curMovetime;
    int timeBuffer;

    float randomMoveFactor;
    int expectedGameLength;
    int threshMove;
    float moveFactor;
    float incrementFactor;
    int timeBufferFactor;

    /**
     * @brief apply_random_factor Applies the current randomly generated move factor on the given movetime.
     * In case the randomMoveFactor is 0.0 the function returns the original curMovetime instead.
     * @param curMovetime Movetime in ms
     * @return adjusted movetime with curMovetime +/- currentRandomMoveFactor * curMovetime
     */
    int inline apply_random_factor(int curMovetime);

    /**
     * @brief get_current_random_factor Return a randomly generated move factor in [-randomMoveFactor, +randomMoveFactor]
     * @return randomly generated factor
     */
    float inline get_current_random_factor();
public:

    /**
     * @brief TimeManager
     * @param randomMoveFactor Factor which is used to apply a slight random modification on the final calculated movetime.
     * (e.g. when randomMoveFactor is 0.1, on every move the movtime will be either increased or decreased by a factor of between -10% and +10%)
     * @param expectedGameLength Expected game length for the game in full moves
     * @param threshMove Threshold move on which the constant move regime will switch to a proportional one
     * @param moveFactor Portion of the current move time which will be used in the proportional movetime regime
     * @param timeBufferFactor Factor which is applied on the moveOverhead to calculate a time buffer for avoiding losing on time
     */
    TimeManager(float randomMoveFactor=0, int expectedGameLength=40, int threshMove=30, float moveFactor=0.05f, float incrementFactor=0.7f, int timeBufferFactor=30.0f);

    /**
     * @brief get_time_for_move Calculates the movetime based on the searchSettigs
     * It uses a constant movetime for the first moves until the ``threshMove`` is reached.
     * Afterwards it uses a portion of the remaining time as defined in ``moveFact``
     * @param searchLimits Limit specification for the current position
     * @param me Color of the current player
     * @param moveNumber Move number of the position (ply//2)
     * @return movetime in ms
     */
    int get_time_for_move(const SearchLimits* searchLimits, SideToMove me, int moveNumber);
    int get_thresh_move() const;
};


#endif // TIMEMANAGER_H
