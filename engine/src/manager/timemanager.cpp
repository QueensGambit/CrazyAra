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
 * @file: timemanager.cpp
 * Created on 16.07.2019
 * @author: queensgambit
 */

#include "timemanager.h"
#include "../util/communication.h"
#include <cassert>

using namespace std;


TimeManager::TimeManager(float randomMoveFactor, int expectedGameLength, int threshMove, float moveFactor, float incrementFactor, int timeBufferFactor):
    curMovetime(0),  // will be updated later
    timeBuffer(0),   // will be updated later
    randomMoveFactor(randomMoveFactor),
    expectedGameLength(expectedGameLength),
    threshMove(threshMove),
    moveFactor(moveFactor),
    incrementFactor(incrementFactor),
    timeBufferFactor(timeBufferFactor)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    srand(unsigned(int(seed)));
    assert(threshMove < expectedGameLength);
}

int TimeManager::get_time_for_move(const SearchLimits* searchLimits, SideToMove me, int moveNumber)
{
    if (searchLimits->infinite) {
        return 0;
    }
    if (searchLimits->nodes != 0 || searchLimits->simulations != 0 || searchLimits->depth != 0) {
        if (searchLimits->movetime == 0) {
            return 0;
        }
    }

    // leave an additional time buffer to avoid losing on time
    timeBuffer = searchLimits->moveOverhead * timeBufferFactor;

    if (searchLimits->movetime != 0) {
        // only return the plain move time substracted by the move overhead
        curMovetime = searchLimits->movetime - searchLimits->moveOverhead;
    }
    else if (searchLimits->movestogo != 0) {
        // calculate a constant move time based on increment and moves left
        curMovetime = int(((searchLimits->time[me] - timeBuffer) / float(searchLimits->movestogo) + 0.5f)
                + searchLimits->inc[me] - searchLimits->moveOverhead);
    }
    else if (searchLimits->time[me] != 0) {
        // calculate a movetime in sudden death mode
        if (moveNumber < threshMove) {
            curMovetime = int((searchLimits->time[me] - timeBuffer) / float(expectedGameLength-moveNumber) + 0.5f)
                    + int(searchLimits->inc[me] * incrementFactor) - searchLimits->moveOverhead;
        }
        else {
            curMovetime = int((searchLimits->time[me] - timeBuffer) * moveFactor + 0.5f)
                    + int(searchLimits->inc[me] * incrementFactor) - searchLimits->moveOverhead;
        }
    }
    else {
        curMovetime = 1000 - searchLimits->moveOverhead;
        info_string("No limit specification given, setting movetime[ms] to", curMovetime);
    }

    if (curMovetime <= 0) {
        curMovetime = searchLimits->moveOverhead * 2;
    }
    return apply_random_factor(curMovetime);
}

int TimeManager::get_thresh_move() const
{
    return threshMove;
}

int TimeManager::apply_random_factor(int curMovetime)
{
    if (randomMoveFactor > 0) {
        return curMovetime + int(get_current_random_factor() * curMovetime);
    }
    return curMovetime;
}

float TimeManager::get_current_random_factor()
{
    return (float(rand()) / RAND_MAX) * randomMoveFactor * 2 - randomMoveFactor;
}
