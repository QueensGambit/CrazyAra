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


TimeManager::TimeManager(float randomMoveFactor, int expectedGameLength, int threshMove, int timePropMovesToGo, float incrementFactor):
    curMovetime(0),  // will be updated later
    randomMoveFactor(randomMoveFactor),
    expectedGameLength(expectedGameLength),
    threshMove(threshMove),
    timePropMovesToGo(timePropMovesToGo),
    incrementFactor(incrementFactor)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    srand(unsigned(int(seed)));
    assert(threshMove < expectedGameLength);
}

inline int get_constant_movetime(const SearchLimits* searchLimits, SideToMove me, int movesToGo, float incrementFactor) {
    return searchLimits->get_safe_remaining_time(me) / movesToGo + incrementFactor * searchLimits->inc[me];
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

    if (searchLimits->movetime != 0) {
        // only return the plain move time substracted by the move overhead
        curMovetime = searchLimits->movetime;
    }
    else if (searchLimits->movestogo != 0) {
        // calculate a constant move time based on increment and moves left
        curMovetime = get_constant_movetime(searchLimits, me, searchLimits->movestogo, incrementFactor);
    }
    else if (searchLimits->time[me] != 0) {
        // calculate a movetime in sudden death mode
        if (moveNumber < threshMove) {
            curMovetime = get_constant_movetime(searchLimits, me, expectedGameLength-moveNumber, incrementFactor);
        }
        else {
            curMovetime = get_constant_movetime(searchLimits, me, timePropMovesToGo, incrementFactor);
        }
    }
    else {
        curMovetime = 1000;
        info_string("No limit specification given, setting movetime[ms] to", curMovetime);
    }

    // substract the move overhead
    curMovetime -= searchLimits->moveOverhead;

    if (curMovetime <= 0) {
        curMovetime = searchLimits->moveOverhead * 2;
    }

    curMovetime = apply_random_factor(curMovetime);

    if (searchLimits->time[me] != 0) {
        // make sure the returned movetime is within bounds
        return min(searchLimits->get_safe_remaining_time(me), curMovetime);
    }
    return curMovetime;
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
    return (double(rand()) / RAND_MAX) * randomMoveFactor * 2 - randomMoveFactor;
}
