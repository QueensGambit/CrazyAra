/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: timemanager.cpp
 * Created on 16.07.2019
 * @author: queensgambit
 */

#include "timemanager.h"

using namespace std;


TimeManager::TimeManager(int expectedGameLength, int threshMove, float moveFact):
    expectedGameLength(expectedGameLength),
    threshMove(threshMove),
    moveFact(moveFact)
{
    assert(threshMove < expectedGameLength);
}

int TimeManager::get_time_for_move(SearchLimits* searchLimits, Color me, int moveNumber)
{
    // leave an additional time buffer to avoid losing on time
    time_buffer = searchLimits->moveOverhead * 3;

    if (searchLimits->movetime != 0) {
        // only return the plain move time substracted by the move overhead
        curMovetime = searchLimits->movetime - searchLimits->moveOverhead;
    }
    else if (searchLimits->movestogo != 0) {
        // calculate a constant move time based on increment and moves left
        curMovetime = int(((searchLimits->time[me] - time_buffer) / float(searchLimits->movestogo) + 0.5f)
                + searchLimits->inc[me] - searchLimits->moveOverhead);
    }
    else if (searchLimits->time[me] != 0) {
        // calculate a movetime in sudden death mode
        if (moveNumber < threshMove) {
            curMovetime = int((searchLimits->time[me] - time_buffer) / float(expectedGameLength-moveNumber) + 0.5f)
                    + max(searchLimits->inc[me]-time_buffer, 0) - searchLimits->moveOverhead;
        }
        else {
            curMovetime = int((searchLimits->time[me] - time_buffer) * moveFact + 0.5f)
                    + max(searchLimits->inc[me]-time_buffer, 0) - searchLimits->moveOverhead;
        }
    }
    else if (searchLimits->nodes) {
        // TODO
        assert(false);
    }
    else {
        curMovetime = 1000 - searchLimits->moveOverhead;
//        sync_cout << "string info No limit specification given, setting movetime to (1000 - moveOverhead)" << sync_endl;
    }

    assert(curMovetime > 0);
    return curMovetime;
}
