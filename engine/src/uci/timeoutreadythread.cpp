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
 * @file: timeoutreadythread.cpp
 * Created on 05.05.2021
 * @author: queensgambit
 */

#include "timeoutreadythread.h"

void TimeOutReadyThread::print_is_ready()
{
    isRunning = true;
    size_t remainingMoveTimeMS = timeOutMS;
    remainingMoveTimeMS = timeOutMS;
    if (wait_for(chrono::milliseconds(timeOutMS))){
        if (isRunning) {
            cout << "readyok" << endl;
            hasReplied = true;
        }
    }
}

bool TimeOutReadyThread::has_replied()
{
    return hasReplied;
}

void run_timeout_thread(TimeOutReadyThread* t) {
    t->print_is_ready();
}
