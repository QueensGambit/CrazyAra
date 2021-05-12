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
 * @file: timeoutreadythread.h
 * Created on 05.05.2021
 * @author: queensgambit
 *
 * Thread which prints out "readyok" after a given amount of ms unless killed.
 * This is to avoid running into time outs of e.g. cutechess on Multi-GPU systems when deserializing complex NN architectures.
 */

#ifndef TIMEOUTREADYTHREAD_H
#define TIMEOUTREADYTHREAD_H

#include <iostream>
#include "util/killablethread.h"

using namespace std;

/**
 * @brief The TimeOutReadyThread class prints out "readyok" after a given amout of "timeOutMS" unless it is killed before.
 */
class TimeOutReadyThread : public KillableThread
{
private:
    size_t timeOutMS;
    bool isRunning;
    bool hasReplied;
public:
    TimeOutReadyThread(size_t timeOutMS) :
        timeOutMS(timeOutMS),
        isRunning(false),
        hasReplied(false) {}

    void print_is_ready();

    /**
     * @brief hasReplied Returns true if the timer has already replied with "readyok", else false
     * @return
     */
    bool has_replied();
};


/**
 * @brief run_timeout_thread Runner function to start the time out thread
 * @param t TimeOutReady thread object
 */
void run_timeout_thread(TimeOutReadyThread* t);


#endif // TIMEOUTREADYTHREAD_H
