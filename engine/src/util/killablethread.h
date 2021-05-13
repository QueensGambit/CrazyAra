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
 * @file: stoppablethread.h
 * Created on 23.04.2020
 * @author: queensgambit
 *
 * Abstract class for a thread object which can be awakened during a waiting block from a different thread.
 * The class is modeled after a post by Yakk - Adam Nevraumont:
 * https://stackoverflow.com/questions/29775153/stopping-long-sleep-threads
 */

#ifndef STOPPABLETHREAD_H
#define STOPPABLETHREAD_H

#include <condition_variable>
#include <mutex>

using namespace std;

/**
 * @brief The KillableThread class can be awakaned from an external thread due to the conditional variable
 */
class KillableThread
{
protected:
    mutable condition_variable cv;
    mutable mutex mtx;
    bool isRunning = true;
    bool terminate = false;

public:
    /**
     * @brief wait_for Waits for a given time but can be interrupted by a kill() call from an external thread
     * @param time Amount of time to wait
     * @return True if the time elapsed and false if it was triggered by kill()
     */
    template<class R, class P>
    bool wait_for( std::chrono::duration<R,P> const& time ) const {
        unique_lock<std::mutex> lock(mtx);
        return !cv.wait_for(lock, time, [&]{return terminate;});
    }

    /**
     * @brief kill Kills the current thread by triggering the conditional variable
     */
    void kill() {
        unique_lock<std::mutex> lock(mtx);
        terminate = true;
        cv.notify_all();
        isRunning = false;
    }
    /**
     * @brief stop Stops the current thre without triggering the conditional variable
     */
    void stop() {
        isRunning = false;
    }

    KillableThread() = default;
    KillableThread(KillableThread&&)=delete;
    KillableThread(KillableThread const&)=delete;
    KillableThread& operator=(KillableThread&&)=delete;
    KillableThread& operator=(KillableThread const&)=delete;
};

#endif // STOPPABLETHREAD_H
