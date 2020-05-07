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
 * @file: threadmanager.h
 * Created on 23.04.2020
 * @author: queensgambit
 *
 * Manages all search threads
 */

#ifndef THREADMANAGER_H
#define THREADMANAGER_H

#include <vector>
#include <thread>
#include "../searchthread.h"
#include "../agents/util/loggerthread.h"
#include "../util/killablethread.h"
#include <condition_variable>

using namespace std;

/**
 * @brief The ThreadManager class contains a reference to all search threads and loggerThreads can trigger early stopping
 * or a stop when the given search time has been reached.
 */
class ThreadManager : public KillableThread
{
private:
    Node* rootNode;
    vector<SearchThread*> searchThreads;
    LoggerThread* loggerThread;
    size_t movetimeMS;
    size_t remainingMoveTimeMS;
    size_t updateIntervalMS;
    float overallNPS;
    float lastValueEval;

    int checkedContinueSearch = 0;
    bool isRunning;
    /**
     * @brief check_early_stopping Checks if the search can be ended prematurely based on the current tree statistics (visits & Q-values)
     * @return True, if early stopping is recommended
     */
    inline bool early_stopping();

    /**
     * @brief continue_search Checks if the search should which is based on the initial value prediciton
     * @return True, if search extension is recommend
     */
    inline bool continue_search();

public:
    ThreadManager(Node* rootNode, vector<SearchThread*>& searchThreads, LoggerThread* loggerThread, size_t movetimeMS, size_t updateIntervalMS, float overallNPS, float lastValueEval);

    /**
    * @brief stop_search_based_on_limits Checks for the search limit condition and possible early break-ups
    * and stops all running search threads accordingly
    * @param evalInfo Evaluation struct which updated during search
    */
    void stop_search_based_on_limits();

    /**
     * @brief stop_search_based_on_kill_event Locks the thread until the kill event was triggerend and
     *  stops all running search threads afterwards
     */
    void stop_search_based_on_kill_event();

    /**
     * @brief stop Stops the current thread
     */
    void stop();

    /**
     * @brief stop_search Stops the current mcts search
     */
    void stop_search();

    /**
     * @brief await_kill_signal Locks the thread until kill signal was received
     */
    void await_kill_signal();

    size_t get_movetime_ms() const;
};

/**
 * @brief run_thread_manager Runner function to start the thread manager
 * @param t Thread manager object
 */
void run_thread_manager(ThreadManager* t);

/**
 * @brief stop_search_threads Stops all search threads in the given list
 * @param searchThreads Vector of mcts search threads
 */
void stop_search_threads(vector<SearchThread*>& searchThreads);

#endif // THREADMANAGER_H
