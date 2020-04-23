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
 * @file: threadmanager.cpp
 * Created on 23.04.2020
 * @author: queensgambit
 */

#include "threadmanager.h"
#include <chrono>

ThreadManager::ThreadManager(Node* rootNode, vector<SearchThread*>& searchThreads, LoggerThread* loggerThread, size_t movetimeMS, size_t updateIntervalMS, float overallNPS, float lastValueEval):
    rootNode(rootNode),
    searchThreads(searchThreads),
    loggerThread(loggerThread),
    movetimeMS(movetimeMS),
    remainingMoveTimeMS(movetimeMS),
    updateIntervalMS(updateIntervalMS),
    overallNPS(overallNPS),
    lastValueEval(lastValueEval),
    checkedContinueSearch(0),
    isRunning(true)
{

}

void run_thread_manager(ThreadManager *t)
{
    t->stop_search_based_on_limits();
}

void ThreadManager::stop_search_based_on_limits()
{
    do {
        remainingMoveTimeMS = movetimeMS;
        for (size_t var = 0; var < movetimeMS / updateIntervalMS && isRunning; ++var) {
            if (wait_for(chrono::milliseconds(updateIntervalMS))){
                remainingMoveTimeMS -= updateIntervalMS;
                if (checkedContinueSearch == 0 && early_stopping() && !continue_search()) {
                    stop_search();
                }
            }
            else {
                return;
            }
        }
    } while(continue_search());

    if (!wait_for(chrono::milliseconds(movetimeMS % updateIntervalMS))){
        return;
    }
    stop_search();
}

void ThreadManager::stop()
{
    isRunning = false;
}

bool ThreadManager::early_stopping()
{
    if (overallNPS == 0) {
        return false;
    }

    if (rootNode->get_visits() > overallNPS * (movetimeMS / 1000.0f) * 1.75) {
        info_string("Early stopping (max nodes), saved time:", remainingMoveTimeMS);
        return true;
    }

    const size_t max_visits = max(rootNode->get_child_number_visits());
    size_t second_max_visits = 0;
    for (float visit : rootNode->get_child_number_visits()) {
        if (visit > second_max_visits && visit != max_visits) {
            second_max_visits = visit;
        }
    }
    if (second_max_visits + remainingMoveTimeMS * (overallNPS / 1000) < max_visits * 1.75) {
        info_string("Early stopping, saved time:", remainingMoveTimeMS);
        return true;
    }
    return false;
}


bool ThreadManager::continue_search() {
    if (overallNPS == 0 || checkedContinueSearch > 1) {
        return false;
    }
    const float newEval = rootNode->updated_value_eval();
    if (newEval < lastValueEval) {
        info_string("Increase search time");
        lastValueEval = newEval;
        ++checkedContinueSearch;
        return true;
    }
    return false;
}

void ThreadManager::stop_search()
{
    stop_search_threads(searchThreads);
    loggerThread->kill();
}

void stop_search_threads(vector<SearchThread*>& searchThreads)
{
    for (auto searchThread : searchThreads) {
        searchThread->stop();
    }
}
