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
#include "../util/blazeutil.h"
#include <chrono>

ThreadManager::ThreadManager(Node* rootNode, vector<SearchThread*>& searchThreads, LoggerThread* loggerThread, size_t movetimeMS, size_t updateIntervalMS, float overallNPS, float lastValueEval, bool inGame, bool canProlong):
    rootNode(rootNode),
    searchThreads(searchThreads),
    loggerThread(loggerThread),
    movetimeMS(movetimeMS),
    remainingMoveTimeMS(movetimeMS),
    updateIntervalMS(updateIntervalMS),
    overallNPS(overallNPS),
    lastValueEval(lastValueEval),
    checkedContinueSearch(0),
    inGame(inGame),
    canProlong(canProlong),
    isRunning(true)
{
}

void ThreadManager::await_kill_signal()
{
    wait_for(chrono::milliseconds(INT_MAX));
}

void run_thread_manager(ThreadManager* t)
{
    if (t->get_movetime_ms() == 0) {
        t->stop_search_based_on_kill_event();
    }
    else {
        t->stop_search_based_on_limits();
    }
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

void ThreadManager::stop_search_based_on_kill_event()
{
    await_kill_signal();
    stop_search();
}

void ThreadManager::stop()
{
    isRunning = false;
}

size_t ThreadManager::get_movetime_ms() const
{
    return movetimeMS;
}

bool ThreadManager::isInGame() const
{
    return inGame;
}

bool ThreadManager::early_stopping()
{
    if (!inGame) {
        return false;
    }

    if (overallNPS == 0) {
        return false;
    }

    if (rootNode->get_visits()-rootNode->get_terminal_visits() > overallNPS * (movetimeMS / 1000.0f) * 2 &&
        rootNode->max_q_child() == rootNode->max_visits_child()) {
        info_string("Early stopping (max nodes), saved time:", remainingMoveTimeMS);
        return true;
    }

    float firstMax;
    float secondMax;
    size_t firstArg;
    size_t secondArg;
    first_and_second_max(rootNode->get_child_number_visits(), rootNode->get_no_visit_idx(), firstMax, secondMax, firstArg, secondArg);
    firstMax -= rootNode->get_child_node(firstArg)->get_terminal_visits();
    secondMax -= rootNode->get_child_node(secondArg)->get_terminal_visits();
    if (secondMax + remainingMoveTimeMS * (overallNPS / 1000) < firstMax * 2 &&
        rootNode->get_q_value(firstArg) > rootNode->get_q_value(secondArg)) {
        info_string("Early stopping, saved time:", remainingMoveTimeMS);
        return true;
    }
    return false;
}


bool ThreadManager::continue_search() {
    if (!inGame || !canProlong || overallNPS == 0 || checkedContinueSearch > 1) {
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

bool can_prolong_search(size_t curMoveNumber, size_t expectedGameLength)
{
    return curMoveNumber < expectedGameLength;
}
