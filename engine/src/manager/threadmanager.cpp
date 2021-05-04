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

ThreadManager::ThreadManager(Node* rootNode, EvalInfo* evalInfo, vector<SearchThread*>& searchThreads, size_t movetimeMS, size_t updateIntervalMS, size_t moveOverhead, const SearchSettings* searchSettings, float overallNPS, float lastValueEval, bool inGame, bool canProlong):
    rootNode(rootNode),
    evalInfo(evalInfo),
    searchThreads(searchThreads),
    movetimeMS(movetimeMS),
    remainingMoveTimeMS(movetimeMS),
    updateIntervalMS(updateIntervalMS),
    moveOverhead(moveOverhead),
    searchSettings(searchSettings),
    overallNPS(overallNPS),
    lastValueEval(lastValueEval),
    checkedContinueSearch(0),
    inGame(inGame),
    canProlong(canProlong),
    isRunning(true)
{
}

void ThreadManager::print_info()
{
    evalInfo->end = chrono::steady_clock::now();
    update_eval_info(*evalInfo, rootNode, get_tb_hits(searchThreads), get_max_depth(searchThreads), searchSettings);
    info_msg(*evalInfo);
}

void ThreadManager::await_kill_signal()
{
    while(isRunning && searchThreads.front()->is_running()) {
        if (wait_for(chrono::milliseconds(updateIntervalMS*4))){
            print_info();
        }
        else {
            return;
        }
    }
}

void run_thread_manager(ThreadManager* t)
{
    if (t->get_movetime_ms() == 0) {
        t->await_kill_signal();
    }
    else {
        t->stop_search_based_on_limits();
    }
    t->stop_search();
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
                // log every fourth iteration
                if (var % 4 == 3) {
                    print_info();
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

    if (rootNode->get_node_count() > overallNPS * (movetimeMS / 1000.0f) * 2 &&
            rootNode->max_q_child() == rootNode->max_visits_child()) {
        info_string("Early stopping (max nodes), saved time:", remainingMoveTimeMS);
        return true;
    }

    uint32_t firstMax;
    uint32_t secondMax;
    size_t firstArg;
    size_t secondArg;
    first_and_second_max(rootNode->get_child_number_visits(), rootNode->get_no_visit_idx(), firstMax, secondMax, firstArg, secondArg);
    const Node* firstNode = rootNode->get_child_node(firstArg);
    const Node* secondNode = rootNode->get_child_node(secondArg);
    if (firstNode != nullptr && firstNode->is_playout_node()) {
        firstMax -= firstNode->get_free_visits();
    }
    if (secondNode != nullptr && secondNode->is_playout_node()) {
        secondMax -= secondNode->get_free_visits();
    }
    if (secondMax + remainingMoveTimeMS * (overallNPS / 1000) < firstMax * 2 &&
            rootNode->get_q_value(firstArg) > rootNode->get_q_value(secondArg)) {
        info_string("Early stopping, saved time:", remainingMoveTimeMS);
        return true;
    }
    return false;
}


bool ThreadManager::continue_search() {
    if (!inGame || !canProlong || overallNPS == 0 || checkedContinueSearch > 1 || !searchThreads.front()->is_running()) {
        return false;
    }
    const float newEval = rootNode->updated_value_eval();
    if (newEval < lastValueEval) {
        if (remainingMoveTimeMS < updateIntervalMS + moveOverhead) {
            return false;
        }
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

size_t get_avg_depth(const vector<SearchThread*>& searchThreads)
{
    size_t avgDetph = 0;
    for (SearchThread* searchThread : searchThreads) {
        avgDetph += searchThread->get_avg_depth();
    }
    avgDetph = size_t(double(avgDetph) / searchThreads.size() + 0.5);
    return avgDetph;
}

size_t get_max_depth(const vector<SearchThread*>& searchThreads)
{
    size_t maxDepth = 0;
    for (SearchThread* searchThread : searchThreads) {
        maxDepth = max(maxDepth, searchThread->get_max_depth());
    }
    return maxDepth;
}

size_t get_tb_hits(const vector<SearchThread*>& searchThreads)
{
    size_t tbHits = 0;
    for (SearchThread* searchThread : searchThreads) {
        tbHits += searchThread->get_tb_hits();
    }
    return tbHits;
}
