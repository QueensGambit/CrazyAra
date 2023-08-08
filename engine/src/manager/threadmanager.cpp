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

ThreadManager::ThreadManager(ThreadManagerData* tData, ThreadManagerInfo* tInfo, ThreadManagerParams* tParams):
    tData(tData),
    tInfo(tInfo),
    tParams(tParams),
    checkedContinueSearch(0),
    isRunning(true)
{
}

void ThreadManager::print_info()
{
    tData->evalInfo->end = chrono::steady_clock::now();
    update_eval_info(*tData->evalInfo, tData->rootNode, get_tb_hits(tData->searchThreads), get_max_depth(tData->searchThreads), tInfo->searchSettings);
    info_msg(*tData->evalInfo);
}

void ThreadManager::await_kill_signal()
{
    while(isRunning && tData->searchThreads.front()->is_running()) {
        if (wait_for(chrono::milliseconds(tParams->updateIntervalMS*4))){
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
        tData->remainingMoveTimeMS = tParams->moveTimeMS;
        for (int var = 0; var < tParams->moveTimeMS / tParams->updateIntervalMS && isRunning; ++var) {
            if (wait_for(chrono::milliseconds(tParams->updateIntervalMS))){
                tData->remainingMoveTimeMS -= tParams->updateIntervalMS;
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

    if (!wait_for(chrono::milliseconds(tParams->moveTimeMS % tParams->updateIntervalMS))){
        return;
    }
}

void ThreadManager::stop()
{
    isRunning = false;
}

size_t ThreadManager::get_movetime_ms() const
{
    return tParams->moveTimeMS;
}

bool ThreadManager::isInGame() const
{
    return tParams->inGame;
}

bool ThreadManager::early_stopping()
{
    if (!tParams->inGame) {
        return false;
    }

    if (tInfo->overallNPS == 0) {
        return false;
    }

    if (tData->rootNode->get_node_count() > tInfo->overallNPS * (tParams->moveTimeMS / 1000.0f) * 2 &&
            tData->rootNode->max_q_child() == tData->rootNode->max_visits_child()) {
        info_string("Early stopping (max nodes), saved time:", tData->remainingMoveTimeMS);
        return true;
    }

    uint32_t firstMax;
    uint32_t secondMax;
    ChildIdx firstArg;
    ChildIdx secondArg;
    first_and_second_max(tData->rootNode->get_child_number_visits(), ChildIdx(tData->rootNode->get_no_visit_idx()), firstMax, secondMax, firstArg, secondArg);
    const Node* firstNode = tData->rootNode->get_child_node(firstArg);
    const Node* secondNode = tData->rootNode->get_child_node(secondArg);
    if (firstNode != nullptr && firstNode->is_playout_node()) {
        firstMax -= firstNode->get_free_visits();
    }
    if (secondNode != nullptr && secondNode->is_playout_node()) {
        secondMax -= secondNode->get_free_visits();
    }
    if (secondMax + tData->remainingMoveTimeMS * (tInfo->overallNPS / 1000) < firstMax * 2 &&
            tData->rootNode->get_q_value(firstArg) > tData->rootNode->get_q_value(secondArg)) {
        info_string("Early stopping, saved time:", tData->remainingMoveTimeMS);
        return true;
    }
    return false;
}


bool ThreadManager::continue_search() {
    if (!tParams->inGame || !tParams->canProlong || tInfo->overallNPS == 0 || checkedContinueSearch > 1 || !tData->searchThreads.front()->is_running()) {
        return false;
    }
    // make sure not to flag when continuing search
    if (tParams->moveTimeMS * 2 > tInfo->searchLimits->get_safe_remaining_time(tInfo->sideToMove)) {
        return false;
    }
    const float newEval = tData->rootNode->updated_value_eval();
    if (newEval < tData->lastValueEval) {
        if (tData->remainingMoveTimeMS < tParams->updateIntervalMS + tInfo->searchLimits->moveOverhead) {
            return false;
        }
        info_string("Increase search time");
        tData->lastValueEval = newEval;
        ++checkedContinueSearch;
        return true;
    }
    return false;
}

void ThreadManager::stop_search()
{
    stop_search_threads(tData->searchThreads);
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
    return 0; // TODO
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
