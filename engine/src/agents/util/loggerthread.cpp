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
 * @file: loggerthread.cpp
 * Created on 23.04.2020
 * @author: queensgambit
 */

#include "loggerthread.h"
#include <thread>
#include <chrono>

LoggerThread::LoggerThread(Node* rootNode, EvalInfo* evalInfo, size_t updateIntervalMS, vector<SearchThread*>& searchThreads):
    KillableThread(),
    rootNode(rootNode),
    searchThreads(searchThreads),
    evalInfo(evalInfo),
    updateIntervalMS(updateIntervalMS)
{

}

void LoggerThread::wait_and_log()
{
    while(isRunning) {
        if (wait_for(chrono::milliseconds(updateIntervalMS))){
            evalInfo->end = chrono::steady_clock::now();
            update_eval_info(*evalInfo, rootNode, get_tb_hits(searchThreads));
            info_score(*evalInfo);
        }
    }
}

void run_logger_thread(LoggerThread *t)
{
    t->wait_and_log();
}

size_t get_tb_hits(const vector<SearchThread*>& searchThreads)
{
    size_t tbHits = 0;
    for (SearchThread* searchThread : searchThreads) {
        tbHits += searchThread->get_tb_hits();
    }
    return tbHits;
}
