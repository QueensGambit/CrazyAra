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
 * @file: loggerthread.h
 * Created on 23.04.2020
 * @author: queensgambit
 *
 * The logger thread handles all logging tree functionality of the mcts
 */

#ifndef LOGGERTHREAD_H
#define LOGGERTHREAD_H

#include <vector>
#include <condition_variable>
#include "../../node.h"
#include "../../evalinfo.h"
#include "../../searchthread.h"
#include "../../util/killablethread.h"

using namespace std;

/**
 * @brief The LoggerThread class continues to log the evaluation information until the thread is stopped
 */
class LoggerThread : public KillableThread
{
private:
    Node* rootNode;
    vector<SearchThread*> searchThreads;

    EvalInfo* evalInfo;
    size_t updateIntervalMS;
public:
    /**
     * @brief wait_and_log Logs indefinetly with a certain logging interval until the conditional variable is triggered
     */
    void wait_and_log();

    LoggerThread(Node* rootNode, EvalInfo* evalInfo, size_t updateIntervalMS, vector<SearchThread*>& searchThreads);
};

/**
 * @brief run_logger_thread Runner function for the LoggerThread
 * @param t logger thread object
 */
void run_logger_thread(LoggerThread *t);

/**
 * @brief get_tb_hits Returns the number of current table base hits during search
 * @param searchThreads MCTS search threads
 * @return number of table base hits
 */
size_t get_tb_hits(const vector<SearchThread*>& searchThreads);

size_t get_avg_depth(const vector<SearchThread*>& searchThreads);
size_t get_max_depth(const vector<SearchThread*>& searchThreads);

#endif // LOGGERTHREAD_H
