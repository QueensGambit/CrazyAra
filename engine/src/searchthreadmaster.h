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
 * @file: searchthreadmaster.h
 * Created on 31.10.2021
 * @author: queensgambit
 *
 * Manages a single mini-batch for neural network inference.
 */

#ifndef SEARCHTHREADMASTER_H
#define SEARCHTHREADMASTER_H

#include "nn/neuralnetapiuser.h"
#include "searchthread.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "node.h"

class SearchThreadMaster : NeuralNetAPIUser
{
private:
    int numberChildThreads;
    vector<unique_ptr<SearchThread>> searchThreads;
    vector<unique_ptr<NeuralNetData>> nnData;

    Node* rootNode;
    StateObj* rootState;
    SearchLimits* searchLimits;

    bool isRunning;
    bool reachedTablebases;

    void launch_child_threads();
    void child_threads_backup();
public:
    SearchThreadMaster(NeuralNetAPI *netBatch, int numberChildThreads,
                       const SearchSettings* searchSettings, MapWithMutex* mapWithMutex);
    void thread_iteration();
    bool is_root_node_unsolved();

    bool is_running() const;
    void set_is_running(bool value);
    void reset_stats();
    bool nodes_limits_ok();

    void set_root_node(Node* value);
    void set_root_state(StateObj* value);
    void set_search_limits(SearchLimits* value);
    void set_reached_tablebases(bool value);

    size_t get_tb_hits() const;
    size_t get_avg_depth();
    size_t get_max_depth() const;

    /**
     * @brief stop Stops the rollouts of the current thread
     */
    void stop();
};

void run_search_thread_master(SearchThreadMaster *t);

#endif // SEARCHTHREADMASTER_H
