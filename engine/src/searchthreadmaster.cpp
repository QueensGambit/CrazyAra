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
 * @file: searchthreadmaster.cpp
 * Created on 31.10.2021
 * @author: queensgambit
 */

#include "searchthreadmaster.h"
#include <thread>

void SearchThreadMaster::launch_child_threads()
{
    thread** threads = new thread*[numberChildThreads];
    for (size_t i = 0; i < numberChildThreads; ++i) {
        searchThreads[i]->set_root_node(rootNode);
        searchThreads[i]->set_root_state(rootState);
        searchThreads[i]->set_search_limits(searchLimits);
        searchThreads[i]->set_reached_tablebases(reachedTablebases);
        threads[i] = new thread(run_create_mini_batch, searchThreads[i].get());
    }
    for (size_t i = 0; i < numberChildThreads; ++i) {
        threads[i]->join();
    }
    delete[] threads;
}

void SearchThreadMaster::child_threads_backup()
{
    thread** threads = new thread*[numberChildThreads];
    for (size_t i = 0; i < numberChildThreads; ++i) {
        threads[i] = new thread(run_backup_values, searchThreads[i].get());
    }
    for (size_t i = 0; i < numberChildThreads; ++i) {
        threads[i]->join();
    }
    delete[] threads;
}

SearchThreadMaster::SearchThreadMaster(NeuralNetAPI *netBatch, int numberChildThreads,
                                       const SearchSettings* searchSettings, MapWithMutex* mapWithMutex) :
    NeuralNetAPIUser(netBatch),
    numberChildThreads(numberChildThreads)
{

    const int nbSamples = net->get_batch_size() / numberChildThreads;
    int offset = 0;
    for (int idx = 0; idx < numberChildThreads; ++idx) {
        nnData.emplace_back(make_unique<NeuralNetData>(inputPlanes+offset*net->get_nb_input_values_total()*nbSamples,
                                                       valueOutputs+offset*nbSamples,
                                                       probOutputs+offset*net->get_nb_policy_values()*nbSamples,
                                                       auxiliaryOutputs+offset*net->get_nb_auxiliary_outputs()*nbSamples,
                                                       net, nbSamples));
        searchThreads.emplace_back(make_unique<SearchThread>(nnData[idx].get(), searchSettings, mapWithMutex));
        ++offset;
    }
}

void SearchThreadMaster::thread_iteration()
{
    launch_child_threads();
#ifndef SEARCH_UCT
    net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
#endif
    child_threads_backup();
}

bool SearchThreadMaster::is_root_node_unsolved()
{
#ifdef MCTS_TB_SUPPORT
    return is_unsolved_or_tablebase(rootNode->get_node_type());
#else
    return rootNode->get_node_type() == UNSOLVED;
#endif
}

bool SearchThreadMaster::is_running() const
{
    return isRunning;
}

void SearchThreadMaster::set_is_running(bool value)
{
    isRunning = value;
}

void run_search_thread_master(SearchThreadMaster *t)
{
    t->set_is_running(true);
    t->reset_stats();
    while(t->is_running() && t->nodes_limits_ok() && t->is_root_node_unsolved()) {
        t->thread_iteration();
    }
    t->set_is_running(false);
}

void SearchThreadMaster::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
}

bool SearchThreadMaster::nodes_limits_ok()
{
    return (searchLimits->nodes == 0 || (rootNode->get_node_count() < searchLimits->nodes)) &&
            (searchLimits->simulations == 0 || (rootNode->get_visits() < searchLimits->simulations)) &&
            (searchLimits->nodesLimit == 0 || (rootNode->get_node_count() < searchLimits->nodesLimit));
}

void SearchThreadMaster::set_root_node(Node *value)
{
    rootNode = value;
}

void SearchThreadMaster::set_root_state(StateObj *value)
{
    rootState = value;
}

void SearchThreadMaster::set_search_limits(SearchLimits *value)
{
    searchLimits = value;
}

void SearchThreadMaster::set_reached_tablebases(bool value)
{
    reachedTablebases = value;
}

size_t SearchThreadMaster::get_tb_hits() const
{
    return tbHits;
}

size_t SearchThreadMaster::get_avg_depth()
{
    return 0; // TODO
}

size_t SearchThreadMaster::get_max_depth() const
{
    return 0;  // TODO
}

void SearchThreadMaster::stop()
{
    isRunning = false;
}
