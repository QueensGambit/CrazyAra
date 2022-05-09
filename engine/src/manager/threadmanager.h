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
 * Manages all search threads and logs intermediate search results
 */

#ifndef THREADMANAGER_H
#define THREADMANAGER_H

#include <vector>
#include <thread>
#include <condition_variable>
#include "../searchthread.h"
#include "../evalinfo.h"
#include "../util/killablethread.h"

using namespace std;


struct ThreadManagerData {
    const Node* rootNode;
    vector<SearchThread*> searchThreads;
    EvalInfo* evalInfo;
    int remainingMoveTimeMS;
    float lastValueEval;

    ThreadManagerData(const Node* rootNode, vector<SearchThread*> searchThreads, EvalInfo* evalInfo, float lastValueEval) :
        rootNode(rootNode), searchThreads(searchThreads), evalInfo(evalInfo), remainingMoveTimeMS(0), lastValueEval(lastValueEval)
    {}
};

struct ThreadManagerInfo {
    const SearchSettings* searchSettings;
    const SearchLimits* searchLimits;
    const float overallNPS;
    const SideToMove sideToMove;

    ThreadManagerInfo(const SearchSettings* searchSettings, const SearchLimits* searchLimits, const float overallNPS, const SideToMove sideToMove) :
        searchSettings(searchSettings), searchLimits(searchLimits), overallNPS(overallNPS), sideToMove(sideToMove)
    {}
};

struct ThreadManagerParams {
    const int moveTimeMS;
    const int updateIntervalMS;
    const bool inGame;
    const bool canProlong;

    ThreadManagerParams(const int moveTimeMS, const int updateIntervalMS, const bool inGame, const bool canProlong) :
        moveTimeMS(moveTimeMS), updateIntervalMS(updateIntervalMS), inGame(inGame), canProlong(canProlong)
    {}
};

/**
 * @brief The ThreadManager class contains a reference to all search threads and can trigger early stopping
 * or a stop when the given search time has been reached. It also logs intermdediate search results.
 */
class ThreadManager : public KillableThread
{
private:
    ThreadManagerData* tData;
    ThreadManagerInfo* tInfo;
    ThreadManagerParams* tParams;
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

    /**
     * @brief print_info Updates and prints the uci eval info to stdout
     */
    void print_info();

public:
    ThreadManager(ThreadManagerData* tData, ThreadManagerInfo* tInfo, ThreadManagerParams* tParams);

    /**
    * @brief stop_search_based_on_limits Checks for the search limit condition and possible early break-ups
    * and stops all running search threads accordingly
    * @param evalInfo Evaluation struct which updated during search
    */
    void stop_search_based_on_limits();

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
    bool isInGame() const;
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

/**
 * @brief can_prolong_search Returns true if it is allowed to prolong the current search
 * @param curMoveNumber Current move number (plies / 2)
 * @param expectedGameLength Expected game length
 * @return True, if search can be extended else false
 */
bool can_prolong_search(size_t curMoveNumber, size_t expectedGameLength);

/**
 * @brief get_tb_hits Returns the number of current table base hits during search
 * @param searchThreads MCTS search threads
 * @return number of table base hits
 */
size_t get_tb_hits(const vector<SearchThread*>& searchThreads);

/**
 * @brief get_avg_depth Returns the average number of search depth in the tree
 * @param searchThreads MCTS search threads
 * @return average depth for all simulations
 */
size_t get_avg_depth(const vector<SearchThread*>& searchThreads);

/**
 * @brief get_max_depth Returns the maximum reached search depth for all threads
 * @param searchThreads MCTS search threads
 * @return maximum reached depth
 */
size_t get_max_depth(const vector<SearchThread*>& searchThreads);


#endif // THREADMANAGER_H
