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
 * @file: agent.h
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Abstract class for defining a playing agent.
 */

#ifndef AGENT_H
#define AGENT_H

#include "../board.h"
#include "../evalinfo.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#ifdef USE_RL
#include "../rl/traindataexporter.h"
#endif

class Agent
{
private:
    /**
     * @brief set_best_move Sets the "best" (chosen) move by the engine to the evalInformation
     * @param evalInfo Evaluation information
     * @param moveCounter Current move counter (ply//2)
     */
    void set_best_move(size_t moveCounter);

protected:
    SearchLimits* searchLimits;
    PlaySettings* playSettings;
    Board* pos;
    EvalInfo* evalInfo;
    bool verbose;

public:
    Agent(PlaySettings* playSettings, bool verbose);

    /**
     * @brief perform_action Selects an action based on the evaluation result
     */
    void perform_action();

    /**
     * @brief evalute_board_state Pure virtual method which acts as an interface for all agents
     * @param pos Board position to evaluate
     * @param evalInfo Returns the evaluation information
     */
    virtual void evaluate_board_state() = 0;

    /**
     * @brief set_search_settings Sets all relevant parameters for the next search
     * @param pos Board position to evaluate
     * @param limits Pointer to the search limit
     * @param evalInfo Returns the evaluation information
     */
    void set_search_settings(Board *pos, SearchLimits* searchLimits, EvalInfo* evalInfo);

    /**
     * @brief stop Stops the current search is called after "stop" from the stdin
     */
    virtual void stop() = 0;

    /**
     * @brief apply_move_to_tree Applies the given move to the search tree by adding the expanded node to the candidate list
     * @param move Move which has been played
     * @param ownMove Boolean indicating if it was CrazyAra's move
     */
    virtual void apply_move_to_tree(Move move, bool ownMove) = 0;

    Move get_best_move();
};

void run_agent_thread(Agent* agent);

/**
 * @brief apply_quantile_clipping Sets all value in the given quantile to 0
 * @param quantile Quantile specification (assumed to be in [0,1])
 * @param policyProbSmall Policy to be modified
 */
void apply_quantile_clipping(float quantile, DynamicVector<double>& policyProbSmall);

#endif // AGENT_H
