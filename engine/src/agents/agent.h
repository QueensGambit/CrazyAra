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
    void set_best_move(EvalInfo& evalInfo, size_t moveCounter);

protected:
    SearchLimits* searchLimits;
    PlaySettings* playSettings;
    bool verbose;

public:
    Agent(PlaySettings* playSettings, bool verbose);

    /**
     * @brief perform_action Selects an action based on the evaluation result
     * @param pos Board position to evaluate
     * @param limits Pointer to the search limit
     * @param evalInfo Returns the evaluation information
     */
    void perform_action(Board *pos, SearchLimits* searchLimits, EvalInfo& evalInfo);

    /**
     * @brief evalute_board_state Pure virtual method which acts as an interface for all agents
     * @param pos Board position to evaluate
     * @param evalInfo Returns the evaluation information
     */
    virtual void evaluate_board_state(Board *pos,  EvalInfo& evalInfo) = 0;
};

#endif // AGENT_H
