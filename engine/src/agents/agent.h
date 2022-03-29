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

#include "../stateobj.h"
#include "../evalinfo.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#ifdef USE_RL
#include "../rl/traindataexporter.h"
#endif
#include "nn/neuralnetapiuser.h"

namespace crazyara {
/**
 * @brief The Agent class defines a generic agent interface which use to find the best move.
 * It is assumed that the agent uses a neural network in some way,
 * therefore it inherits from NeuralNetAPIUser.
 */
class Agent : public NeuralNetAPIUser
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
    StateObj* state;
    EvalInfo* evalInfo;
    // protect the isRunning attribute and makes sure that the stop() command can only be called after the search has actually been started
    mutex runnerMutex;
    bool verbose;
    // boolean which can be triggered by "stop" from std-in to stop the current search
    bool isRunning;

public:
    Agent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose);

    /**
     * @brief perform_action Selects an action based on the evaluation result
     */
    virtual void perform_action();

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
    void set_search_settings(StateObj *state, SearchLimits* searchLimits, EvalInfo* evalInfo);

    /**
     * @brief stop Stops the current search is called after "stop" from the stdin
     */
    virtual void stop() = 0;

    /**
     * @brief apply_move_to_tree Applies the given move to the search tree by adding the expanded node to the candidate list
     * @param move Move which has been played
     * @param ownMove Boolean indicating if it was CrazyAra's move
     */
    virtual void apply_move_to_tree(Action move, bool ownMove) = 0;

    /**
     * @brief get_best_action Returns the best action. It is assumed this function gets called after the search.
     * @return Action
     */
    Action get_best_action();

    void lock();

    void unlock();
};
}

void run_agent_thread(crazyara::Agent* agent);

/**
 * @brief apply_quantile_clipping Sets all value in the given quantile to 0
 * @param quantile Quantile specification (assumed to be in [0,1])
 * @param policyProbSmall Policy to be modified
 */
void apply_quantile_clipping(float quantile, DynamicVector<double>& policyProbSmall);


#endif // AGENT_H
