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
 * @file: alphabetaagent.h
 * Created on 14.03.2021
 * @author: queensgambit
 *
 * The AlphaBetaAgent runs  an mini-max search with alpha beta pruning.
 * It makes use of the policy for move ordering as well as skipping moves at earlier depths.
 */

#ifndef ALPHABETAAGENT_H
#define ALPHABETAAGENT_H

#include "agent.h"

using namespace crazyara;


/**
 * @brief The ActionTrajectory struct stores a trajectory of actions.
 * Based on Datastructure as in https://www.chessprogramming.org/Principal_Variation.
 */
struct ActionTrajectory {
    const static uint maxCapacity = 64U;
    int nbMoves = 0;
    Action moves[maxCapacity];
};


class AlphaBetaAgent : public Agent
{
private:
    // boolean which can be triggered by "stop" from std-in to stop the current search
    bool isRunning;

public:
    AlphaBetaAgent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose);

    // Agent interface
public:
    void evaluate_board_state();
    void stop();
    void apply_move_to_tree(Action move, bool ownMove);

    /**
     * @brief negamax Evaluates all nodes at a given depth and back-propagates their values to their respective parent nodes.
     * @param state Current state object
     * @param depth Current search depth
     * @param alpha Current alpha value which is used for pruning
     * @param beta Current beta value which is used for pruning
     * @param color Integer color value 1 for white, -1 for black
     * @param allMoves All possible moves
     * @þaram inChecks Indicates if the player is in check for the current position
     * @return Negamax value
     */
    float negamax(StateObj* state, int depth, ActionTrajectory * pline, float alpha=-__FLT_MAX__, float beta=__FLT_MAX__, SideToMove color=1, bool allMoves=1, bool inCheck=false);

};

#endif // ALPHABETAAGENT_H
