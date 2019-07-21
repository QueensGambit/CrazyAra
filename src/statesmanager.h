/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
 * @file: statesmanager.h
 * Created on 04.07.2019
 * @author: queensgambit
 *
 * The state manager holds the state sequence for all past game moves.
 * This is needed for 3-fold repeptition detection.
 * In the case the MCTS tree gets resued the passive states vector
 * becomes the active states vector again by calling swap_states().
 *
 * Keeps track of the states list of the current new game history generated in the uci-loop
 * and the old states which are connected to the current search tree.
 */

#ifndef STATESMANAGER_H
#define STATESMANAGER_H

using namespace std;
#include "position.h"

class StatesManager
{
private:
    // the passive states vector stores the previously used state vector. This is needed if the tree is reused so the
    // nodes in the tree point to the correct state information which are still in memory.
    vector<StateInfo*> passiveStates;
public:
    StatesManager();

    /**
     * @brief swap_states Swaps the active states with the passive states
     */
    void swap_states();

    /**
     * @brief clear_states Cleares the active states vector and all the memory it points to
     */
    void clear_states();

    // the active states vector stores the state information which are currently used for the 3-fold-repetition check
    vector<StateInfo*> activeStates;
};

#endif // STATESMANAGER_H
