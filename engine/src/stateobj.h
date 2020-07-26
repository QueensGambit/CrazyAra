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
 * @file: stateobj.h
 * Created on 17.07.2020
 * @author: queensgambit
 *
 * This file defines the wrapper function and classes which are used during MCTS.
 * Edit this file and its corresponding source file to activate a custom environment.
 */

#ifndef STATEOBJ_H
#define STATEOBJ_H

#include "state.h"
#ifdef MODE_POMMERMAN
#include "pommermanstate.h"
#else
#include "boardstate.h"
#endif

#ifdef MODE_POMMERMAN
    using SelectedState = PommermanState;
#else
    using SelectedState = BoardState;
#endif
using StateObj = State<SelectedState>;

std::string action_to_uci(Action action, bool is960);

#endif // STATEOBJ_H
