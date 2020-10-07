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
 * @file: rawnetagent.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * The raw network uses the the single network prediction for it's evaluation.
 * No mcts search is being done.
 */

#ifndef RAWNETAGENT_H
#define RAWNETAGENT_H

#include "../evalinfo.h"
#include "../node.h"
#include <thread>
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "agent.h"

using namespace crazyara;

class RawNetAgent: public Agent
{
public:
    RawNetAgent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose);
    RawNetAgent(const RawNetAgent&) = delete;
    RawNetAgent& operator=(RawNetAgent const&) = delete;

    void evaluate_board_state() override;

    void stop() override;

    void apply_move_to_tree(Action move, bool ownMove) override;
};

#endif // RAWNETAGENT_H
