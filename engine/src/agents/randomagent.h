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
 * @file: mctsagentrandom.h
 * Created on 05.2021
 * @author: BluemlJ
 *
 * This agents plays random moves and has nothing todo with MCTS. It inherits from the MCTSAgent to be used in his place.
 * TODO: Remove the MCTSAgent Inheritance and let him inherit directly from agent
 */

#ifndef MCTSAGENTRANDOM_H
#define MCTSAGENTRANDOM_H

#include "mctsagent.h"
#include "../evalinfo.h"
#include "../node.h"
#include "../stateobj.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "../searchthread.h"
#include "../manager/timemanager.h"
#include "../manager/threadmanager.h"
#include "util/gcthread.h"


using namespace crazyara;

class MCTSAgentRandom : public MCTSAgent
{
public:

public:
    MCTSAgentRandom(NeuralNetAPI* netSingle,
              vector<unique_ptr<NeuralNetAPI>>& netBatches,
              SearchSettings* searchSettings,
              PlaySettings* playSettings);
    ~MCTSAgentRandom();
    MCTSAgentRandom(const MCTSAgentRandom&) = delete;
    MCTSAgentRandom& operator=(MCTSAgentRandom const&) = delete;

    //void evaluate_board_state() override;
    string get_name() const override;
    void perform_action() override;

};


#endif // MCTSAGENTRANDOM_H
