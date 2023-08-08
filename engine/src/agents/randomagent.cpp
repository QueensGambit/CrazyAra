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
 * @file: mctsagentrandom.cpp
 * Created on 05.2021
 * @author: BluemlJ
 */

#include <thread>
#include <fstream>
#include <vector>
#include "randomagent.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"
#include "util/gcthread.h"


MCTSAgentRandom::MCTSAgentRandom(NeuralNetAPI *netSingle, vector<unique_ptr<NeuralNetAPI>>& netBatches,
                     SearchSettings* searchSettings, PlaySettings* playSettings):
    MCTSAgent(netSingle, netBatches, searchSettings, playSettings)
    {

    }

MCTSAgentRandom::~MCTSAgentRandom()
{
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

string MCTSAgentRandom::get_name() const
{
    return "MCTSRandom";
}

void MCTSAgentRandom::perform_action()
{
    vector<Action> lM  = state->legal_actions();
    if (lM.size() != 0){
        int randomIndex = rand() % lM.size();
        
        info_string(state->fen());
        info_bestmove(state->action_to_san(lM[randomIndex], state->legal_actions(), false, false));
        evalInfo->bestMove = lM[randomIndex];
    }
}
