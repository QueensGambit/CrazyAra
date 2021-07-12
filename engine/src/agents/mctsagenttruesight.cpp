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
 * @file: mctsagenttruesight.cpp
 * Created on 05.2021
 * @author: BluemlJ
 */

#include <string>
#include <thread>
#include <fstream>
#include "mctsagenttruesight.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"
#include "util/gcthread.h"


MCTSAgentTrueSight::MCTSAgentTrueSight(NeuralNetAPI *netSingle, vector<unique_ptr<NeuralNetAPI>>& netBatches,
                     SearchSettings* searchSettings, PlaySettings* playSettings):
    MCTSAgent(netSingle, netBatches, searchSettings, playSettings)
    {
        
    }

MCTSAgentTrueSight::~MCTSAgentTrueSight()
{
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

string MCTSAgentTrueSight::get_name() const
{   
   return "MCTSTrueSight-" + engineVersion + "-" + net->get_model_name();
}

void MCTSAgentTrueSight::evaluate_board_state()
{
    evalInfo->nodesPreSearch = init_root_node(state);

    thread tGCThread = thread(run_gc_thread, &gcThread);
    evalInfo->isChess960 = state->is_chess960();
    #ifdef MODE_STRATEGO
        rootState = unique_ptr<StateObj>(state->openBoard());
    #else
        rootState = unique_ptr<StateObj>(state->clone());
    #endif
    
    if (rootNode->get_number_child_nodes() == 1) {
        info_string("Only single move available -> early stopping");
    }
    else if (rootNode->get_number_child_nodes() == 0) {
        info_string("The given position has no legal moves");
    }
    else {
        if (searchSettings->dirichletEpsilon > 0.009f) {
            info_string("apply dirichlet noise");
            // TODO: Check for dirichlet compability
            rootNode->apply_dirichlet_noise_to_prior_policy(searchSettings);
            rootNode->fully_expand_node();
        }

        if (!rootNode->is_root_node()) {
            rootNode->make_to_root();
        }
        info_string("run mcts search");
        run_mcts_search();
        update_stats();
    }
    update_eval_info(*evalInfo, rootNode.get(), tbHits, maxDepth, searchSettings);
    lastValueEval = evalInfo->bestMoveQ[0];
    update_nps_measurement(evalInfo->calculate_nps());
    tGCThread.join();
}
