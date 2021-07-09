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
 * @file: mctsagentbatch.h
 * Created on 05.2021
 * @author: BluemlJ
 *
 * The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
 * The final move is chosen according to the visit count of each direct child node and optionally taken Q-values into account.
 * One playout is defined as expanding one new node in the tree.
 * In the case of chess this means evaluating a new board position.
 * For more details and the mathematical equations please refer to our Journal Paper:
 * https://arxiv.org/abs/1908.06660 as well as the official DeepMind-papers.

 * This MCTSAgent starts several MCTSAgents after another and calculates the best move based on all of the MCTSAgents. 
 */

#ifndef MCTSAGENTBATCH_H
#define MCTSAGENTBATCH_H

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

class MCTSAgentBatch : public MCTSAgent
{
public:

  // how many trees should be generated
  int numberOfAgents;
  // boolean, deciding if the given nodes are player per tree or are split between the trees
  bool splitNodes;

public:
    MCTSAgentBatch(NeuralNetAPI* netSingle,
              vector<unique_ptr<NeuralNetAPI>>& netBatches,
              SearchSettings* searchSettings,
              PlaySettings* playSettings,
              int iterations,
              bool splitNodes);
    ~MCTSAgentBatch();
    MCTSAgentBatch(const MCTSAgentBatch&) = delete;
    MCTSAgentBatch& operator=(MCTSAgentBatch const&) = delete;

    // overrides the name of the agent
    string get_name() const override;
    // overrides the evaluation method to combine the results of multiple MCTS agents
    void evaluate_board_state() override;


};


#endif // MCTSAGENTBATCH_H
