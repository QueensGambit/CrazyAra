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
 * @file: rawnetagent.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include <blaze/Math.h>
#include "rawnetagent.h"
#include "../util/blazeutil.h"

using blaze::HybridVector;

RawNetAgent::RawNetAgent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose):
    Agent(net, playSettings, verbose)
{
}

void RawNetAgent::evaluate_board_state()
{
    evalInfo->legalMoves = state->legal_actions();
    evalInfo->init_vectors_for_multi_pv(1UL);

    // sanity check
    assert(evalInfo->legalMoves.size() >= 1);

    // immediately stop the search if there's only one legal move
    if (evalInfo->legalMoves.size() == 1) {
        evalInfo->policyProbSmall.resize(1UL);
        evalInfo->policyProbSmall = 1;
          // a value of 0 is likely a wrong evaluation but won't be written to stdout
        evalInfo->centipawns[0] = value_to_centipawn(0);
        evalInfo->depth = 0;
        evalInfo->nodes = 0;
        evalInfo->pv[0] = {evalInfo->legalMoves[0]};
        return;
    }
    state->get_state_planes(true, inputPlanes, net->get_version());
    net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
    state->set_auxiliary_outputs(auxiliaryOutputs);

    evalInfo->policyProbSmall.resize(evalInfo->legalMoves.size());
    get_probs_of_move_list(0, probOutputs, evalInfo->legalMoves, state->mirror_policy(state->side_to_move()),
                           !net->is_policy_map(), evalInfo->policyProbSmall, net->is_policy_map());
    size_t selIdx = argmax(evalInfo->policyProbSmall);
    Action bestmove = evalInfo->legalMoves[selIdx];

    evalInfo->centipawns[0] = value_to_centipawn(valueOutputs[0]);
    evalInfo->movesToMate[0] = 0;
    evalInfo->depth = 1;
    evalInfo->selDepth = 1;
    evalInfo->tbHits = 0;
    evalInfo->nodes = 1;
    evalInfo->isChess960 = state->is_chess960();
    evalInfo->pv[0] = { bestmove };
}

void RawNetAgent::stop()
{
    // pass
}

void RawNetAgent::apply_move_to_tree(Action move, bool ownMove)
{
    // pass
}
