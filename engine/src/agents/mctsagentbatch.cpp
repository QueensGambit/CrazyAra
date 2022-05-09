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
 * @file: mctsagentbatch.cpp
 * Created on 05.2021
 * @author: BluemlJ
 * 
 */

#include <string>
#include <thread>
#include <fstream>
#include "mctsagentbatch.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"
#include "util/gcthread.h"


MCTSAgentBatch::MCTSAgentBatch(NeuralNetAPI *netSingle, vector<unique_ptr<NeuralNetAPI>>& netBatches,
                     SearchSettings* searchSettings, PlaySettings* playSettings, int noa, bool sN):
    MCTSAgent(netSingle, netBatches, searchSettings, playSettings)
    {
        numberOfAgents = noa;
        splitNodes = sN;
    }

MCTSAgentBatch::~MCTSAgentBatch()
{
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

string MCTSAgentBatch::get_name() const
{   
    string ret = "MCTSBatch-" + std::to_string(numberOfAgents) + "-" + engineVersion + "-" + net->get_model_name();
    if(splitNodes){
        ret = "MCTSBatch-Split-" + std::to_string(numberOfAgents) + "-" + engineVersion + "-" + net->get_model_name();
    }
    return ret;
}

void MCTSAgentBatch::evaluate_board_state()
{
    vector<EvalInfo> evals;
    evalInfo->isChess960 = state->is_chess960();

    for (size_t i = 0; i < numberOfAgents; i++)
    {
        EvalInfo eval = *evalInfo;
        auto rt = state->clone();
        rootState = unique_ptr<StateObj>(rt);
        eval.nodesPreSearch = init_root_node(rt);
        
        info_string(rootState->fen());
        
        thread tGCThread = thread(run_gc_thread, &gcThread);
        
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
            int tmp = searchLimits->nodes;
            if(splitNodes){  
                searchLimits->nodes = tmp/numberOfAgents;
            }
            info_string("run mcts search");
            run_mcts_search();
            update_stats();
            searchLimits->nodes = tmp;
        }
        
        const size_t targetLength = rootNode->get_number_child_nodes();
        eval.childNumberVisits = rootNode->get_child_number_visits();
        eval.qValues = rootNode->get_q_values();
        if (targetLength == 1) {
            eval.policyProbSmall = DynamicVector<float>(1);
            eval.policyProbSmall[0] = 1.0f;
        }
        else {
            ChildIdx bestMoveIdx;
            rootNode->get_mcts_policy(eval.policyProbSmall, bestMoveIdx, searchSettings->qValueWeight, searchSettings->qVetoDelta);
        }
       
        eval.legalMoves = rootNode->get_legal_actions();

        vector<size_t> indices;
        uint16_t maxIdx = min(searchSettings->multiPV, rootNode->get_no_visit_idx());

        if (maxIdx > 1) {
            sort_eval_lists(eval, indices);
        }
        
        auto p = sort_permutation(eval.legalMoves, std::greater<float>());
        for (size_t idx = 0; idx < eval.legalMoves.size(); ++idx) {
        indices.emplace_back(idx);
        }
        apply_permutation_in_place(eval.legalMoves, p);
        apply_permutation_in_place(indices, p);

        eval.init_vectors_for_multi_pv(searchSettings->multiPV);

        if (targetLength == 1 && rootNode->is_blank_root_node()) {
            // single move with no tree reuse
            eval.pv[0] = {rootNode->get_action(0)};
            // there are no q-values available, therefore use the state value evaluation as bestMoveQ
            eval.bestMoveQ[0] = rootNode->get_value();
            eval.centipawns[0] = value_to_centipawn(eval.bestMoveQ[0]);
        }
        else {
            for (size_t idx = 0; idx < maxIdx; ++idx) {
                set_eval_for_single_pv(eval, rootNode.get(), idx, indices, searchSettings);
            }
        }
        eval.selDepth = maxDepth;
        eval.nodes = rootNode->get_node_count();
        eval.tbHits = tbHits;

        evals.push_back(eval);
        tGCThread.join();
    }

    evalInfo->nodesPreSearch = init_root_node(state);
    evalInfo->legalMoves = rootNode->get_legal_actions();
    
    auto combinedPolicy = evals[0].policyProbSmall;
    auto combinedChildVisits = evals[0].childNumberVisits;
    auto combinedQValues = evals[0].qValues;

    for (size_t i = 1; i < numberOfAgents; i++)
    {
        for(auto j = 0; j < combinedPolicy.size(); ++j){
            combinedPolicy[j] += evals[i].policyProbSmall[j];
        }
        for(auto j = 0; j < combinedChildVisits.size(); ++j){
            combinedChildVisits[j] += evals[i].childNumberVisits[j];
        }
        for(auto j = 0; j < combinedQValues.size(); ++j){
            combinedQValues[j] += evals[i].qValues[j];
        }
    }

    for(auto j = 0; j < combinedPolicy.size(); ++j){
        combinedPolicy[j] += combinedPolicy[j]/numberOfAgents;
    }
    for(auto j = 0; j < combinedChildVisits.size(); ++j){
        combinedChildVisits[j] += combinedChildVisits[j]/numberOfAgents;
    }
    for(auto j = 0; j < combinedQValues.size(); ++j){
        combinedQValues[j] += combinedQValues[j]/numberOfAgents;
    }

    vector<float> diffs;
    for (size_t i = 0; i < numberOfAgents; i++)
    {   
        diffs.push_back(0.0);
        for(auto j = 0; j< combinedPolicy.size(); ++j){
            diffs[i] += std::sqrt(std::pow(evals[i].policyProbSmall[j] - combinedPolicy[j],2));
        }
    }
    std::vector<float>::iterator result = std::min_element(diffs.begin(), diffs.end()); 
    int stateIdx = std::distance(diffs.begin(), result);

    *evalInfo = evals[stateIdx];
    update_nps_measurement(evalInfo->calculate_nps());
    
    info_string("Selected State: " + std::to_string(stateIdx));
}
