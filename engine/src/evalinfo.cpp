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
 * @file: evalinfo.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "evalinfo.h"
#include "../util/blazeutil.h"

void print_single_pv(std::ostream& os, const EvalInfo& evalInfo, size_t idx, size_t elapsedTimeMS)
{
    if (idx != 0) {
        os << "info ";
    }

    os << "depth " << evalInfo.depth
       << " seldepth " << evalInfo.selDepth
       << " multipv " << idx+1
       << " score";

    if (evalInfo.movesToMate[idx] == 0) {
        os << " cp " << evalInfo.centipawns[idx];
    }
    else {
        os << " mate " << evalInfo.movesToMate[idx];
    }

    os << " nodes " << evalInfo.nodes
       << " nps " << evalInfo.calculate_nps(elapsedTimeMS)
       << " tbhits " << evalInfo.tbHits
       << " time " << elapsedTimeMS
       << " pv";
    for (Action move: evalInfo.pv[idx]) {
        os << " " << StateConstants::action_to_uci(move, evalInfo.isChess960);
    }
    os << endl;
}

std::ostream& operator<<(std::ostream& os, const EvalInfo& evalInfo)
{
    const size_t elapsedTimeMS = evalInfo.calculate_elapsed_time_ms();

    for (size_t idx = 0; idx < evalInfo.centipawns.size(); ++idx) {
        print_single_pv(os, evalInfo, idx, elapsedTimeMS);
    }
    return os;
}

size_t EvalInfo::calculate_elapsed_time_ms() const
{
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

size_t EvalInfo::calculate_nps(size_t elapsedTimeMS) const
{
    // avoid division by 0
    if (elapsedTimeMS == 0) {
        elapsedTimeMS = 1;
    }
    return int(((nodes-nodesPreSearch) / (elapsedTimeMS / 1000.0f)) + 0.5f);
}

size_t EvalInfo::calculate_nps() const
{
    return calculate_nps(calculate_elapsed_time_ms());
}

void EvalInfo::init_vectors_for_multi_pv(size_t multiPV)
{
    pv.resize(multiPV);
    movesToMate.resize(multiPV);
    bestMoveQ.resize(multiPV);
    centipawns.resize(multiPV);
}

// https://helloacm.com/how-to-implement-the-sgn-function-in-c/
template <class T>
inline int
sgn(T v) {
    return (v > T(0)) - (v < T(0));
}

int value_to_centipawn(float value)
{
    if (std::abs(value) >= 1) {
        // return a constant if the given value is 1 (otherwise log will result in infinity)
        return sgn(value) * 9999;
    }
    // use logarithmic scaling with basis 1.1 as a pseudo centipawn conversion
    return int(-(sgn(value) * std::log(1.0f - std::abs(value)) / std::log(VALUE_TO_CENTI_PARAM)) * 100.0f);
}

float get_best_move_q(const Node* nextNode, const SearchSettings* searchSettings)
{
    switch(searchSettings->searchPlayerMode) {
    case MODE_TWO_PLAYER:
        return -nextNode->get_value_display();
    case MODE_SINGLE_PLAYER:
    default:
        return nextNode->get_value_display();
    }
}

void set_eval_for_single_pv(EvalInfo& evalInfo, const Node* rootNode, size_t idx, vector<size_t>& indices, const SearchSettings* searchSettings)
{
    vector<Action> pv;
    size_t childIdx;
    if (idx == 0) {
        childIdx = get_best_action_index(rootNode, false, searchSettings);
    }
    else {
        childIdx = indices[idx];
    }
    pv.push_back(rootNode->get_action(childIdx));

    Node* nextNode = rootNode->get_child_node(childIdx);
    // make sure the nextNode has been expanded (e.g. when inference of the NN is too slow on the given hardware to evaluate the next node in time)
    if (nextNode != nullptr) {
        nextNode->get_principal_variation(pv, searchSettings);
        evalInfo.pv[idx] = pv;

        // scores
        // return mate score for known wins and losses
        if (nextNode->is_playout_node()) {
            evalInfo.bestMoveQ[idx] = get_best_move_q(nextNode, searchSettings);

            if (nextNode->get_node_type() == LOSS) {
                // always round up the ply counter
                evalInfo.movesToMate[idx] = (int(pv.size())+1) / 2;
                switch (searchSettings->searchPlayerMode) {
                case MODE_SINGLE_PLAYER:
                    evalInfo.movesToMate[idx] = -evalInfo.movesToMate[idx];
                case MODE_TWO_PLAYER: ;
                }
                return;
            }
            if (nextNode->get_node_type() == WIN) {
                // always round up the ply counter
                evalInfo.movesToMate[idx] = -(int(pv.size())+1) / 2;
                switch (searchSettings->searchPlayerMode) {
                case MODE_SINGLE_PLAYER:
                    evalInfo.movesToMate[idx] = -evalInfo.movesToMate[idx];
                case MODE_TWO_PLAYER: ;
                }
                return;
            }
        }
        else {
            switch (searchSettings->searchPlayerMode) {
            case MODE_TWO_PLAYER:
                evalInfo.bestMoveQ[idx] = -nextNode->get_value();
            break;
            case MODE_SINGLE_PLAYER:
                evalInfo.bestMoveQ[idx] = nextNode->get_value();
            }
        }
    }
    else {
        evalInfo.bestMoveQ[idx] = Q_INIT;
    }
    evalInfo.movesToMate[idx] = 0;
    evalInfo.centipawns[idx] = value_to_centipawn(evalInfo.bestMoveQ[idx]);
}

void sort_eval_lists(EvalInfo& evalInfo, vector<size_t>& indices)
{
    auto p = sort_permutation(evalInfo.policyProbSmall, std::greater<float>());
    for (size_t idx = 0; idx < evalInfo.legalMoves.size(); ++idx) {
        indices.emplace_back(idx);
    }
    apply_permutation_in_place(evalInfo.policyProbSmall, p);
    apply_permutation_in_place(evalInfo.legalMoves, p);
    apply_permutation_in_place(indices, p);
}

void update_eval_info(EvalInfo& evalInfo, const Node* rootNode, size_t tbHits, size_t selDepth, const SearchSettings* searchSettings)
{
    const size_t targetLength = rootNode->get_number_child_nodes();
    evalInfo.childNumberVisits = rootNode->get_child_number_visits();
    evalInfo.qValues = rootNode->get_q_values();
    if (targetLength == 1) {
        evalInfo.policyProbSmall = DynamicVector<float>(1);
        evalInfo.policyProbSmall[0] = 1.0f;
    }
    else {
        ChildIdx bestMoveIdx;
        rootNode->get_mcts_policy(evalInfo.policyProbSmall, bestMoveIdx, searchSettings);
    }
    // ensure the policy has the correct length even if some child nodes have not been visited
    if (evalInfo.policyProbSmall.size() != targetLength) {
        const size_t startIdx = evalInfo.policyProbSmall.size();
        fill_missing_values<double>(evalInfo.policyProbSmall, startIdx, targetLength, 0.0);
        fill_missing_values<double>(evalInfo.childNumberVisits, startIdx, targetLength, 0.0);
        fill_missing_values<float>(evalInfo.qValues, startIdx, targetLength, LOSS_VALUE);
    }
    evalInfo.legalMoves = rootNode->get_legal_actions();

    vector<size_t> indices;
    uint_fast16_t maxIdx = min(uint_fast16_t(searchSettings->multiPV), uint_fast16_t(rootNode->get_no_visit_idx()));

    if (maxIdx > 1) {
        sort_eval_lists(evalInfo, indices);
    }

    evalInfo.init_vectors_for_multi_pv(searchSettings->multiPV);

    if (targetLength == 1 && rootNode->is_blank_root_node()) {
        // single move with no tree reuse
        evalInfo.pv[0] = {rootNode->get_action(0)};
        // there are no q-values available, therefore use the state value evaluation as bestMoveQ
        evalInfo.bestMoveQ[0] = rootNode->get_value_display();
        evalInfo.centipawns[0] = value_to_centipawn(evalInfo.bestMoveQ[0]);
    }
    else {
        for (size_t idx = 0; idx < maxIdx; ++idx) {
            set_eval_for_single_pv(evalInfo, rootNode, idx, indices, searchSettings);
        }
    }

    // rawAgent has no pv line and only single best move
    if (evalInfo.pv.size() == 0) {
        evalInfo.depth = 1;
    }
    else {
        evalInfo.depth = evalInfo.pv[0].size();
    }
    evalInfo.selDepth = selDepth;
    evalInfo.nodes = rootNode->get_node_count();
    evalInfo.tbHits = tbHits;
}
