/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: rawnetagent.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "rawnetagent.h"
#include "inputrepresentation.h"
#include "outputrepresentation.h"
#include "constants.h"
#include "misc.h"
//#include <numeric>      // std::accumulate
#include <blaze/Math.h>
#include "../blazeutil.h"
//#include "../../lib/sf/uci.h"
#include "../../lib/sf/uci.h"

using blaze::HybridVector;

RawNetAgent::RawNetAgent(NeuralNetAPI *net, PlaySettings playSettings, float temperature, unsigned int temperature_moves, bool verbose) : Agent (temperature, temperature_moves, verbose)
{

    this->net = net;
    this->playSettings = playSettings;
}

EvalInfo RawNetAgent::evalute_board_state(Board *pos)
{
    EvalInfo evalInfo;
    for (const ExtMove& move : MoveList<LEGAL>(*pos)) {
        evalInfo.legalMoves.push_back(move);
    }

    // sanity check
    assert(evalInfo.legalMoves.size() >= 1);

    // immediatly stop the search if there's only one legal move
    if (evalInfo.legalMoves.size() == 1) {
        evalInfo.policyProbSmall.resize(1UL);
        evalInfo.policyProbSmall = 1;
          // a value of 0 is likely a wron evaluation but won't be written to stdout
        evalInfo.centipawns = value_to_centipawn(0);
        evalInfo.depth = 0;
        evalInfo.nodes = 0;
        evalInfo.pv = {evalInfo.legalMoves[0]};
//        sync_cout << "bestmove " << UCI::move(evalInfo.legalMoves[0], pos->is_chess960()) << sync_endl;
        return evalInfo;
    }

//    float *input_planes_start = &input_planes[0][0][0];

    board_to_planes(pos, 0, true, begin(input_planes)); //input_planes_start);

//    float sum = 0;
//    float key = 0;
//    for (int i = 0; i < 34*8*8; ++i) {
////    std::cout << "input_planes" << *(input_planes_start+i) << std::endl;
//    sum += *(input_planes_start+i);
//    key += float(i)  * *(input_planes_start+i);
//    }
//    std::cout << "sum" << sum << std::endl;
//    std::cout << "key" << key << std::endl;

//    Eigen::VectorXf policyProb(NB_LABELS);
    float value;

//    net->predict_single(begin(input_planes), valueOutput, probOutputs);
    NDArray probOutputs = net->predict(begin(input_planes), value);

//    value = valueOutput.At(0, 0);
//    cout << "value: " << value << endl;
//    cout << "prob_vec: " << prob_vec << endl;

//    int index;
//    policyProb.maxCoeff(&index);

    /*
     * Find out the maximum accuracy and the index associated with that accuracy.
     * This is done by using the argmax operator on NDArray.
     */
    auto predicted = probOutputs.ArgmaxChannel();
    /*
     * Wait until all the previous write operations on the 'predicted'
     * NDArray to be complete before we read it.
     * This method guarantees that all previous write operations that pushed into the backend engine
     * for execution are actually finished.
     */
    predicted.WaitToRead();


    int best_idx = predicted.At(0); //, 0);

//    best_accuracy = array.At(0, best_idx);

//    std::cout << "array " << array << std::endl;
//    Constants::init();

    string bestmove_mxnet;
    if (pos->side_to_move() == WHITE) {
        bestmove_mxnet = LABELS[best_idx];
    }
    else {
        bestmove_mxnet = LABELS_MIRRORED[best_idx];
    }

    get_probs_of_move_list(0, probOutputs, evalInfo.legalMoves, pos->side_to_move(), true, evalInfo.policyProbSmall);
    size_t sel_idx = argmax(evalInfo.policyProbSmall);

//    sync_cout << "sel_idx " << sel_idx << sync_endl;
//    sync_cout << "policyProbSmall" << evalInfo.policyProbSmall/sum(evalInfo.policyProbSmall) << sync_endl;
    Move bestmove = evalInfo.legalMoves[sel_idx];
    assert(bestmove_mxnet == UCI::move(bestmove, pos->is_chess960()));

//    sync_cout << "bestmove " << UCI::move(bestmove, pos->is_chess960()) << sync_endl;

    evalInfo.centipawns = value_to_centipawn(value);
    evalInfo.depth = 1;
    evalInfo.nodes = 1;
    evalInfo.is_chess960 = pos->is_chess960();
    evalInfo.pv = {bestmove};
//    eval_info.legalMoves = this->rootNode->getLegalMoves();
//    eval_info.pVecSmall = this->rootNode->getPVecSmall();
    return evalInfo;

}
