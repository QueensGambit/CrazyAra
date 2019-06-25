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
 * @file: crazyara.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Main entry point for the executable which manages the UCI communication.
 */

#ifndef CRAZYARA_H
#define CRAZYARA_H

#include <iostream>

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"
#include "nn/neuralnetapi.h"
#include "agents/config/searchsettings.h"
#include "agents/config/searchlimits.h"
#include "agents/config/playsettings.h"
#include "node.h"

class CrazyAra
{
private:
    std::string intro = std::string("\n") +
            std::string("                                  _                                           \n") +
            std::string("                   _..           /   ._   _.  _        /\\   ._   _.           \n") +
            std::string("                 .' _ `\\         \\_  |   (_|  /_  \\/  /--\\  |   (_|           \n") +
            std::string("                /  /e)-,\\                         /                           \n") +
            std::string("               /  |  ,_ |                    __    __    __    __             \n") +
            std::string("              /   '-(-.)/          bw     8 /__////__////__////__////         \n") +
            std::string("            .'--.   \\  `                 7 ////__////__////__////__/          \n") +
            std::string("           /    `\\   |                  6 /__////__////__////__////           \n") +
            std::string("         /`       |  / /`\\.-.          5 ////__////__////__////__/            \n") +
            std::string("       .'        ;  /  \\_/__/         4 /__////__////__////__////             \n") +
            std::string("     .'`-'_     /_.'))).-` \\         3 ////__////__////__////__/              \n") +
            std::string("    / -'_.'---;`'-))).-'`\\_/        2 /__////__////__////__////               \n") +
            std::string("   (__.'/   /` .'`                 1 ////__////__////__////__/                \n") +
            std::string("    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n") +
            std::string("      _|.' /`                                                                 \n") +
            std::string("jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          \n") +
            std::string("    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n") +
            std::string("       \\_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.   \n") +
            std::string("              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n");
    RawNetAgent *rawAgent;
    MCTSAgent *mctsAgent;
    NeuralNetAPI *netSingle;
    NeuralNetAPI *netBatch;
//    unordered_map<Key, Node*> *hashTable;
public:
    CrazyAra();
    void welcome();
    void uci_loop(int argc, char* argv[]);
    void init();
    void go(Board& pos, istringstream& is, StateListPtr& states);
    void position(Board& pos, istringstream& is, StateListPtr& states);

};


int main(int argc, char* argv[]) {
    CrazyAra crazyara;
    crazyara.init();
    crazyara.welcome();
    crazyara.uci_loop(argc, argv);
}


#endif // CRAZYARA_H
