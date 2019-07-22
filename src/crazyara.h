/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
#include "statesmanager.h"
#include "tests/tests.h"

class CrazyAra
{
private:
    string name = "CrazyAra";
    string version = "0.6.0";
    string authors = "Johannes Czech, Moritz Willig, Alena Beyer et al.";
    string intro =  string("\n") +
                    string("                                  _                                           \n") +
                    string("                   _..           /   ._   _.  _        /\\   ._   _.           \n") +
                    string("                 .' _ `\\         \\_  |   (_|  /_  \\/  /--\\  |   (_|           \n") +
                    string("                /  /e)-,\\                         /                           \n") +
                    string("               /  |  ,_ |                    __    __    __    __             \n") +
                    string("              /   '-(-.)/          bw     8 /__////__////__////__////         \n") +
                    string("            .'--.   \\  `                 7 ////__////__////__////__/          \n") +
                    string("           /    `\\   |                  6 /__////__////__////__////           \n") +
                    string("         /`       |  / /`\\.-.          5 ////__////__////__////__/            \n") +
                    string("       .'        ;  /  \\_/__/         4 /__////__////__////__////             \n") +
                    string("     .'`-'_     /_.'))).-` \\         3 ////__////__////__////__/              \n") +
                    string("    / -'_.'---;`'-))).-'`\\_/        2 /__////__////__////__////               \n") +
                    string("   (__.'/   /` .'`                 1 ////__////__////__////__/                \n") +
                    string("    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n") +
                    string("      _|.' /`                                                                 \n") +
                    string("jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          \n") +
                    string("    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n") +
                    string("       \\_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.   \n") +
                    string("              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n");
    RawNetAgent *rawAgent;
    MCTSAgent *mctsAgent;
    NeuralNetAPI *netSingle;
    NeuralNetAPI *netBatch;
    bool networkLoaded = false;
    StatesManager *states;

    /**
     * @brief engine_info Returns a string about the engine version and authors
     * @return string
     */
    string engine_info();

public:
    CrazyAra();
    void welcome();
    void uci_loop(int argc, char* argv[]);
    void init();

    /**
     * @brief is_ready Loads the neural network weights and creates the agent object in case there haven't loaded already
     * @return True, if everything isReady
     */
    bool is_ready();

    void go(Board *pos, istringstream& is);
    void position(Board *pos, istringstream &is);
};

#ifndef BUILD_TESTS
int main(int argc, char* argv[]) {
    CrazyAra crazyara;
    crazyara.init();
    crazyara.welcome();
    crazyara.uci_loop(argc, argv);
}
#endif

// Challenging FENs
/*
NR1n1k1r/ppPbbppp/3p1b1n/8/4B3/2P5/P1P1pPPP/2n2R1K/QPQPr w - - 0 29
-> Q@e1
r1b1Rq1k/ppp2pqp/5Nn1/1B2p1B1/3P4/8/PPP2bpP/2KR2R1/PNPppn b - - 0 20
-> @h6 (Sf needs very high depth to find this move > ) (h7h6 is also viable) -> SF needs depth>= 19 to find this
r1b1kb1r/ppp1pppp/2n5/1B1qN3/3P2p1/2P5/P1P2PpP/R1BQK1R1/NPn w Qkq - 0 10
-> N@a6
3k2r1/pBpr1p1p/Pp3p1B/3p4/2PPn2B/5NPp/q4PpP/1R1QR1K1/NNbp w - - 1 23
-> N@e6
r1bq2nr/ppppbkpp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R[Pb] w KQ - 0 5
-> @d4
r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1[-] b kq - 0 4
-> d6?!

1r1k2r1/pP1bbpPp/5p2/3N4/4p3/8/PPnPQPPP/R1B2K1R/PBNqnpp b - - 1 21
-> N@c7
8/3pbkbP/N2pp1p1/1p2pprp/1P1P4/1Q6/P1P2P1P/2N1NK1R/BQprrbn b - - 0 57
-> P@e2 M#7
8/3pbkbP/N2pp1p1/1p2pprp/1PQP4/8/P1P2P1P/2N1NK1R/BBQprrn b - - 0 58
-> R@g1 M#11
r4rk1/ppp2pbp/8/4p1q1/3nB3/2NP1BPp/PPP2P1P/R4RK1[QBNPnp] b - - 0 21
-> N@d2 ?!
3r3k/pP3rp1/1p2pnKp/1Q6/3bB3/P2P2P1/1P4Np/R6R/PPPNBQPNPb b - - 0 51
-> Bc5 M#10
r1b2r2/pp3Bpk/3b1q2/2p5/3pPp2/3P3P/PPP2PP1/RN1Q1RK1/PNNpbn b - - 0 19
-> @h6 oder N@g5

r2q2k1/ppp2ppp/5p2/1B1p4/8/2r3P1/PPn1Q1KP/R1B5/PNNBBRpppn w - - 0 20
-> @e6

r4bk1/ppq3pp/5pp1/3N4/3Qp1b1/P7/1PP2PKP/R4R2/PPNNRNbbpp b - - 0 21
r4bk1/ppq3pp/5pp1/3NQ3/4p1b1/P7/1PP2PKP/R4R2/BPPNNRNbpp b - - 0 22
-> probably first mate sequence for white but Sf doesn't see it at depth 29
info depth 29 seldepth 18 multipv 1 score cp 0 nodes 638207891 nps 954325 hashfull 999 tbhits 0 time 668753 pv P@f3 g2g1 c7e5 P@f7 g8f7 N@h8 f7e8 N@c7 e8d7 P@c6 b7c6 R@f7 f8e7 N@c5 d7d6 c5b7 d6d7

*/



#endif // CRAZYARA_H
