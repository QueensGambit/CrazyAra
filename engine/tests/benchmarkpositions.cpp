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
 * @file: benchmarkpositions.cpp
 * Created on 26.07.2019
 * @author: queensgambit
 */

#include "benchmarkpositions.h"

BenchmarkPositions::BenchmarkPositions():
    goCommand("movetime 3000"),
    positions({
        TestPosition("r1b2bk1/pp3ppp/2pn1bn1/4r3/3Q3P/2N1PB1p/PPP1PPP1/3RK2R/NQp w K - 0 24", "h4h5", "Q@h2"),
//        TestPosition("r1b2bk1/pp3ppp/2pn2n1/4r1P1/3Q2n1/2N1pB2/PPP1PPPP/R1B1K2R/Qp w KQ - 0 19", "f3g4", "c1e3"),
        TestPosition("r1bq1rk1/pppp1ppp/2n2n2/1Bb1p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1/ b - - 11 6", "c6d4", "f6g4"),
        TestPosition("r2qr1k1/ppp2ppp/2n1bp2/8/1b1P4/2N5/1PP1NPPP/R1BQKB1R/PPNp w KQ - 0 11", "N@e3", "P@h6"),
        TestPosition("r1bq1bk1/ppp2ppp/5p2/3pNn2/3PpB2/P1N5/1PP1QPPP/R4RK1/RNb b - - 0 13", "f6e5", "c8e6"),
        TestPosition("r2q1rk1/pp3ppp/2np2b1/6BB/3p4/3P2N1/PPrQBPKP/R7/PNPPPn w - - 0 29", "P@c7", "d2c2"),
        TestPosition("r2qr3/p1p3pk/2p3pp/3b1p1n/3P4/4PPB1/PPPBQ1PP/R4RK1/NBNpn b - - 0 30", "P@h4", "h5g3"),
        TestPosition("1r1qr3/p1p3pk/2p3pp/3b1p1n/3P3p/1P2PP2/P1PB2PP/R3BQK1/NNBNr b - - 0 34", "b8b3", "R@g5"),
        TestPosition("r2q4/1pp2kPp/5prP/2pP1N2/5PB1/2N2P2/PP3PPN/2r1rQ1K/Nbpbbp w - - 0 54", "N@h8", "f1e1"),
        TestPosition("r2q2kN/1pp3Pp/5prP/2pP1N2/5PB1/2N2P2/PP3PPN/4r2K/Rqbpbbp w - - 0 62", "R@g1", "R@f1"),
        TestPosition("r1bqk1r1/2p1bppp/p1p2n1P/3P4/2B5/2N2p2/PPP2PRP/R1BQK3/PNPn w Qq - 24 13", "d1f3", "g2g7"),
//        TestPosition("r1b2B2/pp2RP1p/3ppp2/k2Nq3/1NB5/P3NpP1/2PP1PpP/R3K1R1/qbnp w Q - 0 30", "a1b1", "b4c6"),
//        TestPosition("r3kr2/pbpp2PQ/1p2pPn1/4Pp2/1b1P3n/2NBBP1p/2P1N1PP/4RRK1/qp b - - 0 20", "h3g2", "Q@h1"),
        TestPosition("r4rk1/2pPbppp/p3p3/8/4P2n/2N4Q/PPP2PPP/R1B1K2R/BNPPqbn b KQ - 2 16", "B@g5", "Q@g6"),
        TestPosition("r3k2r/1pp2bpp/p3b1p1/3np1N1/6R1/bP5p/PnPP1P1P/R3Q1K1[QNbpp] w - - 0 26", "e1e5", "g5f7"),
        TestPosition("r1b1kb1r/pp3npp/3p1p2/Q1n5/3PP3/2P5/pBpP1PPP/R3KB1R[Qnnp] b Kkq - 0 21", "c5e4", "N@d3"),
        TestPosition("3q1rk1/p1p2p1B/2p4b/8/1PnP2Pb/4P2p/1PPR2PP/2R2R1K/PBQPnnpn b - - 0 41", "g8h8", "g8h7)"),
        // this position is lost but you should avoid mate #4
        TestPosition("r1b4r/ppp1kp1p/2bp4/6Pn/4n2N/8/P1P1BPPP/R4RK1/QBpnqppp w - - 0 21", "e2h5", "B@f6")
    }),
    totalNPS(0),
    totalDepth(0)
{
}
