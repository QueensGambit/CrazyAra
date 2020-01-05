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
 * @file: benchmarkpositions.h
 * Created on 26.07.2019
 * @author: queensgambit
 *
 * Run of the engine using a set of FENs and test the result for blunder moves
 */

#ifndef BENCHMARKPOSITIONS_H
#define BENCHMARKPOSITIONS_H

#include "tests.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;

struct TestPosition {
public:
    string fen;
    string blunderMove;
    string alternativeMove;
    TestPosition(string fen, string blunderMove, string alternativeMove):
        fen(fen), blunderMove(blunderMove), alternativeMove(alternativeMove) {}
};

struct BenchmarkPositions
{
public:
    string goCommand;
    vector<TestPosition> positions;
    float totalNPS;
    float totalDepth;
    BenchmarkPositions();
};

#ifdef BENCHMARK
int main() {
    BenchmarkPositions benchmark;
    benchmark.run_benchmark();
}
#endif

#endif // BENCHMARKPOSITIONS_H
