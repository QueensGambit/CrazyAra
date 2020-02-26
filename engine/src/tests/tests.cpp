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
 * @file: tests.cpp
 * Created on 18.07.2019
 * @author: queensgambit
 */

#include "tests.h"

#ifdef BUILD_TESTS
#include <iostream>
#include <string>
#include "catch.hpp"
#include "uci.h"
#include "../util/sfutil.h"
#include "../domain/variants.h"
#include "thread.h"
#include "../domain/crazyhouse/constants.h"
#include "../domain/crazyhouse/inputrepresentation.h"
using namespace Catch::literals;
using namespace std;

void init() {
    Bitboards::init();
    Position::init();
    Bitbases::init();
}

TEST_CASE("En-passent moves") {
    vector<string> en_passent_moves;
    fill_en_passent_moves(en_passent_moves);

    for (auto uciMove: en_passent_moves) {
        bool returnVal = is_en_passent_candidate(get_origin_square(uciMove), get_destination_square(uciMove));
        if (!returnVal) {
            cerr << "uciMove: " << uciMove << " returned false!" << endl;
        }
        REQUIRE(returnVal == true);
    }
}

TEST_CASE("Anti-Chess StartFEN"){
    init();

    Board pos;
    string token, cmd;
    auto uiThread = make_shared<Thread>(0);

    float *inputPlanes = new float[NB_VALUES_TOTAL];

    StateInfo* newState = new StateInfo;
    pos.set(StartFENs[ANTI_VARIANT], false, ANTI_VARIANT, newState, uiThread.get());
    board_to_planes(&pos, pos.number_repetitions(), false, inputPlanes);
    size_t sum = 0;
    float max_num = 0;
    int key = 0;
    for (size_t i = 0; i < 3000; ++i) {
        const float val = inputPlanes[i];
        sum += val;
        if (inputPlanes[i] > max_num) {
            max_num = val;
        }
        key += i * val;
    }
    REQUIRE(NB_VALUES_TOTAL == 3008);
    REQUIRE(int(max_num) == 1);
    REQUIRE(int(sum) == 224);
    REQUIRE(int(key) == 417296);
}

TEST_CASE("PGN_Move_Ambiguity"){
    init();

    Board pos;
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    pos.set("r1bq1rk1/ppppbppp/2n2n2/4p3/4P3/1N1P1N2/PPP2PPP/R1BQKB1R w KQ - 5 6", false,
            CRAZYHOUSE_VARIANT, newState, uiThread.get());
    string uci_move = "f3d2";
    Move move = UCI::to_move(pos, uci_move);

    vector<Move> legalMoves;
    // generate legal moves
    for (const ExtMove& move : MoveList<LEGAL>(pos)) {
        legalMoves.push_back(move);
    }
    bool isRankAmbigious;
    bool isFileAmbigious;
    bool isAmbigious = is_pgn_move_ambiguous(move, pos, legalMoves, isFileAmbigious, isRankAmbigious);

    REQUIRE(isRankAmbigious == true);
    REQUIRE(isFileAmbigious == false);
    REQUIRE(isAmbigious == true);
}
#endif
