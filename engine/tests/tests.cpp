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
#ifndef MODE_XIANGQI
#include <string>
#include "catch.hpp"
#include "uci.h"
#include "uci/optionsuci.h"
#include "environments/chess_related/sfutil.h"
#include "uci/variants.h"
#include "thread.h"
#include "constants.h"
#include "environments/chess_related/inputrepresentation.h"
#include "legacyconstants.h"
#include "util/blazeutil.h"
using namespace Catch::literals;
using namespace std;
using namespace OptionsUCI;

void init() {
    OptionsUCI::init(Options);
    Bitboards::init();
    Position::init();
    Bitbases::init();
}

void get_planes_statistics(const Board* pos, bool normalize, double& sum, double& maxNum, double& key, size_t& argMax) {
    float inputPlanes[StateConstants::NB_VALUES_TOTAL()];
    board_to_planes(pos, pos->number_repetitions(), normalize, inputPlanes);
    sum = 0;
    maxNum = 0;
    key = 0;
    argMax = 0;
    for (int i = 0; i < StateConstants::NB_VALUES_TOTAL(); ++i) {
        const float val = inputPlanes[i];
        sum += val;
        if (inputPlanes[i] > maxNum) {
            maxNum = val;
            argMax = i;
        }
        key += i * val;
    }
}

void apply_moves_to_board(const vector<string>& uciMoves, Board& pos, StateListPtr& states) {
    for (string uciMove : uciMoves) {
        Move m = UCI::to_move(pos, uciMove);
        states->emplace_back();
        pos.do_move(m, states->back());
    }
}

bool are_all_entries_true(const vector<string>& uciMoves, bool (*foo)(Square, Square)) {
    for (auto uciMove : uciMoves) {
        if (!foo(get_origin_square(uciMove), get_destination_square(uciMove))) {
            cerr << "uciMove: " << uciMove << " returned false!" << endl;
            return false;
        }
    }
    return true;
}

TEST_CASE("En-passent moves") {
    vector<string> en_passent_moves = create_en_passent_moves();
    REQUIRE(are_all_entries_true(en_passent_moves, is_en_passent_candidate) == true);
}

TEST_CASE("Chess960 castling moves") {
    vector<string> castlingMoves = create_castling_moves(true);
    REQUIRE(are_all_entries_true(castlingMoves, is_960_castling_candidate_move) == true);
}

#ifdef MODE_LICHESS
TEST_CASE("Anti-Chess StartFEN"){
    init();

    Board pos;
    string token, cmd;
    auto uiThread = make_shared<Thread>(0);

    float *inputPlanes = new float[StateConstants::NB_VALUES_TOTAL()];

    StateInfo newState;
    pos.set(StartFENs[ANTI_VARIANT], false, ANTI_VARIANT, &newState, uiThread.get());
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
//    REQUIRE(StateConstants::NB_VALUES_TOTAL() == 3008); // no last move planes
    REQUIRE(StateConstants::NB_VALUES_TOTAL() == 4032); // with last move planes
    REQUIRE(int(max_num) == 1);
    REQUIRE(int(sum) == 224);
    REQUIRE(int(key) == 417296);
}
#endif

TEST_CASE("PGN_Move_Ambiguity"){
    init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);
    StateInfo newState;

    pos.set("r1bq1rk1/ppppbppp/2n2n2/4p3/4P3/1N1P1N2/PPP2PPP/R1BQKB1R w KQ - 5 6", false,
            CRAZYHOUSE_VARIANT, &newState, uiThread.get());
    string uci_move = "f3d2";
    Move move = UCI::to_move(pos, uci_move);

    vector<Action> legalMoves;
    // generate legal moves
    for (const ExtMove& move : MoveList<LEGAL>(pos)) {
        legalMoves.push_back(Action(move.move));
    }
    bool isRankAmbigious;
    bool isFileAmbigious;
    bool isAmbigious = is_pgn_move_ambiguous(move, pos, legalMoves, isFileAmbigious, isRankAmbigious);

    REQUIRE(isRankAmbigious == true);
    REQUIRE(isFileAmbigious == false);
    REQUIRE(isAmbigious == true);
}

TEST_CASE("Draw_by_insufficient_material"){
    init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);
    StateInfo newState;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // positive cases
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1) K v K
    pos.set("8/8/2k5/8/8/4K3/8/8 w - - 0 1", false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == true);
    // 2) KB vs K
    pos.set("8/8/2k5/8/5B2/4K3/8/8 w - - 0 1", false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == true);
    // 3) KN vs K
    pos.set("8/8/2k5/8/5N2/4K3/8/8 w - - 0 1", false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == true);
    // 4) KNN vs K
    pos.set("8/8/2k5/8/8/3NKN2/8/8 w - - 0 1", false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == true);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // negative cases
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pos.set("kn6/8/NK6/8/8/8/8/8 w - - 0 2", false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
    pos.set("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
            false, CHESS_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);

#ifdef MODE_LICHESS
    // 1) K v K
    pos.set("8/8/2k5/8/8/4K3/8/8 w - - 0 1", false, KOTH_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
    // 2) KB vs K
    pos.set("8/8/2k5/8/5B2/4K3/8/8 w - - 0 1", false, RACE_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
    // 3) KN vs K
    pos.set("8/8/2k5/8/5N2/4K3/8/8 w - - 0 1", false, ANTI_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
    // 4) KNN vs K
    pos.set("8/8/2k5/8/8/3NKN2/8/8 w - - 0 1", false, HORDE_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
    // 5) Horde -> P vs k
    pos.set("8/8/3k4/8/4P3/8/8/8 w - - 0 1", false, HORDE_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == false);
#endif
}

#ifdef MODE_CHESS
TEST_CASE("Chess_Input_Planes"){
    init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);

    // TEST BOARD POSITION I:
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(StartFENs[CHESS_VARIANT], false, CHESS_VARIANT, &states->back(), uiThread.get());

    // these move correspond lead to: "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    vector<string> uciMoves = {"e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"};
    apply_moves_to_board(uciMoves, pos, states);

    double sum, maxNum, key;
    size_t argMax;
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);

    REQUIRE(StateConstants::NB_VALUES_TOTAL() == 39*64);
    REQUIRE(maxNum == 4);
    REQUIRE(argMax == 1024);
    REQUIRE(sum == 557);
    REQUIRE(key == 617997);

    get_planes_statistics(&pos, true, sum, maxNum, key, argMax);
    REQUIRE(maxNum > 0.99);
    REQUIRE(maxNum < 1.01);
    REQUIRE(sum > 301.512);
    REQUIRE(sum < 301.513);
    REQUIRE(key > 348329.41);
    REQUIRE(key < 348329.42);

    // TEST BOARD POSITION II (more than 8 moves, white is checkmated):
    // 'r1bqkb1r/pp1ppppp/5n2/2p5/2P1P3/2Nn2P1/PP1PNP1P/R1BQKB1R w KQkq - 1 6'
    vector<string> uciMoves2 = {"e2e4","c7c5","c2c4","b8c6","g1e2","g8f6","b1c3","c6b4","g2g3","b4d3"};
    StateListPtr states2 = StateListPtr(new std::deque<StateInfo>(1));
    Board pos2;
    pos2.set(StartFENs[CHESS_VARIANT], false, CHESS_VARIANT, &states2->back(), uiThread.get());
    apply_moves_to_board(uciMoves2, pos2, states2);

    get_planes_statistics(&pos2, false, sum, maxNum, key, argMax);
    REQUIRE(maxNum == 6);
    REQUIRE(argMax == 1024);
    REQUIRE(sum == 816);
    REQUIRE(key == 909458);
}
#endif

TEST_CASE("LABELS length"){
    StateConstants::init(true);
    REQUIRE(OutputRepresentation::LABELS.size() == size_t(StateConstants::NB_LABELS()));
    REQUIRE(OutputRepresentation::LABELS_MIRRORED.size() == size_t(StateConstants::NB_LABELS()));
}

TEST_CASE("LABELS equality"){
    for (int idx = 0; idx < StateConstants::NB_LABELS(); ++idx) {
        REQUIRE(OutputRepresentation::LABELS[idx] == legacy_constants::LABELS[idx]);
    }
}

TEST_CASE("Board representation constants"){
    REQUIRE(StateConstants::BOARD_WIDTH() == legacy_constants::BOARD_WIDTH);
    REQUIRE(StateConstants::BOARD_HEIGHT() == legacy_constants::BOARD_HEIGHT);
    REQUIRE(StateConstants::NB_PLAYERS() == legacy_constants::NB_PLAYERS);
    REQUIRE(StateConstants::NB_CHANNELS_TOTAL() == legacy_constants::NB_CHANNELS_TOTAL);
    REQUIRE(StateConstants::NB_CHANNELS_CONST() == legacy_constants::NB_CHANNELS_CONST);
    REQUIRE(StateConstants::NB_CHANNELS_POLICY_MAP() == legacy_constants::NB_CHANNELS_POLICY_MAP);
    REQUIRE(StateConstants::NB_CHANNELS_HISTORY() == legacy_constants::NB_CHANNELS_HISTORY);
    REQUIRE(StateConstants::NB_VALUES_TOTAL() == legacy_constants::NB_VALUES_TOTAL);
    REQUIRE(StateConstants::NB_PIECE_TYPES() == legacy_constants::NB_PIECE_TYPES);
    REQUIRE(StateConstants::MAX_NB_NO_PROGRESS() == legacy_constants::MAX_NB_NO_PROGRESS);
    REQUIRE(StateConstants::MAX_NB_PRISONERS() == legacy_constants::MAX_NB_PRISONERS);
    REQUIRE(StateConstants::NB_CHANNELS_VARIANTS() == legacy_constants::NB_CHANNELS_VARIANTS);
    REQUIRE(StateConstants::MAX_FULL_MOVE_COUNTER() == legacy_constants::MAX_FULL_MOVE_COUNTER);
}


// ==========================================================================================================
// ||                                      Blaze-Util Tests                                                ||
// ==========================================================================================================

TEST_CASE("Blaze: first_and_second_max()"){
    DynamicVector<float> list = {3, 42, 1, 3, 99, 8, 7};
    float firstMax;
    float secondMax;
    size_t firstArg;
    size_t secondArg;
    first_and_second_max(list, list.size(), firstMax, secondMax, firstArg, secondArg);

    REQUIRE(firstMax == 99);
    REQUIRE(secondMax == 42);
    REQUIRE(firstArg == 4);
    REQUIRE(secondArg == 1);
}

// ==========================================================================================================
// ||                                   State Environment Tests                                            ||
// ==========================================================================================================

GameInfo apply_random_moves(StateObj& state, uint movesToApply) {
    GameInfo gameInfo;
    while (gameInfo.nbAppliedMoves < movesToApply) {
        REQUIRE(state.steps_from_null() == gameInfo.nbAppliedMoves);
        vector<Action> actions = state.legal_actions();
        const Action randomAction = actions[random() % actions.size()];
        gameInfo.givesCheck = state.gives_check(randomAction);
        state.do_action(actions[random() % actions.size()]);
        ++gameInfo.nbAppliedMoves;
        float dummy;
        if (state.is_terminal(actions.size(), gameInfo.givesCheck, dummy) != TERMINAL_NONE)  {
            gameInfo.reachedTerminal = true;
            return gameInfo;
        }
    }
    return gameInfo;
}

TEST_CASE("State: steps_from_null()"){
    srand(42);
    StateObj state;
    state.init(0, false);
    REQUIRE(state.steps_from_null() == 0);
    const uint movesToApply = 42;
    apply_random_moves(state, movesToApply);
    REQUIRE(state.steps_from_null() == movesToApply);
}

TEST_CASE("State: Reach terminal state"){
    srand(543);
    StateObj state;
    state.init(0, false);
    const uint movesToApply = 10000;
    GameInfo gameInfo = apply_random_moves(state, movesToApply);
    REQUIRE(gameInfo.reachedTerminal == true);
}

TEST_CASE("State: check_result()"){
    srand(1048);
    // check if we reach a terminal state when choosing random moves
    StateObj state;
    state.init(0, false);
    const uint movesToApply = 10000;
    GameInfo gameInfo = apply_random_moves(state, movesToApply);
    const Result result = state.check_result(gameInfo.givesCheck);
    REQUIRE(result != NO_RESULT);
    float dummy;
    const TerminalType terminalType = state.is_terminal(state.legal_actions().size(), gameInfo.givesCheck, dummy);
    switch(terminalType) {
    case TERMINAL_DRAW:
        REQUIRE(result == DRAWN);
        break;
    case TERMINAL_WIN:
        if (state.side_to_move() == FIRST_PLAYER_IDX) {
            REQUIRE(result == WHITE_WIN);
        } else {
            REQUIRE(result == BLACK_WIN);
        }
        break;
    case TERMINAL_LOSS:
        if (state.side_to_move() == FIRST_PLAYER_IDX) {
            REQUIRE(result == BLACK_WIN);
        } else {
            REQUIRE(result == WHITE_WIN);
        }
        break;
    case TERMINAL_NONE:
        REQUIRE(false);
        break;
    case TERMINAL_CUSTOM:
        // Custom behaviour
        break;
    }
}

TEST_CASE("State: clone()"){
    srand(543);
    StateObj state;
    state.init(0, false);
    const uint movesToApply = 7;
    apply_random_moves(state, movesToApply);
    unique_ptr<StateObj> state2 = unique_ptr<StateObj>(state.clone());
    REQUIRE(state2->fen() == state.fen());
}
#else
#include "catch.hpp"
#include "piece.h"
#include "thread.h"
#include "uci.h"
#include "uci/optionsuci.h"
#include "variant.h"
#include "environments/fairy_state/fairyboard.h"
#include "environments/fairy_state/fairystate.h"
#include "environments/fairy_state/fairyutil.h"
#include "environments/fairy_state/fairyinputrepresentation.h"


void init() {
    pieceMap.init();
    variants.init();
    OptionsUCI::init(Options);
    UCI::init(Options);
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();
    Tablebases::init("");
}

void get_planes_statistics(const FairyBoard* pos, bool normalize, double& sum, double& maxNum, double& key, size_t& argMax) {
    float inputPlanes[StateConstantsFairy::NB_VALUES_TOTAL()];
    board_to_planes(pos, normalize, inputPlanes);
    sum = 0;
    maxNum = 0;
    key = 0;
    argMax = 0;
    for (unsigned int i = 0; i < StateConstantsFairy::NB_VALUES_TOTAL(); ++i) {
        const float val = inputPlanes[i];
        sum += val;
        if (val > maxNum) {
            maxNum = val;
            argMax = i;
        }
        key += i * val;
    }
}

void apply_moves_to_board(const vector<string>& uciMoves, FairyBoard& pos, StateListPtr& states) {
    for (string uciMove : uciMoves) {
        Move m = UCI::to_move(pos, uciMove);
        states->emplace_back();
        pos.do_move(m, states->back());
    }
}

void apply_move_to_board(string uciMove, FairyBoard& pos, StateListPtr& states) {
    Move m = UCI::to_move(pos, uciMove);
    states->emplace_back();
    pos.do_move(m, states->back());
}

TEST_CASE("Xiangqi_Input_Planes") {
    init();
    FairyBoard pos;
    StateInfo newState;
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));

    auto uiThread = make_shared<Thread>(0);

    const Variant *xiangqiVariant = variants.find("xiangqi")->second;
    string startFen = xiangqiVariant->startFen;
    pos.set(xiangqiVariant, startFen, false, &states->back(), uiThread.get(), false);

    // starting position test
    double sum, maxNum, key;
    size_t argMax;
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 122);
    REQUIRE(maxNum == 1);
    REQUIRE(key == 236909);
    REQUIRE(argMax == 85);
    REQUIRE(pos.fen() == startFen);

    string uciMove = "c4c5";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 32);
    REQUIRE(maxNum == 1);
    REQUIRE(key == 22313);
    REQUIRE(argMax == 85);
    REQUIRE(pos.fen() == "rnbakabnr/9/1c5c1/p1p1p1p1p/9/2P6/P3P1P1P/1C5C1/9/RNBAKABNR b - - 1 1");

    uciMove = "g7g6";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 212);
    REQUIRE(maxNum == 1);
    REQUIRE(key == 459614);
    REQUIRE(argMax == 85);
    REQUIRE(pos.fen() == "rnbakabnr/9/1c5c1/p1p1p3p/6p2/2P6/P3P1P1P/1C5C1/9/RNBAKABNR w - - 2 2");

    uciMove = "h3g3";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 122);
    REQUIRE(maxNum == 1);
    REQUIRE(key == 245008);
    REQUIRE(argMax == 85);
    REQUIRE(pos.fen() == "rnbakabnr/9/1c5c1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C2/9/RNBAKABNR b - - 3 2");

    uciMove = "c10e8";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 302);
    REQUIRE(maxNum == 2);
    REQUIRE(key == 682338);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabnr/9/1c2b2c1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C2/9/RNBAKABNR w - - 4 3");

    uciMove = "h1i3";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 212);
    REQUIRE(maxNum == 2);
    REQUIRE(key == 467716);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabnr/9/1c2b2c1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C1N/9/RNBAKAB1R b - - 5 3");

    uciMove = "h10g8";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 392);
    REQUIRE(maxNum == 3);
    REQUIRE(key == 905043);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akab1r/9/1c2b1nc1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C1N/9/RNBAKAB1R w - - 6 4");

    uciMove = "i1h1";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 302);
    REQUIRE(maxNum == 3);
    REQUIRE(key == 690401);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akab1r/9/1c2b1nc1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C1N/9/RNBAKABR1 b - - 7 4");

    uciMove = "i10h10";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 482);
    REQUIRE(maxNum == 4);
    REQUIRE(key == 1127746);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabr1/9/1c2b1nc1/p1p1p3p/6p2/2P6/P3P1P1P/1C4C1N/9/RNBAKABR1 w - - 8 5");

    uciMove = "b3e3";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 392);
    REQUIRE(maxNum == 4);
    REQUIRE(key == 913108);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabr1/9/1c2b1nc1/p1p1p3p/6p2/2P6/P3P1P1P/4C1C1N/9/RNBAKABR1 b - - 9 5");

    uciMove = "h8h4";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 572);
    REQUIRE(maxNum == 5);
    REQUIRE(key == 1350490);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabr1/9/1c2b1n2/p1p1p3p/6p2/2P6/P3P1PcP/4C1C1N/9/RNBAKABR1 w - - 10 6");

    uciMove = "b1c3";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 482);
    REQUIRE(maxNum == 5);
    REQUIRE(key == 1135796);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "rn1akabr1/9/1c2b1n2/p1p1p3p/6p2/2P6/P3P1PcP/2N1C1C1N/9/R1BAKABR1 b - - 11 6");

    uciMove = "b10d9";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 662);
    REQUIRE(maxNum == 6);
    REQUIRE(key == 1573189);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "r2akabr1/3n5/1c2b1n2/p1p1p3p/6p2/2P6/P3P1PcP/2N1C1C1N/9/R1BAKABR1 w - - 12 7");

    uciMove = "a1a2";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 572);
    REQUIRE(maxNum == 6);
    REQUIRE(key == 1358503);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "r2akabr1/3n5/1c2b1n2/p1p1p3p/6p2/2P6/P3P1PcP/2N1C1C1N/R8/2BAKABR1 b - - 13 7");

    uciMove = "d10e9";
    apply_move_to_board(uciMove, pos, states);
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 752);
    REQUIRE(maxNum == 7);
    REQUIRE(key == 1795895);
    REQUIRE(argMax == 2430);
    REQUIRE(pos.fen() == "r3kabr1/3na4/1c2b1n2/p1p1p3p/6p2/2P6/P3P1PcP/2N1C1C1N/R8/2BAKABR1 w - - 14 8");
    REQUIRE(StateConstantsFairy::NB_VALUES_TOTAL() == 28*90);
}
#endif // MODE_XIANGQI
#endif
