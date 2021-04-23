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

#ifdef MODE_LICHESS
    // Horde -> Q vs K
    pos.set("8/8/2k5/8/5Q2/8/8/8 w - - 0 1", false, HORDE_VARIANT, &newState, uiThread.get());
    REQUIRE(pos.draw_by_insufficient_material() == true);
#endif
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

#endif
