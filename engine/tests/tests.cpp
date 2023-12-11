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
#include "catch_amalgamated.hpp"
using namespace Catch::literals;
using namespace std;
#include <string>
#ifndef MODE_STRATEGO
#if !defined(MODE_XIANGQI) && !defined(MODE_BOARDGAMES)
#ifdef SF_DEPENDENCY
#include "uci.h"
#endif
#include "uci/optionsuci.h"
#include "environments/chess_related/sfutil.h"
#include "thread.h"
#include "constants.h"
#include "environments/chess_related/inputrepresentation.h"
#include "legacyconstants.h"
#include "util/blazeutil.h"
#include "environments/chess_related/boardstate.h"
using namespace OptionsUCI;

#ifdef SF_DEPENDENCY
void init() {
    OptionsUCI::init(Options);
    Bitboards::init();
    Position::init();
    Bitbases::init();
}
#endif

struct PlaneStatistics {
    double sum;
    double maxNum;
    double key;
    size_t argMax;
    PlaneStatistics() :
        sum(0), maxNum(0), key(0), argMax(0) {}
};

PlaneStatistics get_stats_from_input_planes(const float* inputPlanes, uint nbValuesTotal)
{
    PlaneStatistics stats;
    for (uint i = 0; i < nbValuesTotal; ++i) {
        const float val = inputPlanes[i];
        stats.sum += val;
        if (inputPlanes[i] > stats.maxNum) {
            stats.maxNum = val;
            stats.argMax = i;
        }
        stats.key += i * val;
    }
    return stats;
}

PlaneStatistics get_planes_statistics(const StateObj& state, bool normalize, Version version=StateConstants::CURRENT_VERSION(), uint nbValuesTotal=StateConstants::NB_VALUES_TOTAL()) {
    vector<float> inputPlanes(nbValuesTotal);
    state.get_state_planes(normalize, inputPlanes.data(), version);
    return get_stats_from_input_planes(inputPlanes.data(), nbValuesTotal);
}

PlaneStatistics get_planes_statistics(const Board& pos, bool normalize, Version version=StateConstants::CURRENT_VERSION(), uint nbValuesTotal=StateConstants::NB_VALUES_TOTAL()) {
    vector<float> inputPlanes(nbValuesTotal);
    board_to_planes(&pos, pos.number_repetitions(), normalize, inputPlanes.data(), version);
    return get_stats_from_input_planes(inputPlanes.data(), nbValuesTotal);
}

void get_planes_statistics(const Board* pos, bool normalize, double& sum, double& maxNum, double& key, size_t& argMax) {
    PlaneStatistics stats = get_planes_statistics(*pos, normalize);
    sum = stats.sum;
    maxNum = stats.maxNum;
    key = stats.key;
    argMax = stats.argMax;
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


bool is_uci_move_legal(const BoardState& pos, const string& move, bool is960) {
    StateConstantsBoard scb;
    for (Action action : pos.legal_actions()) {
        if (scb.action_to_uci(action, is960) == move) {
            return true;
        }
    }
    return false;
}

bool legal_actions_equal_ucimoves(const BoardState& pos, const vector<string>& uciMoves, bool is960) {
    vector<string> compareMoves;
    StateConstantsBoard scb;
    for (Action action : pos.legal_actions()) {
        compareMoves.push_back(scb.action_to_uci(action, is960));
    }
    return std::is_permutation(uciMoves.begin(), uciMoves.end(), compareMoves.begin());
}

bool are_uci_moves_legal_bool(const BoardState& pos, const vector<string>& uciMoves, bool equals, bool is960) {
    for (string move : uciMoves) {
        if (is_uci_move_legal(pos, move, is960) != equals) {
            return false;
        }
    }
    return true;
}

Variant get_default_variant()
{
#ifndef MODE_CRAZYHOUSE
    return CHESS_VARIANT;
#else
    return CRAZYHOUSE_VARIANT;
#endif
}

TEST_CASE("En-passent moves") {
    vector<string> en_passent_moves = create_en_passent_moves();
    REQUIRE(are_all_entries_true(en_passent_moves, is_en_passent_candidate) == true);
}

#ifdef MODE_LICHESS
TEST_CASE("Anti-Chess StartFEN"){
    init();
    StateObj state;
    state.set(StateConstants::start_fen(ANTI_VARIANT), false, ANTI_VARIANT);
    PlaneStatistics stats = get_planes_statistics(state, false);

    // REQUIRE(StateConstants::NB_VALUES_TOTAL() == 3008); // no last move planes
    REQUIRE(StateConstants::NB_VALUES_TOTAL() == 4032); // with last move planes
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.sum == 224);
    REQUIRE(stats.key == 417296);
}
#endif

TEST_CASE("PGN_Move_Ambiguity"){
    init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);
    StateInfo newState;

    pos.set("r1bq1rk1/ppppbppp/2n2n2/4p3/4P3/1N1P1N2/PPP2PPP/R1BQKB1R w KQ - 5 6", false,
            get_default_variant(), &newState, uiThread.get());
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

#ifdef MODE_LICHESS
TEST_CASE("Racing Kings No Mirror Test, Input Representation Version 2"){

    // initial position
    BoardState state;
    state.init(RACE_VARIANT, false);
    PlaneStatistics stats;
    stats = get_planes_statistics(state, false);
    REQUIRE(stats.sum == 208);
    REQUIRE(stats.argMax == 68);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 425624);

    // black to move
    state.set("8/8/8/8/8/6K1/krbnNBR1/qrbnNBRQ b - - 1 1", false, RACE_VARIANT);
    stats = get_planes_statistics(state, false);
    REQUIRE(stats.sum == 208);
    REQUIRE(stats.argMax == 67);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 450207);

    // last moves
    state.init(RACE_VARIANT, false);
    apply_given_moves(state, {"h2g3"});
    stats = get_planes_statistics(state, false);
    REQUIRE(stats.sum == 210);
    REQUIRE(stats.argMax == 67);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 456324);
}
#endif

#ifdef MODE_CHESS
#if VERSION == 1
TEST_CASE("Chess_Input_Planes Version 1"){
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
#else  // Version == 2 || Version == 3.0
TEST_CASE("Chess_Input_Planes Version 2.7, 2.8, 3.0"){
    init();
    BoardState state;
    PlaneStatistics stats;

    // Start Pos: normalize=false
    state.init(get_default_variant(), false);
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 2592);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 20);
    REQUIRE(stats.key == 5129584);
#else  // SUB_VERSION == 7
    REQUIRE(stats.sum == 1632);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 20);
    REQUIRE(stats.key == 3006288);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 1312);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3430384);
#endif

    // Start Pos: normalize=true
    state.init(get_default_variant(), false);
    stats = get_planes_statistics(state, true);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 492);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 651530);
#else
    REQUIRE(stats.sum == 372);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 386118);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 472);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 819860);
#endif

    // Checking test: normalize=false
    state.set("rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3", false, get_default_variant());
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 1697);
    REQUIRE(stats.argMax == 2112);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3268193);
#else
    REQUIRE(stats.sum == 737);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 6);
    REQUIRE(stats.key == 1144897);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 1377);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3513153);
#endif

    // last moves
    state.init(get_default_variant(), false);
    apply_given_moves(state, {"e2e4", "c7c5"});
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 3234);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 30);
    REQUIRE(stats.key == 6462788.0);
#else
    REQUIRE(stats.sum == 2274);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 30);
    REQUIRE(stats.key == 4339492.0);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 1316);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3436012);
#endif

    // Checking move test: normalize=true
    state.set("r1br2k1/p4ppp/2p2n2/Q1b1p3/8/NP3N1P/P1P1BPP1/R1B1K2R b KQ - 0 12", false, get_default_variant());
    stats = get_planes_statistics(state, true);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 329);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 481472);
#else
    REQUIRE(stats.sum == 241);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 287212);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 284);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 529254);
#endif

    // en-passant test: normalize=false
    state.init(get_default_variant(), false);
    apply_given_moves(state, {"e2e4", "c7c5", "d2d3", "a7a6", "e4e5", "d7d5"});
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 3491);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 34);
    REQUIRE(stats.key == 6995937.0);
#else
    REQUIRE(stats.sum == 2531);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 34);
    REQUIRE(stats.key == 4872641.0);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 1325);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3451283);
#endif

    // en-passant test + check moves test: normalize=true
    state.init(get_default_variant(), false);
    string uciMove;
    apply_given_moves(state, {"e2e4", "c7c5", "e4e5", "d7d5"});
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 3301);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 31);
    REQUIRE(stats.key == 6600615.0);
#else
    REQUIRE(stats.sum == 2341);
    REQUIRE(stats.argMax == 2048);
    REQUIRE(stats.maxNum == 31);
    REQUIRE(stats.key == 4477319.0);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 1321);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3443613);
#endif

    // material difference
    state.set("r3k1nr/pbp4p/p2p2pb/4P3/3P4/N2q1n2/PPP2PPP/5K1R w kq - 0 14", false, get_default_variant());
    stats = get_planes_statistics(state, false);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 723);
    REQUIRE(stats.argMax == 2112);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 1404265.0);
#else
    REQUIRE(stats.sum == 83);
    REQUIRE(stats.argMax == 1472);
    REQUIRE(stats.maxNum == 2);
    REQUIRE(stats.key == 16041.0);
#endif
#endif
#if VERSION == 3
    REQUIRE(stats.sum == 659);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 1715209);
#endif

    // castle-rights & no-progress counter
    state.set("2kr3r/pbqp1ppp/2n2n2/4b3/4P3/2NPB3/PPP1QPPP/R4RK1 b - - 4 11", false, get_default_variant());
    stats = get_planes_statistics(state, true);
#if VERSION == 2
#if SUB_VERSION == 8
    REQUIRE(stats.sum == 214);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 377604.0);
#else
    REQUIRE(stats.sum == 118);
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 163636.0);
#endif
#endif
#if VERSION == 3
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(179.12, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(442487.2, 0.001));
#endif
}
#endif

TEST_CASE("6-Men WDL"){
    init();
    if (string(Options["SyzygyPath"]).empty() || string(Options["SyzygyPath"]) == "<empty>") {
        cout << "warning: No tablebases found -> skipped test for 6-Men WDL" << endl;
    }
    else {
        // Blunder by ClassicAra in https://tcec-chess.com/#div=q43t&game=293&season=21
        Tablebases::init(UCI::variant_from_name(Options["UCI_Variant"]), Options["SyzygyPath"]);
        StateObj state;
        state.set("8/1K2k3/8/4P3/R3r3/P7/8/8 b - - 0 55", false, get_default_variant());
        Tablebase::ProbeState probeState;
        Tablebase::WDLScore wdl = state.check_for_tablebase_wdl(probeState);
        REQUIRE(probeState != Tablebase::ProbeState::FAIL);
        REQUIRE(wdl == Tablebase::WDLScore::WDLWin);
    }
}
#endif

TEST_CASE("LABELS length"){
    StateConstants::init(true, false);
    REQUIRE(OutputRepresentation::LABELS.size() == size_t(StateConstants::NB_LABELS()));
    REQUIRE(OutputRepresentation::LABELS_MIRRORED.size() == size_t(StateConstants::NB_LABELS()));
}

TEST_CASE("LABELS equality"){
    for (int idx = 0; idx < StateConstants::NB_LABELS(); ++idx) {
        REQUIRE(OutputRepresentation::LABELS[idx] == legacy_constants::LABELS[idx]);
    }
}

#if VERSION == 1
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
#endif

TEST_CASE("3-fold Repetition"){
    init();
    // Blunder by ClassicAra in https://tcec-chess.com/#div=l4&game=100&season=21
    StateObj state;
    state.set("1rr3k1/1pp2ppp/p1n5/P2p1b2/3Pn3/R3PNP1/1P3PBP/2R1B1K1 b - - 4 17", false, get_default_variant());
    string moveB0 = "e4d6";
    string moveW1 = "f3h4";
    string moveB1 = "f5e6";
    string moveW2 = "h4f3";
    string moveB2 = "e6f5";
    vector<string> moves = {moveB0, moveW1, moveB1, moveW2, moveB2, moveW1, moveB1, moveW2};
    float customTerminalValue;
    TerminalType terminal;
    for (string move : moves) {
        state.do_action(state.uci_to_action(move));
        terminal = state.is_terminal(state.legal_actions().size(), customTerminalValue);
        REQUIRE(terminal == TERMINAL_NONE);
    }
    state.do_action(state.uci_to_action(moveB2));
    terminal = state.is_terminal(state.legal_actions().size(), customTerminalValue);
    REQUIRE(terminal == TERMINAL_DRAW);
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

    DynamicVector<float> list2 = {99, 3, 1, 3, 42, 8, 7};
    first_and_second_max(list2, list2.size(), firstMax, secondMax, firstArg, secondArg);

    REQUIRE(firstMax == 99);
    REQUIRE(secondMax == 42);
    REQUIRE(firstArg == 0);
    REQUIRE(secondArg == 4);
}

// ==========================================================================================================
// ||                                   State Environment Tests                                            ||
// ==========================================================================================================

GameInfo apply_random_moves(StateObj& state, uint movesToApply) {
    GameInfo gameInfo;
    while (gameInfo.nbAppliedMoves < movesToApply) {
        REQUIRE(state.steps_from_null() == gameInfo.nbAppliedMoves);
        vector<Action> actions = state.legal_actions();
        float dummy;
        if (state.is_terminal(actions.size(), dummy) != TERMINAL_NONE)  {
            gameInfo.reachedTerminal = true;
            return gameInfo;
        }
        const Action randomAction = actions[random() % actions.size()];
        state.do_action(randomAction);
        ++gameInfo.nbAppliedMoves;
    }
    return gameInfo;
}

void apply_given_moves(StateObj& state, const std::vector<string>& uciMoves) {
    for (string uciMove: uciMoves) {
        state.do_action(state.uci_to_action(uciMove));
    }
}

TEST_CASE("State: steps_from_null()"){
    srand(42);
    StateObj state;
    state.init(get_default_variant(), false);
    REQUIRE(state.steps_from_null() == 0);
    const uint movesToApply = 42;
    apply_random_moves(state, movesToApply);
    REQUIRE(state.steps_from_null() == movesToApply);
}

TEST_CASE("State: Reach terminal state"){
    srand(543);
    StateObj state;
    state.init(get_default_variant(), false);
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
    apply_random_moves(state, movesToApply);
    const Result result = state.check_result();
    REQUIRE(result != NO_RESULT);
    float dummy;
    const TerminalType terminalType = state.is_terminal(state.legal_actions().size(), dummy);
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
#elif defined(MODE_XIANGQI) || defined(MODE_BOARDGAMES)
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
#endif // MODE_XIANGQI || MODE_BOARDGAMES

#ifdef MODE_BOARDGAMES
TEST_CASE("Board_Games_Input_Planes") {
    init();
    FairyBoard pos;
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));

    auto uiThread = make_shared<Thread>(0);

    const Variant *breakthroughVariant = variants.find("breakthrough")->second;
    string startFen = breakthroughVariant->startFen;
    pos.set(breakthroughVariant, startFen, false, &states->back(), uiThread.get(), false);

    // starting position test
    double sum, maxNum, key;
    size_t argMax;
    get_planes_statistics(&pos, false, sum, maxNum, key, argMax);
    REQUIRE(sum == 96);
    REQUIRE(maxNum == 1);
    REQUIRE(key == 12240);
    REQUIRE(argMax == 0);
    REQUIRE(pos.fen() == string("pppppppp/pppppppp/8/8/8/8/PPPPPPPP/PPPPPPPP w - - 0 1"));
}
#endif

#ifdef MODE_XIANGQI
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

#ifdef MODE_LICHESS
TEST_CASE("King of the hill"){
    init();
    BoardState pos;

    // Check for win in the center
    // black wins
    pos.set("8/7p/8/1b1pk3/4p3/5n2/1K3P2/8 w - - 4 43", false, KOTH_VARIANT); // top right
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("8/8/2p2p2/3k3p/4p2P/r6P/8/5K2 w - - 0 47", false, KOTH_VARIANT); // top left
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("rnbq1bnr/pppp1ppp/8/8/P2kp3/1RN4N/1PPPPPPP/2BQKB1R w K - 0 1", false, KOTH_VARIANT); // bottom left
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("rnbq1bnr/ppp1pppp/3p4/8/P3k3/1RN4N/1PPPPPPP/2BQKB1R w K - 0 1", false, KOTH_VARIANT); // bottom right
    REQUIRE(pos.check_result() == BLACK_WIN);
    // white wins
    pos.set("5k2/1p5p/8/p7/2P1Kp2/5N1P/Pr3PP1/3RR3 b - - 1 29", false, KOTH_VARIANT); // bottom right
    REQUIRE(pos.check_result() == WHITE_WIN);
    pos.set("6k1/5p2/bP3B2/3N4/3K4/5P1p/P7/8 b - - 2 37", false, KOTH_VARIANT); // bottom left
    REQUIRE(pos.check_result() == WHITE_WIN);
    pos.set("rnbqkb1r/1pppppp1/p7/3K1n1p/8/5N2/PPPP1PPP/RNBQ1B1R w kq - 0 1", false, KOTH_VARIANT); // top left
    REQUIRE(pos.check_result() == WHITE_WIN);
    pos.set("rnb1kb1r/1ppqppp1/p6n/3PK2p/8/8/PPPP1PPP/RNBQ1BNR w kq - 0 1", false, KOTH_VARIANT); // top right
    REQUIRE(pos.check_result() == WHITE_WIN);
}

TEST_CASE("Variants_Horde"){
    init();
    BoardState pos;

    // The Pieces win by capturing all the pawns
    pos.set("8/8/1p4k1/8/4q3/8/8/8 w - - 0 76", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("6r1/8/4k3/5q2/p7/8/8/8 w - - 0 65", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("8/4k3/8/4q3/8/8/8/8 w - - 0 63", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);

    // Pawns win by checkmating the king
    pos.set("rnbqkbnr/1ppp1P1p/3PP3/2P5/PP5P/P1PPPPPP/PPpPPPPP/PPPPPPPP b kq - 0 10", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // ... even with multiple promoted pieces
    pos.set("8/8/R7/6P1/8/PP1P4/k1P5/Q3QPP1 b - - 3 69", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // Stalmate due to missing of legal moves
    pos.set("6k1/6P1/7q/8/8/8/8/8 w - - 0 1", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == DRAW);
    pos.set("1k6/3R4/2Q5/8/2P5/3P4/8/8 b - - 0 1", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == DRAW);

    // Stalmate due to 50 move rule
    pos.set("6k1/3R4/8/8/8/8/8/8 b - - 99 85", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == NO_RESULT);
    pos.set("6k1/3R4/8/8/8/8/8/8 b - - 100 85", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == DRAW);

    // Pawns on the first rank can move 2 squares (and 1 square)

    pos.set("3k4/8/8/8/8/8/8/PPPPPPPP w - - 0 1", false, HORDE_VARIANT);
    vector<Action> legalActions = pos.legal_actions();
    vector<string> legalActionsSan;
    for (Action action : legalActions) {
        legalActionsSan.push_back(pos.action_to_san(action, legalActions));
    }
    vector<string> match = {"a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"};
    std::sort(match.begin(), match.end());
    std::sort(legalActionsSan.begin(), legalActionsSan.end());
    REQUIRE(legalActionsSan == match);

    // Pawns of the first rank that moved 2 squares cannot be captured en passant

    pos.set("6k1/8/8/8/8/1p1p4/8/2P5 w - - 0 1", false, HORDE_VARIANT);
    string mov = "c1c3";
    Action action = pos.uci_to_action(mov);
    pos.do_action(action);
    legalActions = pos.legal_actions();
    legalActionsSan.clear();
    for (Action action : legalActions) {
        legalActionsSan.push_back(pos.action_to_san(action, legalActions));
    }
    bool moveInLegalActions = std::find(legalActionsSan.begin(), legalActionsSan.end(), "bxc2") != legalActionsSan.end();
    REQUIRE(moveInLegalActions == false);
    moveInLegalActions = std::find(legalActionsSan.begin(), legalActionsSan.end(), "dxc2") != legalActionsSan.end();
    REQUIRE(moveInLegalActions == false);

    // However "normal" en passant from second row is possible

    pos.set("1kb3nr/8/8/8/3p1pP1/8/1P2P3/P6P w - - 0 1", false, HORDE_VARIANT);
    mov = "e2e4";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    legalActions = pos.legal_actions();
    legalActionsSan.clear();
    for (Action action : legalActions) {
        legalActionsSan.push_back(pos.action_to_san(action, legalActions));
    }
    moveInLegalActions = std::find(legalActionsSan.begin(), legalActionsSan.end(), "dxe3") != legalActionsSan.end();
    REQUIRE(moveInLegalActions == true);
    moveInLegalActions = std::find(legalActionsSan.begin(), legalActionsSan.end(), "fxe3") != legalActionsSan.end();
    REQUIRE(moveInLegalActions == true);

    // A pawn that moved from first row to second row can now move 2 squares

    pos.set("1kb3nr/8/8/8/3p1pP1/8/4P3/PP5P w - - 0 1", false, HORDE_VARIANT);
    mov = "b1b2";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "h8h1";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    legalActions = pos.legal_actions();
    legalActionsSan.clear();
    for (Action action : legalActions) {
        legalActionsSan.push_back(pos.action_to_san(action, legalActions));
    }
    moveInLegalActions = std::find(legalActionsSan.begin(), legalActionsSan.end(), "b4") != legalActionsSan.end();
    REQUIRE(moveInLegalActions == true);

    // No draw by insufficient material
    pos.set("4k3/8/8/6P1/8/8/8/8 w - - 0 1", false, HORDE_VARIANT);
    REQUIRE(pos.check_result() == NO_RESULT);

    // 3 fold repetition draw
    pos.set("4k3/6R1/8/8/8/8/8/8 w - - 0 1", false, HORDE_VARIANT);
    mov = "g7g8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e8e7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "g8g7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e7e8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "g7g8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e8e7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "g8g7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e7e8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "g7g8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e8e7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "g8g7";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    mov = "e7e8";
    action = pos.uci_to_action(mov);
    pos.do_action(action);
    REQUIRE(pos.check_result() == DRAW);
}

TEST_CASE("3check") {
    init();
    BoardState pos;

    // Do we count checks
    pos.init(THREECHECK_VARIANT, false);
    REQUIRE("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1" == pos.fen());

    // Do we decrement checks correctly
    pos.set("1r4k1/1p2bp1p/3p2p1/PprPp2n/1R2PPq1/3Q4/1P1B1NPP/5RK1 b - - 3+3 2 22", false, THREECHECK_VARIANT);
    string move = "g4g2";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE("1r4k1/1p2bp1p/3p2p1/PprPp2n/1R2PP2/3Q4/1P1B1NqP/5RK1 w - - 3+2 0 23" == pos.fen());

    move = "g1g2";
    pos.do_action(pos.uci_to_action(move));
    move = "h5f4";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE("1r4k1/1p2bp1p/3p2p1/PprPp3/1R2Pn2/3Q4/1P1B1NKP/5R2 w - - 3+1 0 24" == pos.fen());

    pos.set("2r3k1/1p2bp1p/3p2p1/Pp1P4/1R2Pp2/7Q/1P3N1P/2r2R1K w - - 3+1 4 27", false, THREECHECK_VARIANT);
    move = "h3c8";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE("2Q3k1/1p2bp1p/3p2p1/Pp1P4/1R2Pp2/8/1P3N1P/2r2R1K b - - 2+1 0 27" == pos.fen());

    pos.set("4Rr1k/3P3p/5pp1/8/8/4p3/1P3p1P/5R1K w - - 2+1 0 39", false, THREECHECK_VARIANT);
    move = "e8f8";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE("5R1k/3P3p/5pp1/8/8/4p3/1P3p1P/5R1K b - - 1+1 0 39" == pos.fen());

    pos.set("5R2/3P2kp/5pp1/8/8/4p3/1P3p1P/5R1K w - - 1+1 1 40", false, THREECHECK_VARIANT);
    move = "f8f7";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE("8/3P1Rkp/5pp1/8/8/4p3/1P3p1P/5R1K b - - 0+1 2 40" == pos.fen());
    REQUIRE(pos.check_result() == WHITE_WIN);

    // More checks, if 3check win condition takes effect
    pos.set("8/pk6/8/p1p4p/2P1p3/8/6p1/B2Kr3 w - - 3+0 2 44", false, THREECHECK_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("8/ppp2k2/5b2/1P1n4/P3bP2/3PP1P1/5K1r/5R2 w - - 1+0 3 27", false, THREECHECK_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("8/k1R1P3/p2p4/2pP4/8/1p2P2p/P6P/K7 b - - 0+3 1 34", false, THREECHECK_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);
    pos.set("6R1/6k1/8/5P1p/5P1P/8/6K1/8 b - - 0+2 4 56", false, THREECHECK_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);
}

TEST_CASE("Racing_Kings") {
    init();
    BoardState pos;

    // Check the starting position
    pos.init(RACE_VARIANT, false);
    REQUIRE(pos.fen() == "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1");

    // Checks are forbidden
    pos.set("8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1", false, RACE_VARIANT);
    vector<string> uciMoves = {"e2c3", "e2a3"};
    REQUIRE(are_uci_moves_legal_bool(pos, uciMoves, false, false));
    pos.set("R2R4/4Q3/8/2r5/1q6/bk3N1K/2b5/8 b - - 6 13", false, RACE_VARIANT);
    REQUIRE(are_uci_moves_legal_bool(pos, {"c2f5", "b4g4", "b4h4", "c5h5"}, false, false));
    pos.set("R2R4/4Q3/8/2r5/1q6/1k3N1K/2b5/2b5 w - - 7 14", false, RACE_VARIANT);
    REQUIRE(are_uci_moves_legal_bool(pos, {"f3d2", "f3d4", "e7e6", "e7f7", "e7e3", "d8d3", "a8a3"}, false, false));

    // Win if a king reaches the 8th row
    pos.set("1bk1q3/8/8/6K1/8/8/8/R7 w - - 2 47", false, RACE_VARIANT);
    REQUIRE(pos.check_result() == BLACK_WIN);
    pos.set("6K1/8/8/6Q1/8/8/n1k5/b7 b - - 2 25", false, RACE_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // Draw, if white moves king to 8th row and black follows immediatly
    pos.set("2r2NK1/kn2R3/8/8/8/8/8/8 b - - 8 26", false, RACE_VARIANT);
    string move = "a7b8";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == DRAW);

    // Check for draw if both kings are on 8th row (e.g. bad init in selfplay)
    pos.set("1k3K2/2qQ4/8/8/8/8/8/8 w - - 30 26", false, RACE_VARIANT);
    REQUIRE(pos.check_result() == DRAW);
}

TEST_CASE("Atomic") {
    init();
    BoardState pos;

    // Check starting position
    pos.init(ATOMIC_VARIANT, false);
    REQUIRE(pos.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // The capturing and captured figure always die
    // Everything within a radius of 1 square explodes, except pawns

    pos.set("rn1qkb1r/p1p3pp/b3pp1n/3pP3/1P1P1P2/7P/P5P1/RNBQKBNR w KQkq - 1 7", false, ATOMIC_VARIANT);
    string move = "f1a6";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "rn1qkb1r/p1p3pp/4pp1n/3pP3/1P1P1P2/7P/P5P1/RNBQK1NR b KQkq - 0 7");

    pos.set("1r1qk2r/2p3pp/p1n1pp1n/1P1pP3/1b1P1P2/P6P/4Q1P1/RNB1K1NR w KQk - 1 11", false, ATOMIC_VARIANT);
    move = "a3b4";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "1r1qk2r/2p3pp/p1n1pp1n/1P1pP3/3P1P2/7P/4Q1P1/RNB1K1NR b KQk - 0 11");

    pos.set("1r2k2r/R1p3pp/7n/2npp3/1B1P1PP1/1q5P/2Q4R/1N3KN1 w k - 2 22", false, ATOMIC_VARIANT);
    move = "c2b3";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "1r2k2r/R1p3pp/7n/2npp3/3P1PP1/7P/7R/1N3KN1 b k - 0 22");

    pos.set("1r3rk1/R1p3p1/6Pp/2nppn2/3P1P2/7P/7R/1N3KN1 w - - 0 25", false, ATOMIC_VARIANT);
    move = "a7c7";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "5rk1/6p1/6Pp/2nppn2/3P1P2/7P/7R/1N3KN1 b - - 0 25");

    pos.set("2r3k1/6p1/6Pp/3pp3/3P1P2/2N3nP/1n5R/2K3N1 b - - 8 29", false, ATOMIC_VARIANT);
    move = "e5d4";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "2r3k1/6p1/6Pp/3p4/5P2/6nP/1n5R/2K3N1 w - - 0 30");

    pos.set("6k1/6p1/6Pp/3p4/5P2/6nP/Kn5R/2r3N1 b - - 3 31", false, ATOMIC_VARIANT);
    move = "c1g1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "6k1/6p1/6Pp/3p4/5P2/6nP/Kn6/8 w - - 0 32");

    // Nuking the opposite king wins the game immediatly ...
    pos.set("6k1/5Kp1/2q3P1/5n1p/5P1P/8/1n6/8 b - - 1 40", false, ATOMIC_VARIANT);
    move = "c6g6";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == BLACK_WIN);

    // ... overriding checks ...
    pos.set("8/1q6/8/8/8/5k2/1R4n1/1K6 w - - 0 1", false, ATOMIC_VARIANT);
    move = "b2g2";
    REQUIRE(is_uci_move_legal(pos, move, false));
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == WHITE_WIN);

    // ... and checkmates
    pos.set("8/1q6/r7/8/8/5k2/R5n1/K7 w - - 0 1", false, ATOMIC_VARIANT);
    move = "a2g2";
    REQUIRE(is_uci_move_legal(pos, move, false));
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == WHITE_WIN);

    // Checkmating the king wins the game (the mating figures cannot be captured by the king)
    pos.set("3q4/6Qk/4r3/p7/6Pp/7P/8/1R2R1K1 b - - 8 29", false, ATOMIC_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);
    // Mating figures cannot be captured by other figures, if the king would explode
    pos.set("8/kQ3r2/6p1/2P3Pp/7P/4p3/1K2B3/n7 b - - 3 39", false, ATOMIC_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // Kings are not allowed to attack each other
    pos.set("6k1/6p1/4K1Pp/5n2/5P2/7P/1n6/3q4 w - - 0 37", false, ATOMIC_VARIANT);
    move = "e6f7";
    REQUIRE(is_uci_move_legal(pos, move, false));
    pos.set("5Kk1/6p1/2q3Pp/5n2/5P1P/8/1n6/8 b - - 2 39", false, ATOMIC_VARIANT);
    move = "g8f8";
    REQUIRE(is_uci_move_legal(pos, move, false) == false);
}

TEST_CASE("Chess960") {
    init();
    BoardState pos;

    // Check chess960 start pos
    for(int i = 0; i < 20; ++i) {
        pos.init(CHESS_VARIANT, true);
        std::istringstream ss(pos.fen());
        string token;
        ss >> token;

        // Symmetry
        string firstRank = token.substr(0, 8);
        string lastRank = token.substr(35, 43);
        std::transform(lastRank.begin(), lastRank.end(), lastRank.begin(), ::tolower);
        REQUIRE(firstRank == lastRank);

        // Rooks are on each side of the king
        int posFirstRook = firstRank.find_first_of('r');
        int posLastRook = firstRank.find_last_of('r');
        int posKing = firstRank.find('k');
        REQUIRE(posFirstRook < posKing);
        REQUIRE(posKing < posLastRook);

        // Bishops are on white and black squares
        int posFirstBishop = firstRank.find_first_of('b');
        int posLastBishop = firstRank.find_last_of('b');
        int bishopDiff = posLastBishop - posFirstBishop;
        REQUIRE((bishopDiff % 2) != 0);

        // All other cases are general fen test cases
        // Maybe test somewhere else, if the first and last rank are upper / lower case, etc.
    }

    // Are castling moves legal
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    vector<string> moves = {"e1d1", "e1g1"};
    REQUIRE(are_uci_moves_legal_bool(pos, moves, true, true));

    // Not legal, if squares between king and rook aren't empty
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBRQ w GDgd - 0 4", true, CHESS_VARIANT);
    string move = "e1g1";
    REQUIRE(is_uci_move_legal(pos, move, true) == false);

    // Not legal through check
    pos.set("1nnrkbrq/p1pppppp/1p6/1b6/4P3/8/PPPP1PPP/BN1RK1RQ w GDgd - 0 4", true, CHESS_VARIANT);
    move = "e1g1";
    REQUIRE(is_uci_move_legal(pos, move, true) == false);

    // Not legal if rook is in check
    pos.set("bnnrk1rq/p1pp1ppp/1p2p3/2b5/4P3/5P2/PPPP2PP/BN1RK1RQ w GDgd - 0 4", true, CHESS_VARIANT);
    move = "e1g1";
    REQUIRE(is_uci_move_legal(pos, move, true) == false);

    // Not legal if king is in check
    pos.set("bnnrkbr1/ppppppp1/8/4q2p/8/5P2/PPPP2PP/BN1RK1RQ w GDgd - 0 4", true, CHESS_VARIANT);
    moves = {"e1d1", "e1g1"};
    REQUIRE(are_uci_moves_legal_bool(pos, moves, false, true));

    // Not legal, if the destination squares aren't empty
    pos.set("nrbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/NRBBQNKR w HBhb - 0 4", true, CHESS_VARIANT);
    move = "g1h1";
    REQUIRE(is_uci_move_legal(pos, move, true) == false);
    pos.set("bnnrkbrq/pppppppp/8/8/8/4P3/PPPP1PPP/BNNRK1RQ w GDgd - 0 4", true, CHESS_VARIANT);
    move = "e1d1";
    REQUIRE(is_uci_move_legal(pos, move, true) == false);

    // We can only castle, if fen says yes
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w gd - 0 1", true, CHESS_VARIANT);
    moves = {"e1d1", "e1g1"};
    REQUIRE(are_uci_moves_legal_bool(pos, moves, false, true));

    // If we move a rook, the fen castling should get removed
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    move = "d1c1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BNR1K1RQ b Ggd - 1 1");

    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    move = "g1f1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RKR1Q b Dgd - 1 1");

    // If we move king, castling disappears
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    move = "e1f1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1R1KRQ b gd - 1 1");

    // 'kingside' castling always lands on c1/d1
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    move = "e1d1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BNKR2RQ b gd - 1 1");

    // 'queenside' castling always lands on f1/g1
    pos.set("bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1RK1RQ w GDgd - 0 1", true, CHESS_VARIANT);
    move = "e1g1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "bnnrkbrq/pppppppp/8/8/8/8/PPPPPPPP/BN1R1RKQ b gd - 1 1");

    pos.set("nrbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/NR4KR w HBhb - 0 1", true, CHESS_VARIANT);
    move = "g1h1";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.fen() == "nrbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/NR3RK1 b hb - 1 1");

    // TODO: Why is this important?
    vector<string> castlingMoves = create_castling_moves(true);
    REQUIRE(are_all_entries_true(castlingMoves, is_960_castling_candidate_move) == true);
}

TEST_CASE("Antichess") {
    init();
    BoardState pos;

    // Check starting fen, esp. because there is no castling
    pos.init(ANTI_VARIANT, false);
    REQUIRE(pos.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");

    // If we can, we must capture. If there are multiple captures, we may choose
    pos.set("rnb1kbnr/pp1ppppp/8/q1p5/8/2P1P3/PP1PNPPP/RNBQKB1R b - - 0 3", false, ANTI_VARIANT);
    vector<string> moves = {"a5a2", "a5c3"};
    REQUIRE(legal_actions_equal_ucimoves(pos, moves, false));

    // Figures can be moved, even if the king is in check, even checkmate:
    pos.set("2Q1kb1r/3ppp1p/r4np1/p7/8/P1P1P1P1/4NP1P/RNB1KB1R b - - 0 11", false, ANTI_VARIANT);
    REQUIRE(pos.check_result() == NO_RESULT);
    REQUIRE(is_uci_move_legal(pos, "a6a8", false));

    // The king can be captured
    pos.set("r1Q1kb1r/3ppp1p/5np1/p7/8/P1P1P1P1/4NP1P/RNB1KB1R w - - 1 12", false, ANTI_VARIANT);
    REQUIRE(is_uci_move_legal(pos, "c8e8", false));
    string move = "c8e8";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == NO_RESULT);

    // The game is over, if a player lost all pieces
    pos.set("8/8/6p1/7q/8/8/8/8 w - - 0 39", false, ANTI_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // If a player has no legal moves (and still figures), he wins (no stalemate !!)
    pos.set("5b2/4p3/1p3p2/1P5p/8/8/8/1r6 w - - 0 36", false, ANTI_VARIANT);
    REQUIRE(pos.check_result() == WHITE_WIN);

    // King promotion is legal
    pos.set("2Q5/8/8/8/R6P/2B5/2pP4/8 b - - 1 35", false, ANTI_VARIANT);
    REQUIRE(is_uci_move_legal(pos, "c2c1k", false));

    // Castling is not permitted
    pos.set("r3kbnr/p2pp1pp/bp3p2/8/3P4/P1P5/1B1P1PPP/RN1QK2R b - - 0 9", false, ANTI_VARIANT);
    REQUIRE(is_uci_move_legal(pos, "e8c8", false) == false);
}

TEST_CASE("Lichess Crazyhouse") {
    init();
    BoardState pos;

    // Check pocket notation
    pos.init(CRAZYHOUSE_VARIANT, false);
    REQUIRE(pos.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");

    // Captured pieces can be dropped
    pos.set("1k1r3r/pppb1p2/2nbqn1p/3p2p1/3PP1P1/3Q1PP1/PPN2NBP/R1B2RK1[p] b - - 0 12", false, CRAZYHOUSE_VARIANT);
    REQUIRE(is_uci_move_legal(pos, "P@c4", false));

    // Pieces cannot be dropped on other pieces
    vector<string> moves = {"P@d5", "P@d4", "P@e4"};
    REQUIRE(are_uci_moves_legal_bool(pos, moves, false, false));

    // Dropping pawns on the 1. and 8. rank is prohibited
    moves = {"P@b1", "P@d1", "P@e1", "P@e8", "P@f8", "P@g8"};
    REQUIRE(are_uci_moves_legal_bool(pos, moves, false, false));

    // Checkmating with a drop is allowed
    pos.set("4R2b/1N3rkb/1p2P1pp/p2P4/2P1P3/8/PP4Q1/3R3K[QRBBNNNPPPPpp] w - - 2 53", false, CRAZYHOUSE_VARIANT);
    REQUIRE(is_uci_move_legal(pos, "N@h5", false));
    string move = "N@h5";
    pos.do_action(pos.uci_to_action(move));
    REQUIRE(pos.check_result() == WHITE_WIN);

    // If you have pieces in the pocket, you can prevent certain checks/checkmates by dropping pieces between
    pos.set("r2qk3/1pP2r1n/p1nP4/8/3P1Bb1/2Pp1PP1/PPp2PP1/3q1K1R[Bbnnppr] w - - 2 29", false, CRAZYHOUSE_VARIANT);
    REQUIRE(pos.check_result() == NO_RESULT);
    REQUIRE(is_uci_move_legal(pos, "B@e1", false));
}


#endif //MODE_LICHESS

#else // MODE_STRATEGO
TEST_CASE("Build tests") {
    REQUIRE(true);
}
#endif

#ifdef MODE_CRAZYHOUSE
TEST_CASE("Crazyhouse Input Planes V1") {
    init();
    int variant = StateConstants::variant_to_int("crazyhouse");
    BoardState state;
    const uint nbValuesTotal = 34 * StateConstants::NB_SQUARES();
    state.init(variant, false);
    // starting position test
    PlaneStatistics stats = get_planes_statistics(state, false, make_version<1,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 416);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 746928);
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"));

    state.set("5r2/ppp2pkp/3p4/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[PBRQnbb] w - - 0 28", false, variant);
    string move = "Q@f6";
    state.do_action(state.uci_to_action(move));
    move = "g7g8";
    state.do_action(state.uci_to_action(move));
    move = "R@h8";
    state.do_action(state.uci_to_action(move));

    stats = get_planes_statistics(state, false, make_version<1,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 2395);
    REQUIRE(stats.maxNum == 29);
    REQUIRE(stats.key == 4170903);
    REQUIRE(stats.argMax == 1792);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));

    stats = get_planes_statistics(state, true, make_version<1,0,0>(), nbValuesTotal);
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(45.512, 0.001));
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(37011.632, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));
}

TEST_CASE("Crazyhouse Input Planes V2") {
    init();
    int variant = StateConstants::variant_to_int("crazyhouse");
    BoardState state;
    const uint nbValuesTotal = 51 * StateConstants::NB_SQUARES();
    state.init(variant, false);
    // starting position test
    PlaneStatistics stats = get_planes_statistics(state, false, make_version<2,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 416);
    REQUIRE(stats.maxNum == 1);
    REQUIRE(stats.key == 746928);
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"));

    state.set("5r2/ppp2pkp/3p4/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[PBRQnbb] w - - 0 28", false, variant);
    string move = "Q@f6";
    state.do_action(state.uci_to_action(move));
    move = "g7g8";
    state.do_action(state.uci_to_action(move));
    move = "R@h8";
    state.do_action(state.uci_to_action(move));

    stats = get_planes_statistics(state, false, make_version<2,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 2399);
    REQUIRE(stats.maxNum == 29);
    REQUIRE(stats.key == 4180615);
    REQUIRE(stats.argMax == 1792);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));

    stats = get_planes_statistics(state, true, make_version<2,0,0>(), nbValuesTotal);
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(49.512, 0.001));
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(46723.632, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));
}

TEST_CASE("Crazyhouse Input Planes V3") {
    init();
    int variant = StateConstants::variant_to_int("crazyhouse");
    BoardState state;
    const uint nbValuesTotal = 64 * StateConstants::NB_SQUARES();
    state.init(variant, false);
    // starting position test
    PlaneStatistics stats = get_planes_statistics(state, false, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 1312);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3430384);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(state.fen() == string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"));

    state.set("5r2/ppp2pkp/3p4/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[PBRQnbb] w - - 0 28", false, variant);
    string move = "Q@f6";
    state.do_action(state.uci_to_action(move));
    move = "g7g8";
    state.do_action(state.uci_to_action(move));
    move = "R@h8";
    state.do_action(state.uci_to_action(move));

    stats = get_planes_statistics(state, false, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 1307);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3700213);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));

    stats = get_planes_statistics(state, true, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(193.8, 0.001));
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(474696, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("5rkR/ppp2p1p/3p1Q2/2bP4/2Pnp1N1/3P2pP/PP2n1P1/R2Q1R1K[BPbbn] b - - 3 29"));
}
#endif

#ifdef MODE_LICHESS
TEST_CASE("Atomic Input Planes V3") {
    init();
    int variant = StateConstants::variant_to_int("atomic");
    BoardState state;
    const uint nbValuesTotal = 80 * StateConstants::NB_SQUARES();
    state.init(variant, false);
    // starting position test
    PlaneStatistics stats = get_planes_statistics(state, false, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 1440);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 5932976);
    REQUIRE(stats.argMax == 4736);
    REQUIRE(state.fen() == string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    state.set("r2qk2r/p6p/3p1ppb/3Pp1BP/1p2P1b1/3Q1P2/PPP3P1/R3K2R b KQkq - 1 14", false, variant);
    string move = "d8b6";
    state.do_action(state.uci_to_action(move));
    move = "d3b5";
    state.do_action(state.uci_to_action(move));

    stats = get_planes_statistics(state, false, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 1433);
    REQUIRE(stats.maxNum == 7);
    REQUIRE(stats.key == 5420051);
    REQUIRE(stats.argMax == 4736);
    REQUIRE(state.fen() == string("r3k2r/p6p/1q1p1ppb/1Q1Pp1BP/1p2P1b1/5P2/PPP3P1/R3K2R b KQkq - 3 15"));

    stats = get_planes_statistics(state, true, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(516.84, 0.001));
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(1470726.04, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("r3k2r/p6p/1q1p1ppb/1Q1Pp1BP/1p2P1b1/5P2/PPP3P1/R3K2R b KQkq - 3 15"));
}
#endif

#ifdef MODE_CHESS
TEST_CASE("Chess960 Input Planes V3") {
    init();
    int variant = StateConstants::variant_to_int("chess");
    BoardState state;
    state.set("b1qnrnkr/p2ppppp/1p6/2p1b3/2P5/4N1P1/PP1PPP1P/BBQNRK1R b he - 1 4", true, variant);
    const uint nbValuesTotal = 52 * StateConstants::NB_SQUARES();

    vector<float> planes(nbValuesTotal);
    state.get_state_planes(false, planes.data(), make_version<3,0,0>());

    // custom position test
    PlaneStatistics stats = get_planes_statistics(state, false, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE(stats.sum == 1312);
    REQUIRE(stats.maxNum == 8);
    REQUIRE(stats.key == 3512322);
    REQUIRE(stats.argMax == 3008);
    REQUIRE(state.fen() == string("b1qnrnkr/p2ppppp/1p6/2p1b3/2P5/4N1P1/PP1PPP1P/BBQNRK1R b he - 1 4"));

    // normalize = true test
    stats = get_planes_statistics(state, true, make_version<3,0,0>(), nbValuesTotal);
    REQUIRE_THAT(stats.sum, Catch::Matchers::WithinRel(409.28, 0.001));
    REQUIRE(stats.maxNum == 1);
    REQUIRE_THAT(stats.key, Catch::Matchers::WithinRel(823554.8, 0.001));
    REQUIRE(stats.argMax == 8);
    REQUIRE(state.fen() == string("b1qnrnkr/p2ppppp/1p6/2p1b3/2P5/4N1P1/PP1PPP1P/BBQNRK1R b he - 1 4"));
}
#endif

#endif

