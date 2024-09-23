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
  GNU General Public License fåor more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: boardstate.h
 * Created on 13.07.2020
 * @author: queensgambit
 *
 * BoardState encapsulates Board and StateInfo and inherits from the abstract State class.
 */

#ifndef BOARTSTATE_H
#define BOARTSTATE_H

#ifndef MODE_POMMERMAN
#include "uci.h"
#include "../state.h"
#include "board.h"
#include "outputrepresentation.h"
using namespace std;


class StateConstantsBoard : public StateConstantsInterface<StateConstantsBoard>
{
public:
    static uint BOARD_WIDTH() {
        return 8;
    }
    static uint BOARD_HEIGHT() {
        return 8;
    }
    static uint NB_CHANNELS_TOTAL() {
        return NB_CHANNELS_POS() + NB_CHANNELS_CONST() + NB_CHANNELS_VARIANTS() + NB_CHANNELS_HISTORY() + NB_CHANNELS_AUXILIARY();
    }
    static uint NB_LABELS() {
        // legal moves total which are represented in the NN
#ifdef MODE_CRAZYHOUSE
        return 2272;
#elif defined MODE_LICHESS
        return 2316;
#else  // MODE = MODE_CHESS
        return 1968;
#endif
    }
    static uint NB_LABELS_POLICY_MAP() {
        return NB_CHANNELS_POLICY_MAP() * BOARD_HEIGHT() * BOARD_WIDTH();
    }
    static uint NB_AUXILIARY_OUTPUTS() {
        return 0U;
    }
    static uint NB_PLAYERS() {
        return 2;
    }
    static std::string action_to_uci(Action action, bool is960) {
        return UCI::move(Move(action), is960);
    }
    template<PolicyType p, MirrorType m>
    static MoveIdx action_to_index(Action action) {
        switch (p) {
        case normal:
            switch (m) {
            case notMirrored:
                return OutputRepresentation::MV_LOOKUP[action];
            case mirrored:
                return OutputRepresentation::MV_LOOKUP_MIRRORED[action];
            default:
                return OutputRepresentation::MV_LOOKUP[action];
            }
        case classic:
            switch (m) {
            case notMirrored:
                return OutputRepresentation::MV_LOOKUP_CLASSIC[action];
            case mirrored:
                return OutputRepresentation::MV_LOOKUP_MIRRORED_CLASSIC[action];
            default:
                return OutputRepresentation::MV_LOOKUP_CLASSIC[action];
            }
        default:
            return OutputRepresentation::MV_LOOKUP[action];
        }
    }
    static void init(bool isPolicyMap, bool is960) {
        OutputRepresentation::init_labels();
        OutputRepresentation::init_policy_constants(isPolicyMap, is960);
    }
    // -------------------------------------------------
    // |           Additional custom methods           |
    // -------------------------------------------------
#ifdef MODE_CRAZYHOUSE
    static uint NB_CHANNELS_POS() {
        return 27;
    }
    static uint NB_CHANNELS_CONST() {
        return 7;
    }
    static uint NB_CHANNELS_VARIANTS() {
        return 0;
    }
#if VERSION == 1
    static uint NB_LAST_MOVES() {
        return 0;
    }
    static uint NB_CHANNELS_PER_HISTORY() {
        return 0;
    }
#else  // VERSION == 2 || VERSION == 3
    static uint NB_LAST_MOVES() {
        return 8;
    }
    static uint NB_CHANNELS_PER_HISTORY() {
        return 2;
    }
#endif
    static uint NB_CHANNELS_AUXILIARY() {
        return 0;
    }
#elif defined MODE_LICHESS
    static uint NB_CHANNELS_POS() {
        return 27;
    }
    static uint NB_CHANNELS_CONST() {
        return 11;
    }
    static uint NB_CHANNELS_VARIANTS() {
        return 9;
    }
    static uint NB_LAST_MOVES() {
        return 8;
    }
    static uint NB_CHANNELS_PER_HISTORY() {
        return 2;
    }
    static uint NB_CHANNELS_AUXILIARY() {
        return 0;
    }
#elif defined MODE_CHESS
    static uint NB_CHANNELS_POS() {
#if VERSION == 2
        return 13;
#endif
        return 15;  // VERSION == 1 || VERSION == 3
    }
    static uint NB_CHANNELS_CONST() {
#if VERSION == 1
        return 7;
#endif
#if VERSION == 2
        return 4;
#endif
        return 5;  // VERSION == 3
    }
    static uint NB_CHANNELS_VARIANTS() {
        return 1;
    }
    static uint NB_LAST_MOVES() {
#if VERSION == 2
        return 1;
#endif
        return 8;  // VERSION == 1 or VERSION == 3
    }
    static uint NB_CHANNELS_PER_HISTORY() {
        return 2;
    }
    static uint NB_CHANNELS_AUXILIARY() {
#if VERSION == 1
        return 0;
#endif
#if VERSION == 2
#if SUB_VERSION == 7
    return 13;
#elif SUB_VERSION == 8
    return 18;
#endif
#endif
    return 15;  // VERSION == 3
    }
#endif
    static uint NB_CHANNELS_HISTORY() {
        return NB_LAST_MOVES() * NB_CHANNELS_PER_HISTORY();
    }
    // the number of different piece types in the game
    static uint NB_PIECE_TYPES() {
        return 6;
    }
    // define the number of different pieces one can have in his pocket (the king is excluded)
    static uint POCKETS_SIZE_PIECE_TYPE() {
        return 5;
    }
    //  (this used for normalization the input planes and setting an appropriate integer representation (e.g. int16)
    // these are defined as float to avoid integer division
#ifdef MODE_CRAZYHOUSE
    // define the maximum number of pieces of each type in a pocket
    static float MAX_NB_PRISONERS() {
        return 32;
    }
    // 500 was set as the max number of total moves
    static float MAX_FULL_MOVE_COUNTER() {
        return 500;
    }
    // originally this was set to 40, but actually it is meant to be 50 move rule
    static float MAX_NB_NO_PROGRESS() {
        return 40;
    }
#else  // MODE = MODE_LICHESS or MODE = MODE_CHESS:
    // at maximum you can have only 16 pawns (your own and the ones of the opponent)
    static float MAX_NB_PRISONERS() {
        return 16;
    }
    // 500 was set as the max number of total moves
    static float MAX_FULL_MOVE_COUNTER() {
        return 500;
    }
    // after 50 moves of no progress the 50 moves rule for draw applies
    static float MAX_NB_NO_PROGRESS() {
        return 50;
    }
#endif
    // normalize the relative material by 8
    static float NORMALIZE_PIECE_NUMBER() {
        return 8;
    }
    // normalize the nubmer of attackers by 4
    static float NORMALIZE_ATTACKERS() {
        return 4;
    }
    // normalize the number of legal moves
    static float NORMALIZE_MOBILITY() {
        return 64;
    }
    static uint NB_CHANNELS_POLICY_MAP() {
#ifdef MODE_CRAZYHOUSE
        return 81;
#elif defined MODE_LICHESS
        return 84;
#else  // MODE = MODE_CHESS
        return 76;
#endif
    }
inline static constexpr Version CURRENT_VERSION() {
#if VERSION == 2
#if SUBVERSION == 7
    return make_version<2,7,0>();
#elif SUB_VERSION == 8
    return make_version<2,8,0>();
#endif
#endif
#if VERSION == 3
    return make_version<3,0,0>();
#endif
    return make_version<0,0,0>();
    }
#ifdef MODE_LICHESS
    static std::unordered_map<Variant, int> CHANNEL_MAPPING_VARIANTS() {
        return {{CHESS_VARIANT, 1},
            {CRAZYHOUSE_VARIANT, 2},
            {KOTH_VARIANT, 3},
            {THREECHECK_VARIANT, 4},
            {ANTI_VARIANT, 5},
            {ATOMIC_VARIANT, 6},
            {HORDE_VARIANT, 7},
            {RACE_VARIANT, 8}
        };
    }
#endif

    static std::vector<std::string> available_variants() {
        // list of all current available variants for MultiAra
        return {
        #if defined(MODE_CHESS) || defined(MODE_LICHESS)
            "chess",
            "standard",
        #if defined(SUPPORT960)
            "fischerandom",
            "chess960",
        #endif // SUPPORT960
        #endif // MODE_CHESS && MODE_LICHESS
        #if defined(MODE_CRAZYHOUSE) || defined(MODE_LICHESS)
            "crazyhouse",
        #endif
        #ifdef MODE_LICHESS
            "kingofthehill",
            "atomic",
            "antichess",
            "horde",
            "racingkings",
            "3check",
            "threecheck", // 3check
        #endif
        };
    }

    static int variant_to_int(const std::string& variant) {
        return UCI::variant_from_name(variant);
    }

    static std::string start_fen(int variant) {
        const static string startFENs[SUBVARIANT_NB] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #ifdef ANTI
            // "The king has no royal power and accordingly:
            // it may be captured like any other piece
            // there is no check or checkmate there is no castling"
            // -- https://en.wikipedia.org/wiki/Losing_chess
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            #endif
            #ifdef ATOMIC
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef CRAZYHOUSE
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
            #endif
            #ifdef EXTINCTION
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef GRID
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef HORDE
            "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
            #endif
            #ifdef KOTH
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef LOSERS
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef RACE
            "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1",
            #endif
            #ifdef THREECHECK
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
            #endif
            #ifdef TWOKINGS
            "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
            #endif
            #ifdef SUICIDE
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            #endif
            #ifdef BUGHOUSE
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
            #endif
            #ifdef DISPLACEDGRID
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef LOOP
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
            #endif
            #ifdef PLACEMENT
            "8/pppppppp/8/8/8/8/PPPPPPPP/8[KQRRBBNNkqrrbbnn] w - -",
            #endif
            #ifdef SLIPPEDGRID
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            #endif
            #ifdef TWOKINGSSYMMETRIC
            "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
            #endif
        };
        return startFENs[variant];
    }

    static int DEFAULT_VARIANT() {
#ifdef MODE_CRAZYHOUSE
        return CRAZYHOUSE_VARIANT;  // == 1
#else
        return CHESS_VARIANT;  // == 0
#endif
    }

};

class BoardState : public State
{
private:
    Board board;
    StateListPtr states;
public:
    BoardState();
    BoardState(const BoardState& b);

    // State interface
    bool mirror_policy(SideToMove sideToMove) const;
    vector<Action> legal_actions() const override;
    void set(const string &fenStr, bool isChess960, int variant) override;
    void get_state_planes(bool normalize, float *inputPlanes, Version version) const override;
    unsigned int steps_from_null() const override;
    bool is_chess960() const override;
    string fen() const override;
    void do_action(Action action) override;
    void undo_action(Action action) override;
    void prepare_action() override;
    unsigned int number_repetitions() const override;
    int side_to_move() const override;
    Key hash_key() const override;
    void flip() override;
    Action uci_to_action(string& uciStr) const override;
    string action_to_san(Action action, const vector<Action>& legalActions, bool leadsToWin=false, bool bookMove=false) const override;
    TerminalType is_terminal(size_t numberLegalMoves, float& customTerminalValue) const override;
    bool gives_check(Action action) const override;
    void print(ostream& os) const override;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result) override;
    void set_auxiliary_outputs(const float* auxiliaryOutputs) override;
    BoardState* clone() const override;
    void init(int variant, bool isChess960) override;
    GamePhase get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const override;
};

#endif // BOARTSTATE_H
#endif
