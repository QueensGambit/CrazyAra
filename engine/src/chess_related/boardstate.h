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
#include "outputrepresentation.h
using namespace std;


class StateConstantsBoard : public StateConstantsInterface<StateConstantsBoard>
{
private:
    static action_idx_map MV_LOOKUP;
    static action_idx_map MV_LOOKUP_MIRRORED;
    static action_idx_map MV_LOOKUP_CLASSIC;
    static action_idx_map MV_LOOKUP_MIRRORED_CLASSIC;
public:
    static int BOARD_WIDTH() {
        return 8;
    }
    static int BOARD_HEIGHT() {
        return 8;
    }
    static int NB_CHANNELS_TOTAL() {
        return NB_CHANNELS_POS() + NB_CHANNELS_CONST() + NB_CHANNELS_VARIANTS() + NB_CHANNELS_HISTORY();
    }
    static int NB_LABELS() {
        // legal moves total which are represented in the NN
        #ifdef MODE_CRAZYHOUSE
        return 2272;
        #elif defined MODE_LICHESS
        return 2316;
        #else  // MODE = MODE_CHESS
        return 1968;
        #endif
    }
    static int NB_LABELS_POLICY_MAP() {
        return NB_CHANNELS_POLICY_MAP() * BOARD_HEIGHT() * BOARD_WIDTH();
    }
    static int NB_PLAYERS() {
        return 2;
    }
    static std::string action_to_uci(Action action, bool is960) {
        return UCI::move(Move(action), is960);
    }
    template<PolicyType p, MirrorType m>
    static size_t action_to_index(Action action) {
        switch (p) {
        case normal:
            switch (m) {
            case notMirrored:
                return MV_LOOKUP[action];
            case mirrored:
                return MV_LOOKUP_MIRRORED[action];
            default:
                return MV_LOOKUP[action];
            }
        case classic:
            switch (m) {
            case notMirrored:
                return MV_LOOKUP_CLASSIC[action];
            case mirrored:
                return MV_LOOKUP_MIRRORED_CLASSIC[action];
            default:
                return MV_LOOKUP_CLASSIC[action];
            }
        default:
            return MV_LOOKUP[action];
        }
    }
    static void init(bool isPolicyMap) {
        init_policy_constants(isPolicyMap,
                              MV_LOOKUP,
                              MV_LOOKUP_MIRRORED,
                              MV_LOOKUP_CLASSIC,
                              MV_LOOKUP_MIRRORED_CLASSIC);
    }
    // -------------------------------------------------
    // |           Additional custom methods           |
    // -------------------------------------------------
#ifdef MODE_CRAZYHOUSE
    static int NB_CHANNELS_POS() {
        return 27;
    }
    static int NB_CHANNELS_CONST() {
        return 7;
    }
    static int NB_CHANNELS_VARIANTS() {
        return 0;
    }
    static int NB_LAST_MOVES() {
        return 0;
    }
    static int NB_CHANNELS_PER_HISTORY() {
        return 0;
    }
#elif defined MODE_LICHESS
    static int NB_CHANNELS_POS() {
        return 27;
    }
    static int NB_CHANNELS_CONST() {
        return 11;
    }
    static int NB_CHANNELS_VARIANTS() {
        return 9;
    }
    static int NB_LAST_MOVES() {
        return 8;
    }
    static int NB_CHANNELS_PER_HISTORY() {
        return 2;
    }
#elif defined MODE_CHESS
    static int NB_CHANNELS_POS() {
        return 15;
    }
    static int NB_CHANNELS_CONST() {
        return 7;
    }
    static int NB_CHANNELS_VARIANTS() {
        return 1;
    }
    static int NB_LAST_MOVES() {
        return 8;
    }
    static int NB_CHANNELS_PER_HISTORY() {
        return 2;
    }
#endif
    static int NB_CHANNELS_HISTORY() {
        return NB_LAST_MOVES() * NB_CHANNELS_PER_HISTORY();
    }
    // the number of different piece types in the game
    static int NB_PIECE_TYPES() {
        return 6;
    }
    // define the number of different pieces one can have in his pocket (the king is excluded)
    static int POCKETS_SIZE_PIECE_TYPE() {
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
    static int NB_CHANNELS_POLICY_MAP() {
        #ifdef MODE_CRAZYHOUSE
        return 81;
        #elif defined MODE_LICHESS
        return 84;
        #else  // MODE = MODE_CHESS
        return 76;
        #endif
    }
    #ifdef MODE_LICHESS
    static std::unordered_map<Variant, int> CHANNEL_MAPPING_VARIANTS() {
         {  {CHESS_VARIANT, 1},
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
    vector<Action> legal_actions() const override;
    void set(const string &fenStr, bool isChess960, int variant) override;
    void get_state_planes(bool normalize, float *inputPlanes) const override;
    unsigned int steps_from_null() const override;
    bool is_chess960() const override;
    string fen() const override;
    void do_action(Action action) override;
    void undo_action(Action action) override;
    unsigned int number_repetitions() const override;
    int side_to_move() const override;
    Key hash_key() const override;
    void flip() override;
    Action uci_to_action(string& uciStr) const override;
    string action_to_san(Action action, const vector<Action>& legalActions, bool leadsToWin=false, bool bookMove=false) const override;
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const override;
    Result check_result(bool inCheck) const override;
    bool gives_check(Action action) const override;
    void print(ostream& os) const override;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result) override;
    BoardState* clone() const override;
};

#endif // BOARTSTATE_H
#endif
