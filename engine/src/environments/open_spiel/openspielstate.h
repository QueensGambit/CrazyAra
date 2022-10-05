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
 * @file: openspielstate.h
 * Created on 27.12.2020
 * @author: queensgambit
 *
 * Wrapper for the OpenSpiel environment (https://github.com/deepmind/open_spiel/tree/master/open_spiel/games)
 */

#ifndef OPENSPIELSTATE_H
#define OPENSPIELSTATE_H

#include "state.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/games/hex.h"
<<<<<<< HEAD
#include "open_spiel/games/tic_tac_toe.h"
=======
#include "open_spiel/games/dark_hex.h"
>>>>>>> origin/master

namespace open_spiel {
namespace gametype {
enum SupportedOpenSpielVariants : uint8_t {
    TICTACTOE = 0,
    CONNECTFOUR = 1,
    HEX = 2,  // 11x11 board
    CHESS = 3,
    YORKTOWN = 4,
};
}
}

class StateConstantsOpenSpiel : public StateConstantsInterface<StateConstantsOpenSpiel>
{
public:
    static uint BOARD_WIDTH() {
        return 3;
//        return open_spiel::chess::kDefaultBoardSize;
    }
    static uint BOARD_HEIGHT() {
         return 3;
//        return  open_spiel::chess::kDefaultBoardSize;
    }
    static uint NB_CHANNELS_TOTAL() {
        return 1;  // TODO
    }
    static uint NB_LABELS() {
        return 9;  // TODO
    }
    static uint NB_LABELS_POLICY_MAP() {
        return 9;  // TODO

    }
    static uint NB_AUXILIARY_OUTPUTS() {
        return 0U;
    }
    static int NB_PLAYERS() {
        return  open_spiel::hex::kNumPlayers;
    }
    static std::string action_to_uci(Action action, bool is960) {
        // TODO use actual uci for this
        return std::to_string(action);
    }
    template<PolicyType p = normal, MirrorType m = notMirrored>
    static MoveIdx action_to_index(Action action) {
        return action;
    }
    static void init(bool isPolicyMap) {
        return; // pass
    }

    static std::vector<std::string> available_variants() {
        return {"tic_tac_toe",
                "connect_four",
                "hex",
                "chess",
                "yorktown"};
    }

    static std::string start_fen(int variant) {
        switch (variant) {
        case open_spiel::gametype::SupportedOpenSpielVariants::TICTACTOE:
            return "... ... ...";
        case open_spiel::gametype::SupportedOpenSpielVariants::CONNECTFOUR:
            return "....... ....... ....... ....... ....... .......";
        case open_spiel::gametype::SupportedOpenSpielVariants::HEX:
            return ". . . . . . . . . . .  . . . . . . . . . . .   . . . . . . . . . . .    . . . . . . . . . . .     . . . . . . . . . . .      . . . . . . . . . . .       . . . . . . . . . . .        . . . . . . . . . . .         . . . . . . . . . . .          . . . . . . . . . . .           . . . . . . . . . . .";
        case open_spiel::gametype::SupportedOpenSpielVariants::CHESS:
            return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        case open_spiel::gametype::SupportedOpenSpielVariants::YORKTOWN:
            return "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
        default:
            info_string("Unknown variant:", variant, "given");
            return "";
        }
    }
};

class OpenSpielState : public State
{
private:
    open_spiel::gametype::SupportedOpenSpielVariants currentVariant;
    std::shared_ptr<const open_spiel::Game> spielGame;
    std::unique_ptr<open_spiel::State> spielState;

    /**
     * @brief check_variant Checks the given variant against the current active variant and loads a new game type if necessary.
     * @param variant Variant specification
     */
    inline void check_variant(int variant);

public:
    OpenSpielState();
    OpenSpielState(const OpenSpielState& openSpielState);

    // State interface
public:
    std::vector<Action> legal_actions() const;
    void set(const std::string &fenStr, bool isChess960, int variant);
    void get_state_planes(bool normalize, float *inputPlanes, Version version) const;
    unsigned int steps_from_null() const;
    bool is_chess960() const;
    std::string fen() const;
    void do_action(Action action);
    void undo_action(Action action);
    void prepare_action();
    unsigned int number_repetitions() const;
    int side_to_move() const;
    Key hash_key() const;
    void flip();
    Action uci_to_action(std::string &uciStr) const;
    std::string action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const;
    TerminalType is_terminal(size_t numberLegalMoves, float &customTerminalValue) const;
    bool gives_check(Action action) const;
    void print(std::ostream &os) const;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result);
    void set_auxiliary_outputs(const float* auxiliaryOutputs);
    OpenSpielState *clone() const;
    void init(int variant, bool isChess960);
};

#endif // OPENSPIELSTATE_H
